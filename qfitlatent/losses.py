# created by clay
'''
Training objectives for the qFit dynamics VAE: the reconstruction NLL, the VAE KL,
and the auxiliary terms that shape the predicted rotamer mixture (entropy, Sinkhorn
OT, occupancy matching, slot cross-entropy, stray mass, atom-space set loss,
confidence).
'''

import math

import numpy as np
import torch

from .featurize import N_CHI, ONE_TO_IDX
from .rotamers import LOG_ZERO
from .sidechain import rebuild as sc_rebuild

# minimum von mises concentration to keep kappa > 0 and i0e well behaved
KAPPA_MIN = 1e-2


def ensemble_nll(log_p, mu, sigma, chi_angles, occupancies, valid, idx=None):
    '''
    Occupancy weighted negative log likelihood of the observed qFit altlocs under
    the predicted rotamer mixture.

        log p(𝝌) = logsumexp_s [ log p_s + Sum_c log vM(𝝌_c | mu_sc, kappa_sc) ]

    The concentration comes from the predicted standard deviations, kappa = 1/sigma^2.

    Inputs:
        - log_p: [N, S] predicted log weights over the mixture components
        - mu, sigma: [N, S, N_CHI] component chi means / standard deviations (radians)
        - chi_angles: [N, A, N_CHI], occupancies: [N, A]
        - valid: [N, N_CHI] chi angles that are actually parameterized
        - idx: optional [M] residues to score (defaults to all)

    Outputs:
        - scalar loss
    '''
    if idx is not None:
        log_p, mu, sigma = log_p[idx], mu[idx], sigma[idx]
        chi_angles, occupancies, valid = chi_angles[idx], occupancies[idx], valid[idx]

    # von mises concentration from the library standard deviations
    kappa = (1.0 / sigma.clamp_min(1e-3).pow(2)).clamp_min(KAPPA_MIN)

    # [N, A, S, N_CHI] angular deviation of each altloc from each rotamer mean
    d = chi_angles[:, :, None, :] - mu[:, None, :, :]
    k = kappa[:, None, :, :]
    # log vM written via i0e so large kappa stays finite
    log_vm = k*(torch.cos(d) - 1.0) - torch.log(2*math.pi*torch.special.i0e(k))
    # only sum over chi angles this residue actually has
    log_vm = (log_vm * valid[:, None, None, :].to(log_vm.dtype)).sum(-1) # [N, A, S]

    # mix over rotamer slots
    log_mix = torch.logsumexp(log_p[:, None, :] + log_vm, dim=-1) # [N, A]

    # weight each altloc by its occupancy; empty altloc rows have occupancy 0
    w = occupancies
    return -(w * log_mix).sum() / w.sum().clamp_min(1e-8)


def kl_standard_normal(mu_z, logvar_z, mask, free_bits=0.0):
    '''
    KL( q(z|x) || N(0, I) ) averaged over the observed (resolved) residues.

    This is the VAE regularizer: it pulls each residue's latent posterior toward
    a unit Gaussian, so the dynamics representation space stays smooth, continuous
    and samplable rather than fragmenting into isolated codes.

    free_bits > 0 gives each latent dimension a KL budget (nats) that is not
    penalized, so the regularizer stops squeezing out the small ensemble-specific
    signal - the anti-collapse half of making the dynamics encoding non-trivial.

    Inputs:
        - mu_z, logvar_z: [..., N, d_latent] posterior parameters
        - mask: [..., N] bool, residues to include (those with an observed ensemble)
        - free_bits: per-dimension free KL budget (0 = plain KL)

    Outputs:
        - scalar KL per included residue
    '''
    kl_dim = -0.5 * (1.0 + logvar_z - mu_z.pow(2) - logvar_z.exp())  # [..., N, d]
    if free_bits > 0:
        kl_dim = kl_dim.clamp_min(free_bits)
    kl = kl_dim.sum(-1)  # [..., N]
    m = mask.to(kl.dtype)
    return (kl * m).sum() / m.sum().clamp_min(1.0)


def occ_entropy(occupancies, eps=1e-8):
    '''
    Per-residue conformational entropy (nats) of the altloc occupancy distribution.

    occupancies are already normalized over the altloc axis in featurization, so a
    single conformer gives 0 and a spread ensemble gives a positive value. This is
    the target the hetero_head predicts from z; it is computable only from the full
    ensemble, so a top-altloc-only input cannot supply it.
    '''
    p = occupancies
    return -(p * (p + eps).log()).sum(-1)  # [...]


def mixture_entropy(log_p):
    '''
    Mean entropy (nats) of the predicted rotamer-slot mixtures in `log_p` [M, S].

    Added to the loss as a penalty (minimized), it pushes mass onto a few slots
    instead of hedging broadly across the rotamer space - the mode-covering
    tendency of the reconstruction NLL is what makes multimodal residues smear.
    Invalid slots carry log_p ~ LOG_ZERO so p ~ 0 and contribute ~0 here.
    '''
    if log_p.numel() == 0:
        return log_p.new_zeros(())
    p = log_p.exp()
    return -(p * log_p).sum(-1).mean()


# symmetric terminal chi (0-indexed) per residue type: a 180-deg flip is
# chemically indistinguishable, so the OT ground cost folds them to period pi.
# ASP chi2, GLU chi3, PHE chi2, TYR chi2, indexed by the featurizer's aa tokens.
_SYM_CHI = torch.zeros(21, N_CHI, dtype=torch.bool)
for _a, _c in (("D", 1), ("E", 2), ("F", 1), ("Y", 1)):
    _SYM_CHI[ONE_TO_IDX[_a], _c] = True


def folded_chi_cost(mu, chi_obs, valid, aa_tokens):
    '''
    RMS circular chi distance (radians) between every predicted component and every
    observed altloc, over the residue's valid chi, folded to period pi for the
    symmetric terminal chi of ASP/GLU/PHE/TYR.

    Shapes: mu [M,K,C], chi_obs [M,A,C], valid [M,C] bool, aa_tokens [M] -> [M,K,A].
    The sqrt argument is floored so the gradient stays finite when a predicted mean
    sits exactly on an observed angle (distance 0 -> infinite sqrt' -> nan).
    '''
    sym = _SYM_CHI.to(aa_tokens.device)[aa_tokens.clamp(max=20)]    # [M,C]
    period = torch.where(sym, math.pi, 2 * math.pi)[:, None, None, :]
    d = mu[:, :, None, :] - chi_obs[:, None, :, :]                  # [M,K,A,C]
    d = (d + period / 2) % period - period / 2
    vC = valid[:, None, None, :].to(d.dtype)
    msq = ((d * d) * vC).sum(-1) / vC.sum(-1).clamp_min(1.0)        # [M,K,A]
    return torch.sqrt(msq + 1e-4)


def sinkhorn_ot(log_p, mu, chi_obs, occ, valid, aa_tokens,
                eps=0.2, n_iter=25, topk=16, reduce="mean"):
    '''
    Entropic optimal-transport (Sinkhorn) cost between each residue's predicted
    rotamer mixture and its observed qFit ensemble, on the chi torus - the
    differentiable analog of the W1 eval metric.

    Predicted: weights exp(log_p) at the refined means mu (top-K rotamer slots;
    the rest carry ~0 mass after the entropy penalty). Observed: occupancies at
    the altloc chi angles. Ground cost is the RMS per-chi circular distance
    (radians), folded to period pi for the symmetric terminal chi of ASP/GLU/PHE/
    TYR. Unlike the entropy penalty (which only thins mass), minimizing OT
    *relocates* predicted mass onto the true modes.

    Shapes: log_p [M,S], mu [M,S,C], chi_obs [M,A,C], occ [M,A], valid [M,C] bool,
    aa_tokens [M]. reduce="mean" returns the scalar mean OT cost; reduce="none"
    returns the per-residue cost [M] (the confidence head's regression target).
    '''
    if log_p.numel() == 0:
        return log_p.new_zeros(())
    M, S, C = mu.shape
    K = min(topk, S)
    # keep the top-K predicted slots and renormalize their (log) weights; clamp
    # out of the -1e30 masked range (residues with <K valid slots) so the Sinkhorn
    # recursion stays numerically sane
    log_a, idx = log_p.topk(K, dim=-1)                              # [M,K]
    log_a = (log_a - torch.logsumexp(log_a, -1, keepdim=True)).clamp_min(-30.0)
    mu_k = torch.gather(mu, 1, idx[..., None].expand(-1, -1, C))    # [M,K,C]

    cost = folded_chi_cost(mu_k, chi_obs, valid, aa_tokens)        # [M,K,A]

    # observed marginal (empty altlocs -> -inf so they carry no mass)
    b = occ / occ.sum(-1, keepdim=True).clamp_min(1e-30)
    log_b = torch.where(occ > 0, b.clamp_min(1e-30).log(),
                        b.new_full((), float("-inf")))             # [M,A]

    # log-domain Sinkhorn iterations
    Cs = cost / eps
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(n_iter):
        u = log_a - torch.logsumexp(v[:, None, :] - Cs, dim=2)      # [M,K]
        v = log_b - torch.logsumexp(u[:, :, None] - Cs, dim=1)      # [M,A]
    plan = (u[:, :, None] + v[:, None, :] - Cs).exp()               # [M,K,A]
    per_res = (plan * cost).sum(dim=(1, 2))                          # [M]
    return per_res if reduce == "none" else per_res.mean()


def occupancy_match_loss(log_p, mu, chi_obs, occ, valid, aa_tokens, tau=0.3, topk=16):
    '''
    Penalize the predicted rotamer WEIGHTS for not matching the observed altloc
    OCCUPANCIES.

    The reconstruction NLL rewards placing a mode near each altloc, and the Sinkhorn
    OT minimizes transport COST (dominated by chi location) - neither pins the mixture
    WEIGHTS to the occupancy split. So the model reaches the right conformers with the
    wrong populations (weight<->occupancy Pearson ~0.4) and the wrong dominant state
    ~1 in 4 multi-conformer residues.

    Each predicted mode routes its weight to the observed altloc it is closest to
    (softmax over altlocs of the negative chi cost, temperature tau radians); the
    predicted population landing on each altloc is then matched to its true occupancy
    with an L1 (total-variation) loss. This trains weight<->occupancy correspondence
    directly - a wrong dominant, over-lopsided weights, and spurious extra modes all
    move population away from where the occupancies want it and are penalized. Single-
    conformer residues route all mass to their one altloc and contribute ~0, so the
    term focuses on the multi-conformer residues.

    Shapes match sinkhorn_ot: log_p [M,S], mu [M,S,C], chi_obs [M,A,C], occ [M,A],
    valid [M,C] bool, aa_tokens [M].
    '''
    if log_p.numel() == 0:
        return log_p.new_zeros(())
    M, S, C = mu.shape
    K = min(topk, S)
    log_a, idx = log_p.topk(K, dim=-1)                              # [M,K]
    a = (log_a - torch.logsumexp(log_a, -1, keepdim=True)).exp()    # [M,K] predicted weights
    mu_k = torch.gather(mu, 1, idx[..., None].expand(-1, -1, C))    # [M,K,C]

    cost = folded_chi_cost(mu_k, chi_obs, valid, aa_tokens)        # [M,K,A]

    # route each predicted mode's weight to its nearest observed altloc; only real
    # altlocs (occ > 0) may receive mass
    occ_pos = occ > 0                                              # [M,A]
    neg = torch.where(occ_pos[:, None, :], -cost / tau,
                      cost.new_full((), float("-inf")))            # [M,K,A]
    assign = torch.softmax(neg, dim=-1)                           # [M,K,A]
    q = (a[:, :, None] * assign).sum(1)                           # [M,A] predicted population
    b = occ / occ.sum(-1, keepdim=True).clamp_min(1e-8)           # [M,A] observed occupancy
    return (q - b).abs().sum(-1).mean()


def slot_cross_entropy(log_p, mu, chi_obs, occ, valid, aa_tokens):
    '''
    Precision-aware term: cross-entropy of each observed altloc's best-matching
    rotamer slot under the predicted weights, occupancy-weighted.

        loss = - Sum_a occ_a log p_{s*(a)},   s*(a) = argmin_s cost(mu_s, chi_a)

    The reconstruction NLL is satisfied by spreading weight over every slot whose
    (broad) von Mises kernel overlaps an altloc, so mass leaks onto neighbouring
    rotamers that do not match on every chi. Here only the single nearest slot
    (detached argmin of the folded chi cost at the refined means) earns credit, and
    the optimum puts exactly occ_a on it, so the weights stay calibrated while the
    stray mass is penalized.

    Shapes: log_p [M,S], mu [M,S,C], chi_obs [M,A,C], occ [M,A], valid [M,C] bool,
    aa_tokens [M]. Masked slots (log_p ~ LOG_ZERO) are never targets.
    '''
    if log_p.numel() == 0:
        return log_p.new_zeros(())
    with torch.no_grad():
        cost = folded_chi_cost(mu, chi_obs, valid, aa_tokens)       # [M,S,A]
        real = log_p > LOG_ZERO / 2                                 # [M,S]
        cost = cost.masked_fill(~real[:, :, None], float("inf"))
        target = cost.argmin(1)                                     # [M,A]
    log_hit = log_p.gather(1, target)                               # [M,A]
    w = occ / occ.sum(-1, keepdim=True).clamp_min(1e-8)
    return -(w * log_hit).sum(-1).mean()


def stray_mass_loss(log_p, mu, chi_obs, occ, valid, aa_tokens,
                    tol_deg=15.0, soft_deg=3.0, topk=16):
    '''
    Precision-targeted term: the predicted weight on rotamer modes that match no
    observed altloc, a differentiable version of 1 - weighted rotamer precision.

        loss = Sum_k p_k (1 - max_a hit(mu_k, chi_a)),
        hit  = Prod_c sigmoid((tol - |d_c|) / soft)   over the residue's valid chi

    A mode hits an altloc when every chi agrees within tol_deg (the benchmark's
    rule, symmetric terminal chi folded to period pi). Unlike the slot cross-entropy,
    which only rewards the nearest slot, this charges every unit of stray mass
    directly, and its gradient also pulls refined means that sit just outside the
    tolerance into it. Only the top-K slots by weight are scored (the rest carry
    almost no mass), without renormalizing.

    Shapes match sinkhorn_ot: log_p [M,S], mu [M,S,C], chi_obs [M,A,C], occ [M,A],
    valid [M,C] bool, aa_tokens [M].
    '''
    if log_p.numel() == 0:
        return log_p.new_zeros(())
    M, S, C = mu.shape
    K = min(topk, S)
    log_a, idx = log_p.topk(K, dim=-1)                              # [M,K]
    mu_k = torch.gather(mu, 1, idx[..., None].expand(-1, -1, C))    # [M,K,C]

    sym = _SYM_CHI.to(aa_tokens.device)[aa_tokens.clamp(max=20)]    # [M,C]
    period = torch.where(sym, math.pi, 2 * math.pi)[:, None, None, :]
    d = mu_k[:, :, None, :] - chi_obs[:, None, :, :]                # [M,K,A,C]
    d = (d + period / 2) % period - period / 2
    per_chi = torch.sigmoid((math.radians(tol_deg) - d.abs()) / math.radians(soft_deg))
    per_chi = torch.where(valid[:, None, None, :], per_chi, torch.ones_like(per_chi))
    hit = per_chi.prod(-1).masked_fill(~(occ > 0)[:, None, :], 0.0)  # [M,K,A]
    stray = 1.0 - hit.max(-1).values                                # [M,K]
    return (log_a.exp() * stray).sum(-1).mean()


def matched_atom_loss(chi_pred, log_w, sc_atoms, chi_obs, occ, aa_tokens, atom_mask,
                      topk=8):
    '''
    Set-prediction reconstruction loss in ATOM space (Angstrom) - the metric that
    actually matters, and the one whose gradient does not vanish when a predicted
    conformer is far from its altloc (unlike the von Mises NLL, which stalls dead
    components).

    Each predicted conformer is placed by rebuilding EACH observed altloc's own atoms
    to the predicted chi (differentiable side-chain kinematics), giving a [M,K,A]
    all-atom-RMSD cost between predicted conformer k and observed altloc a. Observed
    altlocs are Hungarian-matched to distinct predicted conformers on the detached
    cost; the loss is the mean matched RMSD plus a weight term pulling each matched
    conformer's weight to its altloc occupancy (and unmatched conformers' weight to 0).

    The predicted mixture may range over many components (e.g. the shift head's ~100
    Dunbrack slots); only the top-`topk` by weight are kept (renormalized), so the loss
    is head-agnostic and cheap for either head.

    Inputs (M scored residues):
        chi_pred:  [M, S, N_CHI] predicted component chi (the head's mu)
        log_w:     [M, S] predicted component log weights (the head's log_p)
        sc_atoms:  [M, A, MAXATOM, 3] observed altloc heavy-atom coords
        chi_obs:   [M, A, N_CHI] observed altloc chi
        occ:       [M, A] observed occupancies (0 = empty altloc slot)
        aa_tokens: [M]
        atom_mask: [M, MAXATOM] real side-chain heavy atoms present

    Returns (atom_rmsd_loss, weight_loss); both scalars.
    '''
    from scipy.optimize import linear_sum_assignment

    M, S, C = chi_pred.shape
    A = sc_atoms.shape[1]
    if M == 0:
        z = chi_pred.new_zeros(())
        return z, z

    # keep the top-K components by weight (renormalized), so this is cheap for the
    # ~100-slot shift head as well as the few-conformer head
    K = min(topk, S)
    logw_k, sel = log_w.topk(K, dim=-1)                      # [M,K]
    chi_pred = torch.gather(chi_pred, 1, sel[..., None].expand(M, K, C))
    log_w = logw_k - torch.logsumexp(logw_k, -1, keepdim=True)

    # rebuild every (k, a): retarget observed altloc a's atoms to predicted chi k
    tmpl = sc_atoms[:, None].expand(M, K, A, sc_atoms.shape[-2], 3)
    cref = chi_obs[:, None].expand(M, K, A, C)
    ctgt = chi_pred[:, :, None].expand(M, K, A, C)
    aa = aa_tokens[:, None, None].expand(M, K, A)
    built = sc_rebuild(tmpl, cref, ctgt, aa)                 # [M,K,A,MAXATOM,3]

    diff = built - sc_atoms[:, None]                         # [M,K,A,MAXATOM,3]
    am = atom_mask[:, None, None].to(diff.dtype)            # [M,1,1,MAXATOM]
    msd = (diff.pow(2).sum(-1) * am).sum(-1) / am.sum(-1).clamp_min(1.0)  # [M,K,A]
    cost = torch.sqrt(msd + 1e-8)                            # [M,K,A] all-atom RMSD

    w = log_w.exp()                                          # [M,K]
    cnp = np.nan_to_num(cost.detach().float().cpu().numpy(),
                        nan=1e3, posinf=1e3, neginf=1e3)     # robust matching
    av = (occ > 0).cpu().numpy()                             # [M,A]
    occ_np = occ.detach().cpu().numpy()

    mi, ki, ai = [], [], []
    tgt_w = torch.zeros_like(w)
    for m in range(M):
        cols = np.where(av[m])[0]                            # valid observed altlocs
        if len(cols) == 0:
            continue
        r, cidx = linear_sum_assignment(cnp[m][:, cols].T)   # rows=altlocs, cols=modes
        for rr, kk in zip(r, cidx):
            a_idx = int(cols[rr])
            mi.append(m); ki.append(int(kk)); ai.append(a_idx)
            tgt_w[m, int(kk)] = float(occ_np[m, a_idx])
    if not mi:
        z = chi_pred.new_zeros(())
        return z, z
    mi = torch.tensor(mi, device=w.device)
    ki = torch.tensor(ki, device=w.device)
    ai = torch.tensor(ai, device=w.device)
    matched = cost[mi, ki, ai]
    finite = torch.isfinite(matched)                        # drop degenerate rebuilds
    atom_rmsd = matched[finite].mean() if finite.any() else chi_pred.new_zeros(())
    # weights: matched -> occupancy, unmatched -> 0 (over residues that had a match)
    matched_res = torch.unique(mi)
    weight_loss = ((w[matched_res] - tgt_w[matched_res]) ** 2).sum(-1).mean()
    return atom_rmsd, weight_loss


# AlphaFold-style confidence: predicted per-residue W1 is discretized into bins
# over [0, CONF_MAX] radians and predicted as a distribution (like pLDDT/PAE).
CONF_BINS = 32
CONF_MAX = math.pi          # covers the full 0-180 deg W1 range
# predicted side-chain RSCC (pRSCC) of the reconstruction: bins over [0, 1]; the
# rare negative correlations fall in the lowest bin
RSCC_BINS = 50
RSCC_MAX = 1.0


def confidence_loss(log_p_bins, target, max_value=CONF_MAX):
    '''
    Cross-entropy of a binned per-residue target under the predicted confidence
    distribution (AF pLDDT/PAE style). log_p_bins [M, n_bins] is the head output on
    the scored residues; target [M] is the actual (detached) value: the W1 in
    radians for the confidence head, the side-chain RSCC for the pRSCC head
    (max_value=RSCC_MAX).
    '''
    if log_p_bins.numel() == 0:
        return log_p_bins.new_zeros(())
    n_bins = log_p_bins.shape[-1]
    idx = (target / max_value * n_bins).long().clamp(0, n_bins - 1)   # [M]
    return -log_p_bins.gather(-1, idx[:, None]).squeeze(-1).mean()


def pick_mask_indices(resolved_mask, occupancies, valid):
    '''
    Residues eligible to be masked: resolved, with at least one parameterized chi
    angle, and with at least one altloc carrying occupancy. GLY/ALA and unresolved
    positions are never valid targets.

    Returns the eligible indices; sample one of these per step to mask a single
    residue ensemble at a time.
    '''
    eligible = resolved_mask & valid.any(-1) & (occupancies > 0).any(-1)
    return eligible.nonzero(as_tuple=True)[0]
