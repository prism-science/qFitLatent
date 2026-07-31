# created by clay
''' 
The loss function for learning qfit latent dynamics through
ground truth multiconformer models refit to better match electron density. 
The model learns a probaility density function for each residue in a protein
from the backbone structure and its sequence. Each residue is treated like a
von mises mixture model in torsional space.
'''

import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from .data.data import N_CHI_PER_AA

class ChiARLoss(nn.Module):
    '''
    Autoregressive negative log-likelihood for the chi-angle von mises mixture.
    Each chi dim d has its own k-component conditional mixture (parameters
    depend on the observed chi_<d via teacher forcing through the AR head).
    The joint log-prob factorises as a sum of per-chi conditional log-probs,
    so the loss is just per-chi vM mixture NLL summed over chi:

        L = -mean (residues) Sum(a) w_a Sum(d) log Sum(k) π_dk · vM(χ_a,d ; μ_dk, κ_dk)

    where (π_dk, μ_dk, κ_dk) at chi d depend on chi_<d. This is the AR
    replacement for the factorised-product vM NLL — each rotameric mode is a
    sequence of component choices, so the model can natively express joint
    coupling on the multi-modal residues (ARG / LYS / aromatic).

    Inputs (broadcast over the altloc axis A from the AR head):
        pi, mu, kappa: [N, A, D, k]
        qfit_chis:     [N, A, D]
        occupancies:   [N, A]
        mask:          [N, D]    (per-residue chi validity)
        symmetry_mask: [N, D]    (π-symmetric chis: ASP χ2 etc.)
    '''
    def forward(
        self,
        pi,
        mu,
        kappa,
        qfit_chis,
        occupancies,
        mask,
        symmetry_mask
    ):
        N, A, D, k = mu.shape
        device = mu.device

        # masks for valid residues
        n_obs = (occupancies > 0).sum(-1)           # observed altloc count
        d_eff = mask.float().sum(-1)                # observed chi count
        valid_res = (n_obs >= 2) & (d_eff >= 1)

        out = {
            "loss": pi.new_zeros(()),
            "per_residue_loss": torch.full((N,), float("nan"), device=device),
            "per_chi_loss": torch.full((N, D), float("nan"), device=device),
        }
        if not valid_res.any():
            return out

        # residual per (altloc, chi, component)
        # qfit_chis: [N, A, D] -> [N, A, D, 1] ; mu: [N, A, D, k]
        error = qfit_chis.unsqueeze(-1).float() - mu.float()
        error_2pi = wrap_2pi(error)
        if symmetry_mask is not None and symmetry_mask.any():
            error_pi = wrap_pi(error)
            sm = symmetry_mask[:, None, :, None].to(torch.bool)   # [N,1,D,1]
            error = torch.where(sm, error_pi, error_2pi)
        else:
            error = error_2pi

        # von mises log density per (altloc, chi, component); stable via i0e
        kap = kappa.float().clamp(min=1e-6)
        log2pi = math.log(2*math.pi)
        log_norm_const = log2pi + kap + torch.special.i0e(kap).log()
        log_vm = kap * torch.cos(error) - log_norm_const          # [N,A,D,k]

        # mixture log-prob per (altloc, chi): logsumexp over components k
        log_pi = pi.float().clamp(min=1e-8).log()                 # [N,A,D,k]
        log_p_chi = torch.logsumexp(log_pi + log_vm, dim=-1)      # [N,A,D]

        # mask out chi dims that dont exist in this residue
        mask_f = mask[:, None, :].float()                         # [N,1,D]
        log_p_chi = log_p_chi * mask_f

        # autoregressive joint log-prob per altloc: sum over chi
        log_p_alt = log_p_chi.sum(-1)                             # [N, A]

        # weight by normalised occupancy, sum over altlocs -> per-residue NLL
        obs_mask = (occupancies > 0).float()
        occ_norm = occupancies / occupancies.sum(-1, keepdim=True).clamp(min=1e-8)
        nll_per_res = -(occ_norm * log_p_alt * obs_mask).sum(-1)  # [N]

        out["loss"] = nll_per_res[valid_res].mean()
        out["per_residue_loss"][valid_res] = nll_per_res[valid_res].detach()

        return out
    
class ConfidenceLoss(nn.Module):
    '''
    Smooth-L1 regression loss for ConfidenceModel: predicts log1p(W1), the
    Wasserstein distance (A) between qFitLatent's predicted chi mixture and
    the true qFit ensemble. log space keeps a few very flexible (large-W1)
    residues from dominating the loss.

    Inputs:
        pred_log_w1: [N]  model output, log1p(W1) prediction
        w1_target:   [N]  real W1 (A); nan where no label (< 2 altlocs)
    '''
    def forward(self, pred_log_w1, w1_target):
        out = {
            "loss": pred_log_w1.new_zeros(()),
            "per_residue_loss": torch.full_like(pred_log_w1, float("nan")),
        }
        valid = torch.isfinite(w1_target)
        if not valid.any():
            return out

        target_log = torch.log1p(w1_target[valid].clamp(min=0))
        per_res = F.smooth_l1_loss(pred_log_w1[valid], target_log, reduction="none")

        out["loss"] = per_res.mean()
        out["per_residue_loss"][valid] = per_res.detach()
        return out


# chi angles are symmetric about 2pi so the distances need to wrap
def wrap_2pi(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

# some residues (PHE, etc.) are symmetric about pi for a given angle
def wrap_pi(x):
    return 0.5 * torch.atan2(torch.sin(2.0 * x), torch.cos(2.0 * x))

