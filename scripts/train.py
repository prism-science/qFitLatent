# created by clay
'''
Train the qFit dynamics VAE.

Each chain is encoded and its full sidechain ensemble reconstructed in one pass:
the model sees the sequence, backbone frames and every residue's observed qFit
altlocs, compresses each residue to a small latent, and reconstructs the altloc
distribution from that latent (free conformers on the chi torus, conditioned on
the residue's amino acid and backbone). The loss is the occupancy-weighted reconstruction NLL over all
scorable residues plus a KL term that keeps the latent space smooth.

    loss = recon_nll + beta_t * kl        beta_t annealed 0 -> beta over kl_warmup

With --mask-frac a random fraction of the scored residues is hidden each step and
must be reconstructed from context; --masked-weight adds their NLL again on its own
weight, --slot-ce adds a cross-entropy on each altloc's nearest rotamer slot, and
--stray-mass charges the hidden residues' weight on modes that match no altloc.
--env-neighbours gives the encoder the side-chain atoms of each residue's nearest
neighbours (qfitlatent/environment.py). --mask-strategy hides residues
independently or as clusters close in sequence or space (qfitlatent/masking.py).

With --prscc and --rscc-targets (scripts/build_rscc_targets.py) the pRSCC head learns
the side-chain RSCC of the reconstructions. Each step hides, per chain, one of the
target mask groups (or nothing), so the head sees the forward pass its target came
from; use it with --freeze-backbone and --init-from the checkpoint that made them.

Structures are read from the train/test directories, or from --train-list /
--val-list files (one PDB path per line).
Features are built once in parallel (--workers), cached in --cache-dir, and kept
in CPU memory; each batch is copied to the device as it is collated. Writes a CSV loss history
(epoch, train_loss, val_loss, train_kl, val_kl) where
train_loss / val_loss are the reconstruction NLL (the metric selected on).
'''

import argparse
import csv
import sys
from pathlib import Path

import torch

# allow "python scripts/train.py" from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qfitlatent.data import attach_rscc_targets, load_or_preload, structure_paths
from qfitlatent.masking import MASK_STRATEGIES, draw_hidden
import torch.nn.functional as F

from qfitlatent.model import (
    qFitLatent, ensemble_nll, kl_standard_normal, mixture_entropy,
    sinkhorn_ot, occupancy_match_loss, matched_atom_loss, confidence_loss,
    occ_entropy, slot_cross_entropy, stray_mass_loss, RSCC_MAX,
)


def make_batches(samples, pair_budget, max_bs):
    '''
    Pack chains into batches under a memory budget.

    Cost is proxied by B * Nmax^2 (the pair-track / triangle tensor size), so long
    chains land in small batches and short chains in large ones - keeping GPU
    memory bounded and padding waste low. Chains are sorted by length so each batch
    holds similar-length chains.
    '''
    order = sorted(range(len(samples)), key=lambda i: len(samples[i][1]["aa_tokens"]))
    batches, cur, nmax = [], [], 0
    for i in order:
        n = len(samples[i][1]["aa_tokens"])
        nmax_new = max(nmax, n)
        if cur and ((len(cur) + 1) * nmax_new ** 2 > pair_budget or len(cur) >= max_bs):
            batches.append(cur)
            cur, nmax_new = [], n
        cur.append(i)
        nmax = nmax_new
    if cur:
        batches.append(cur)
    return batches


def rscc_pass(d, generator):
    '''
    One pRSCC pass for a chain with targets: a random mask group g, or no group
    (g = n_groups). Returns (hidden [N] bool, target [N]): the group is hidden and
    scored against its masked RSCC; with no group, every residue is scored against
    its visible RSCC (nan = not scored).
    '''
    group = d["rscc_group"]
    n_groups = int(group.max()) + 1
    g = int(torch.randint(n_groups + 1, (1,), generator=generator))
    if g == n_groups:
        return torch.zeros_like(group, dtype=torch.bool), d["rscc_visible"]
    hidden = group == g
    return hidden, torch.where(hidden, d["rscc_masked"], float("nan"))


def collate(idxs, samples, device, mask_frac=0.0, generator=None, prscc=False,
            mask_strategy="random", cluster_size=12):
    '''
    Pad a set of chains into batched tensors [B, Nmax, ...] plus the masks the model
    needs: pad_mask (real residues), hide (unresolved + padding, encoded from the
    mask token), score (residues the reconstruction loss and KL are taken over) and
    masked (scored residues whose ensemble is also hidden, a random `mask_frac` of
    them, so the model learns to reconstruct an ensemble from context alone). The
    cluster strategies draw per chain with a CPU `generator`.
    prscc=True instead hides one pRSCC mask group per chain (`rscc_pass`, drawn with
    the CPU `generator`) and adds its targets as out["rscc"] (nan where unscored).
    '''
    dicts = [samples[i][1] for i in idxs]
    Ns = [len(d["aa_tokens"]) for d in dicts]
    B, Nmax = len(dicts), max(Ns)
    A, NC = dicts[0]["chi_angles"].shape[1], dicts[0]["chi_angles"].shape[2]
    MA = dicts[0]["sc_atoms"].shape[2]           # heavy atoms per residue slot

    z = lambda *shape, dt=torch.float32: torch.zeros(B, Nmax, *shape, dtype=dt, device=device)
    keys = ("aa_tokens", "R", "t", "chi_angles", "occupancies", "chi_mask",
            "phi", "psi", "bb_mask", "sc_atoms", "sc_atom_mask")
    out = {
        "aa_tokens": torch.zeros(B, Nmax, dtype=torch.long, device=device),
        "R": torch.eye(3, device=device).expand(B, Nmax, 3, 3).clone(),
        "t": z(3), "chi_angles": z(A, NC), "occupancies": z(A),
        "chi_mask": z(NC, dt=torch.bool), "phi": z(), "psi": z(),
        "bb_mask": z(2, dt=torch.bool),
        "sc_atoms": z(A, MA, 3), "sc_atom_mask": z(MA, dt=torch.bool),
    }
    pad_mask = torch.zeros(B, Nmax, dtype=torch.bool, device=device)
    resolved = torch.zeros(B, Nmax, dtype=torch.bool, device=device)
    score = torch.zeros(B, Nmax, dtype=torch.bool, device=device)
    if prscc:
        group_hidden = torch.zeros(B, Nmax, dtype=torch.bool, device=device)
        out["rscc"] = torch.full((B, Nmax), float("nan"), device=device)
    for b, (d, n) in enumerate(zip(dicts, Ns)):
        for k in keys:
            out[k][b, :n] = d[k]
        pad_mask[b, :n] = True
        resolved[b, :n] = d["resolved_mask"]
        score[b, :n] = d["score"]
        if prscc:
            group_hidden[b, :n], out["rscc"][b, :n] = rscc_pass(d, generator)

    hide = ~resolved            # residues without an observed ensemble to encode
    if prscc:
        return out, pad_mask, hide | group_hidden, resolved, score, group_hidden
    if mask_strategy == "random":
        draw = torch.rand(B, Nmax, device=device, generator=generator)
        masked = score & (draw < mask_frac)
    else:
        masked = torch.zeros_like(score)
        for b, (d, n) in enumerate(zip(dicts, Ns)):
            masked[b, :n] = draw_hidden(d["score"], d["t"], mask_strategy, mask_frac,
                                        cluster_size, generator).to(device)
    return out, pad_mask, hide | masked, resolved, score, masked


def run_epoch(model, samples, beta, optimizer=None, grad_clip=0.0, entropy=0.0,
              sinkhorn=0.0, sinkhorn_eps=0.2, sinkhorn_iter=25, occ_match=0.0,
              atom_loss=0.0, atom_weight=1.0,
              confidence=0.0, hetero=0.0, free_bits=0.0, mask_frac=0.0,
              masked_weight=0.0, slot_ce=0.0, stray_mass=0.0, prscc=0.0,
              freeze=False, pair_budget=1_500_000, max_bs=64, device="cpu",
              mask_strategy="random", cluster_size=12):
    '''
    One pass over every chain: reconstruct all scorable residues, add the KL, the
    mixture-entropy penalty (thins predicted mass onto few rotamers), the Sinkhorn/
    OT term (relocates mass onto the true modes), and the AlphaFold-style
    confidence loss (trains the head to predict each residue's own W1 error).
    optimizer=None runs in eval mode with no gradients and the latent mean
    (deterministic) for validation. freeze=True (confidence-head-only training)
    also runs the backbone deterministically so the head predicts the deployed
    (mean-latent, no-dropout) model's W1. mask_frac hides that fraction of scored
    residues (a fixed draw in eval mode, so val epochs stay comparable) and logs their
    NLL as "mrecon"; masked_weight adds that NLL to the loss again. slot_ce weights the
    nearest-slot cross-entropy ("sce"); stray_mass weights the hidden residues'
    off-altloc weight ("stray"). prscc weights the pRSCC head's loss ("prscc")
    on the mask groups of the rscc targets (`collate` with prscc=True).
    Returns (recon, KL, entropy, OT, conf, ...).
    '''
    training = optimizer is not None
    # frozen-backbone runs keep the backbone deterministic (eval mode) while still
    # letting gradients reach the confidence head via set_grad_enabled below
    model.train(training and not freeze)
    batches = make_batches(samples, pair_budget, max_bs)

    tots = {"recon": 0.0, "kl": 0.0, "ent": 0.0, "ot": 0.0, "occ": 0.0,
            "atom": 0.0, "aw": 0.0, "conf": 0.0, "het": 0.0, "mrecon": 0.0,
            "sce": 0.0, "stray": 0.0, "prscc": 0.0}
    generator = None if training else torch.Generator(device=device).manual_seed(0)
    # the pRSCC group draw and the cluster masks are made on the CPU, with the features
    if prscc > 0 or mask_strategy != "random":
        generator = None if training else torch.Generator().manual_seed(0)
    n_steps, n_masked, oom, bad = 0, 0, 0, 0
    for idxs in batches:
        s, pad_mask, hide, resolved, score, masked = collate(
            idxs, samples, device, mask_frac, generator, prscc=prscc > 0,
            mask_strategy=mask_strategy, cluster_size=cluster_size)
        try:
            with torch.set_grad_enabled(training):
                log_p, mu, sigma, valid, mu_z, logvar_z, conf, het_pred, prscc_p = model(
                    s["aa_tokens"], s["R"], s["t"], s["chi_angles"],
                    s["occupancies"], s["chi_mask"], s["phi"], s["psi"],
                    s["bb_mask"], hide, pad_mask, sc_atoms=s["sc_atoms"],
                )
                recon = ensemble_nll(
                    log_p[score], mu[score], sigma[score],
                    s["chi_angles"][score], s["occupancies"][score], valid[score],
                )
                kl = kl_standard_normal(mu_z, logvar_z, resolved, free_bits)
                # auxiliary: force z to encode the ensemble's conformational entropy
                het_tgt = occ_entropy(s["occupancies"])
                hl = (F.mse_loss(het_pred[score], het_tgt[score])
                      if hetero > 0 else log_p.new_zeros(()))
                ent = mixture_entropy(log_p[score])
                # per-residue OT is needed by the Sinkhorn loss and as the
                # confidence target, so compute it once when either is active
                if sinkhorn > 0 or confidence > 0:
                    ot_per = sinkhorn_ot(
                        log_p[score], mu[score], s["chi_angles"][score],
                        s["occupancies"][score], valid[score], s["aa_tokens"][score],
                        sinkhorn_eps, sinkhorn_iter, reduce="none")
                    ot = ot_per.mean()
                else:
                    ot = log_p.new_zeros(())
                cl = (confidence_loss(conf[score], ot_per.detach())
                      if confidence > 0 else log_p.new_zeros(()))
                # explicit weight<->occupancy matching (the mixture populations, which
                # recon/OT leave loosely constrained, are pinned to the altloc split)
                om = (occupancy_match_loss(
                        log_p[score], mu[score], s["chi_angles"][score],
                        s["occupancies"][score], valid[score], s["aa_tokens"][score])
                      if occ_match > 0 else log_p.new_zeros(()))
                # atom-space set-prediction loss (Angstrom): the head's mu are the
                # predicted conformer chi, log_p their weights; matched to observed
                # altloc atoms. This is the primary objective for head=conformer.
                if atom_loss > 0:
                    al, aw = matched_atom_loss(
                        mu[score], log_p[score], s["sc_atoms"][score],
                        s["chi_angles"][score], s["occupancies"][score],
                        s["aa_tokens"][score], s["sc_atom_mask"][score])
                else:
                    al = aw = log_p.new_zeros(())
                # the atom-space loss REPLACES the von Mises recon NLL (whose gradient
                # vanishes for far conformers); recon is still computed above for logging
                recon_term = recon if atom_loss == 0 else log_p.new_zeros(())
                # the hidden residues' NLL on its own weight (they are also in recon)
                mrecon = (ensemble_nll(
                            log_p[masked], mu[masked], sigma[masked],
                            s["chi_angles"][masked], s["occupancies"][masked],
                            valid[masked])
                          if masked.any() else log_p.new_zeros(()))
                sce = (slot_cross_entropy(
                        log_p[score], mu[score], s["chi_angles"][score],
                        s["occupancies"][score], valid[score], s["aa_tokens"][score])
                       if slot_ce > 0 else log_p.new_zeros(()))
                # 1 - weighted precision, on the hidden residues the benchmark scores
                stray = (stray_mass_loss(
                          log_p[masked], mu[masked], s["chi_angles"][masked],
                          s["occupancies"][masked], valid[masked], s["aa_tokens"][masked])
                         if masked.any() else log_p.new_zeros(()))
                if prscc > 0:
                    has = torch.isfinite(s["rscc"])
                    pl = confidence_loss(prscc_p[has], s["rscc"][has], RSCC_MAX)
                else:
                    pl = log_p.new_zeros(())
                loss = (recon_term + beta * kl + entropy * ent + sinkhorn * ot
                        + occ_match * om + atom_loss * al + atom_loss * atom_weight * aw
                        + confidence * cl + hetero * hl + masked_weight * mrecon
                        + slot_ce * sce + stray_mass * stray + prscc * pl)
            # a non-finite loss (a pathological chi / rotamer geometry) must never
            # poison the weights via a nan gradient; skip the batch and continue
            if not torch.isfinite(loss):
                bad += 1
                if training:
                    optimizer.zero_grad(set_to_none=True)
                continue
            if training:
                optimizer.zero_grad()
                loss.backward()
                # clip always (also gives the grad norm); skip the *step* if the
                # gradient is non-finite so a bad backward can't corrupt the weights
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip if grad_clip > 0 else 1e30)
                if not torch.isfinite(gnorm):
                    bad += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
            tots["recon"] += float(recon.detach()); tots["kl"] += float(kl.detach())
            tots["ent"] += float(ent.detach()); tots["ot"] += float(ot.detach())
            tots["occ"] += float(om.detach())
            tots["atom"] += float(al.detach()); tots["aw"] += float(aw.detach())
            tots["conf"] += float(cl.detach()); tots["het"] += float(hl.detach())
            tots["sce"] += float(sce.detach()); tots["prscc"] += float(pl.detach())
            if masked.any():
                tots["mrecon"] += float(mrecon.detach())
                tots["stray"] += float(stray.detach())
                n_masked += 1
            n_steps += 1
        except torch.cuda.OutOfMemoryError:
            # a single oversized batch must never kill an overnight run; drop it,
            # free the cache, and keep going
            oom += 1
            optimizer.zero_grad(set_to_none=True) if training else None
            del s, pad_mask, hide, resolved, score, masked
            torch.cuda.empty_cache()

    if oom:
        print(f"    WARNING: skipped {oom} batch(es) on CUDA OOM")
    if bad:
        print(f"    WARNING: skipped {bad} batch(es) with non-finite loss")
    n = max(n_steps, 1)
    out = {k: v / n for k, v in tots.items()}
    for key in ("mrecon", "stray"):
        out[key] = tots[key] / n_masked if n_masked else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", default="data/structures/train")
    ap.add_argument("--test-dir", default="data/structures/test")
    ap.add_argument("--train-list", default=None,
                    help="file of training PDB paths (overrides --train-dir)")
    ap.add_argument("--val-list", default=None,
                    help="file of validation PDB paths (overrides --test-dir)")
    ap.add_argument("--seq-dir", default="data/seqeuences")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes used to featurize structures")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=10,
                    help="stop after this many epochs with no val improvement "
                         "(0 disables early stopping)")
    ap.add_argument("--d-single", type=int, default=256)
    ap.add_argument("--d-pair", type=int, default=64)
    ap.add_argument("--d-latent", type=int, default=16,
                    help="per-residue dynamics latent size (the representation)")
    ap.add_argument("--n-ipa", type=int, default=8)
    ap.add_argument("--n-conf", type=int, default=16,
                    help="number of free conformers the decoder predicts per residue")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="KL weight (beta-VAE); lower favors reconstruction fidelity")
    ap.add_argument("--kl-warmup", type=int, default=15,
                    help="epochs to ramp beta 0 -> beta (avoids posterior collapse)")
    ap.add_argument("--entropy", type=float, default=0.0,
                    help="mixture-entropy penalty weight; concentrates predicted "
                         "mass onto few rotamers instead of hedging broadly")
    ap.add_argument("--sinkhorn", type=float, default=0.0,
                    help="Sinkhorn/OT loss weight; relocates predicted mass onto "
                         "the observed modes (differentiable analog of the W1 metric)")
    ap.add_argument("--sinkhorn-eps", type=float, default=0.2,
                    help="entropic regularization for Sinkhorn (radians)")
    ap.add_argument("--sinkhorn-iter", type=int, default=25,
                    help="Sinkhorn iterations")
    ap.add_argument("--occ-match", type=float, default=0.0,
                    help="occupancy-matching loss weight; pins the predicted rotamer "
                         "populations to the observed altloc occupancy split (trains "
                         "weight<->occupancy fidelity the recon/OT leave loose)")
    ap.add_argument("--atom-loss", type=float, default=0.0,
                    help="atom-space (Angstrom) set-prediction loss weight; matches "
                         "predicted conformers to observed altlocs and minimizes all-atom "
                         "side-chain RMSD directly (head=conformer). Selects on val atom RMSD")
    ap.add_argument("--atom-weight", type=float, default=1.0,
                    help="relative weight of the occupancy-MSE term within the atom loss")
    ap.add_argument("--confidence", type=float, default=0.0,
                    help="AlphaFold-style confidence-head loss weight; trains the "
                         "head to predict each residue's own W1 error (target detached)")
    ap.add_argument("--hetero", type=float, default=0.0,
                    help="ensemble-heterogeneity loss weight; forces z to predict "
                         "each residue's conformational entropy so the latent carries "
                         "ensemble-specific (dynamics) information")
    ap.add_argument("--free-bits", type=float, default=0.0,
                    help="per-dimension free KL budget (nats); stops the KL squeezing "
                         "out the small ensemble signal (anti-collapse)")
    ap.add_argument("--mask-frac", type=float, default=0.25,
                    help="fraction of scored residues whose ensemble is hidden behind "
                         "the mask token each step (still scored), training masked-"
                         "residue reconstruction; selects on val masked recon (0.25 "
                         "from the masking sweep; 0 trains a plain autoencoder)")
    ap.add_argument("--mask-strategy", choices=MASK_STRATEGIES, default="random",
                    help="hide residues independently or as clusters close in sequence "
                         "or space")
    ap.add_argument("--cluster-size", type=int, default=12,
                    help="scored residues per cluster of the sequence/spatial strategies")
    ap.add_argument("--val-mask-frac", type=float, default=None,
                    help="mask fraction on the val split (default --mask-frac), so runs "
                         "trained at different fractions are selected on one target")
    ap.add_argument("--masked-weight", type=float, default=0.0,
                    help="extra weight on the masked residues' reconstruction NLL "
                         "(they are already part of the recon term at weight 1)")
    ap.add_argument("--slot-ce", type=float, default=0.0,
                    help="precision-aware loss weight: cross-entropy of each altloc's "
                         "nearest rotamer slot, occupancy-weighted")
    ap.add_argument("--stray-mass", type=float, default=0.0,
                    help="precision-targeted loss weight: the hidden residues' "
                         "predicted weight on modes within 15 deg of no altloc")
    ap.add_argument("--env-neighbours", type=int, default=0,
                    help="encode the side-chain atoms of this many nearest neighbours "
                         "(local atomic environment); 0 disables")
    ap.add_argument("--prscc", type=float, default=0.0,
                    help="pRSCC-head loss weight; needs --rscc-targets")
    ap.add_argument("--prscc-input", choices=["latent", "trunk", "trunk_mix"],
                    default="trunk_mix",
                    help="pRSCC head input: decoder features, + trunk features, "
                         "+ the predicted mixture's entropy and top weight (best on val)")
    ap.add_argument("--rscc-targets", nargs="*", default=[],
                    help="build_rscc_targets.py CSVs (train and val); chains without "
                         "targets are dropped when --prscc is set")
    ap.add_argument("--init-from", default=None,
                    help="warm-start weights from a checkpoint (strict=False), e.g. "
                         "to add/train the confidence head on a converged model")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train only the confidence head, or the pRSCC head with "
                         "--prscc (freeze everything else); preserves a warm-started "
                         "model exactly. Selects on val conf / val prscc")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="max grad norm (0 disables); stabilizes the vM NLL tails")
    ap.add_argument("--cache-dir", default=None,
                    help="cache featurized train/val samples here so repeated runs "
                         "skip featurization (delete if structures/featurizer change)")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--pair-budget", type=int, default=1_500_000,
                    help="max B*Nmax^2 per batch (bounds pair-track GPU memory)")
    ap.add_argument("--max-bs", type=int, default=64, help="hard cap on chains per batch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log", default="runs/loss_history.csv")
    ap.add_argument("--checkpoint", default="runs/qfitlatent.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # list-based sets cache as shard directories named after the list file; the
    # original directory sets keep their single-file caches
    def cache(list_file, name):
        if not args.cache_dir:
            return None
        return str(Path(args.cache_dir) / (Path(list_file).stem if list_file else name))

    print(f"device: {device}")
    print(f"loading train structures from {args.train_list or args.train_dir} ...")
    train_samples = load_or_preload(
        cache(args.train_list, "preload_train.pt"),
        structure_paths(args.train_dir, args.train_list), args.seq_dir, args.max_len,
        args.workers)
    print(f"loading val structures from {args.val_list or args.test_dir} ...")
    val_samples = load_or_preload(
        cache(args.val_list, "preload_val.pt"),
        structure_paths(args.test_dir, args.val_list), args.seq_dir, args.max_len,
        args.workers)

    if args.prscc > 0:
        train_samples = attach_rscc_targets(train_samples, args.rscc_targets)
        val_samples = attach_rscc_targets(val_samples, args.rscc_targets)
        print(f"  pRSCC targets for {len(train_samples)} train / "
              f"{len(val_samples)} val chains")
    n_train_res = sum(int(d["score"].sum()) for _, d in train_samples)
    n_val_res = sum(int(d["score"].sum()) for _, d in val_samples)
    print(f"  train: {len(train_samples)} chains, {n_train_res} scored residues")
    print(f"  val:   {len(val_samples)} chains, {n_val_res} scored residues")
    if not train_samples:
        raise SystemExit("no trainable chains found")

    model = qFitLatent(
        d_single=args.d_single, d_pair=args.d_pair, d_latent=args.d_latent,
        n_ipa=args.n_ipa, dropout=args.dropout, n_conf=args.n_conf,
        env_neighbours=args.env_neighbours, prscc_input=args.prscc_input,
    ).to(device)
    if args.init_from:
        sd = torch.load(args.init_from, map_location=device,
                        weights_only=False)["model"]
        # keep only tensors that exist in the current model with a matching shape, so a
        # warm-start can cross architecture changes (e.g. a redesigned encoder): renamed
        # or reshaped modules simply train from their fresh init instead of erroring.
        cur = model.state_dict()
        drop = [k for k in list(sd) if k not in cur or sd[k].shape != cur[k].shape]
        for k in drop:
            sd.pop(k)
        miss = model.load_state_dict(sd, strict=False)
        print(f"  warm-started from {args.init_from} (loaded {len(sd)}, "
              f"skipped {len(drop)} mismatched, missing {len(miss.missing_keys)}, "
              f"unexpected {len(miss.unexpected_keys)})")
    if args.freeze_backbone:
        head = "rscc_head" if args.prscc > 0 else "confidence_head"
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(head)
        print(f"  frozen backbone: training only {head} "
              f"({sum(p.numel() for p in model.parameters() if p.requires_grad)} params)")
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params/1e6:.2f}M params, d_latent={args.d_latent}, lr={args.lr}, "
          f"beta={args.beta} (warmup {args.kl_warmup}), entropy={args.entropy}, "
          f"sinkhorn={args.sinkhorn} (eps {args.sinkhorn_eps}), "
          f"confidence={args.confidence}, dropout={args.dropout}\n")

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_val, best_epoch, since_improve = float("inf"), 0, 0

    cols = ["recon", "kl", "ent", "ot", "occ", "atom", "aw", "conf", "het", "mrecon",
            "sce", "stray", "prscc"]
    val_mask_frac = args.mask_frac if args.val_mask_frac is None else args.val_mask_frac
    for epoch in range(1, args.epochs + 1):
        beta_t = args.beta * min(1.0, epoch / max(args.kl_warmup, 1))
        ot_kw = dict(sinkhorn=args.sinkhorn, sinkhorn_eps=args.sinkhorn_eps,
                     sinkhorn_iter=args.sinkhorn_iter, occ_match=args.occ_match,
                     atom_loss=args.atom_loss, atom_weight=args.atom_weight,
                     confidence=args.confidence,
                     hetero=args.hetero, free_bits=args.free_bits,
                     masked_weight=args.masked_weight, slot_ce=args.slot_ce,
                     stray_mass=args.stray_mass, prscc=args.prscc,
                     freeze=args.freeze_backbone,
                     mask_strategy=args.mask_strategy, cluster_size=args.cluster_size)
        tr = run_epoch(model, train_samples, beta_t, opt, grad_clip=args.grad_clip,
                       entropy=args.entropy, pair_budget=args.pair_budget,
                       max_bs=args.max_bs, device=device, mask_frac=args.mask_frac,
                       **ot_kw)
        va = (run_epoch(model, val_samples, args.beta, None, entropy=args.entropy,
                        pair_budget=args.pair_budget, max_bs=args.max_bs, device=device,
                        mask_frac=val_mask_frac, **ot_kw)
              if val_samples else {k: float("nan") for k in cols})
        history.append([epoch] + [f(k) for k in cols for f in (lambda k: tr[k], lambda k: va[k])])

        # select on the metric being trained: atom RMSD for the atom-loss model,
        # confidence loss when only that head trains, otherwise the reconstruction NLL
        sel = (va["atom"] if args.atom_loss > 0
               else va["prscc"] if args.prscc > 0
               else va["conf"] if args.freeze_backbone
               else va["mrecon"] if args.mask_frac > 0 else va["recon"])
        improved = sel == sel and sel < best_val
        if improved:
            best_val, best_epoch, since_improve = sel, epoch, 0
            if args.checkpoint:
                best_path = Path(args.checkpoint).with_suffix(".best.pt")
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "epoch": epoch, "val_recon": va["recon"],
                            "val_atom": va["atom"], "val_conf": va["conf"],
                            "val_prscc": va["prscc"]}, best_path)
        else:
            since_improve += 1
        flag = "  *" if improved else ""
        ot_s = f" ot {tr['ot']:4.2f}/{va['ot']:4.2f}" if args.sinkhorn > 0 else ""
        om_s = f" occ {tr['occ']:4.2f}/{va['occ']:4.2f}" if args.occ_match > 0 else ""
        am_s = (f" atomA {tr['atom']:5.3f}/{va['atom']:5.3f} wl {tr['aw']:4.2f}/{va['aw']:4.2f}"
                if args.atom_loss > 0 else "")
        cf_s = f" cf {tr['conf']:4.2f}/{va['conf']:4.2f}" if args.confidence > 0 else ""
        cf_s += f" prscc {tr['prscc']:5.3f}/{va['prscc']:5.3f}" if args.prscc > 0 else ""
        ht_s = f" het {tr['het']:5.3f}/{va['het']:5.3f}" if args.hetero > 0 else ""
        mk_s = (f" masked {tr['mrecon']:7.4f}/{va['mrecon']:7.4f}"
                if args.mask_frac > 0 else "")
        mk_s += f" sce {tr['sce']:5.3f}/{va['sce']:5.3f}" if args.slot_ce > 0 else ""
        mk_s += (f" stray {tr['stray']:5.3f}/{va['stray']:5.3f}"
                 if args.mask_frac > 0 else "")
        print(f"epoch {epoch:3d}/{args.epochs}  b{beta_t:4.2f}  "
              f"train {tr['recon']:7.4f} (kl {tr['kl']:5.2f} ent {tr['ent']:4.2f})  "
              f"val {va['recon']:7.4f} (kl {va['kl']:5.2f} ent {va['ent']:4.2f})"
              f"{ot_s}{om_s}{am_s}{cf_s}{ht_s}{mk_s}{flag}",
              flush=True)

        # rewrite the whole history each epoch so the plot script can run anytime
        with open(log_path, "w", newline="") as fh:
            w = csv.writer(fh)
            header = ["epoch"] + [f"{p}_{k}" for k in cols for p in ("train", "val")]
            w.writerow(header)
            w.writerows(history)

        if args.patience and since_improve >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs "
                  f"(best {best_val:.4f} at epoch {best_epoch})")
            break

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
        print(f"\nwrote {ckpt}")
    metric = ("val atom RMSD" if args.atom_loss > 0 else
              "val prscc" if args.prscc > 0 else
              "val conf" if args.freeze_backbone else
              "val masked recon" if args.mask_frac > 0 else "val recon")
    print(f"best {metric} {best_val:.4f} at epoch {best_epoch}")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
