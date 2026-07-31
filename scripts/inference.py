#!/usr/bin/env python3
'''
qFitLatent inference: model loading, chi-angle von Mises mixture prediction, and
sidechain reconstruction via the NeRF (Natural Extension Reference Frame)
algorithm.

The model forward pass only needs backbone geometry (N, CA, C → residue frames)
and the residue identities, so a backbone-only PDB is a valid input; sidechain
atoms in the input are ignored by the predictor and rebuilt from the predicted
chi angles. CB is needed only to reconstruct sidechains in the output PDB.

Usage (standalone, from the repo root):
    python scripts/inference.py --pdb structures/5v92.pdb
    python scripts/inference.py --pdb structures/5v92.pdb \
        --ckpt checkpoints/vmm_ar_k8/latest.pt --out out.pdb

Importable API (used by qfl_evaluation/benchmark):
    load_model(ckpt, device) -> qFitLatent
    run_inference(model, sample, device) -> (pi, mu, kappa)
'''
import argparse, math, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))    # qFitLatent package

DEFAULT_CKPT = ROOT / "checkpoints/vmm_ar_k8/latest.pt"
DEFAULT_CONFIDENCE_CKPT = ROOT / "checkpoints/confidence/latest.pt"

from qfit_latent.model import qFitLatent, ConfidenceModel
from qfit_latent.data.data import parse_pdb, get_xyz

# ── constants ──────────────────────────────────────────────────────────────────

_PI = math.pi

AA_IDX: dict[int, str] = {
    0:"ALA",  1:"ARG",  2:"ASN",  3:"ASP",  4:"CYS",
    5:"GLN",  6:"GLU",  7:"GLY",  8:"HIS",  9:"ILE",
    10:"LEU", 11:"LYS", 12:"MET", 13:"PHE", 14:"PRO",
    15:"SER", 16:"THR", 17:"TRP", 18:"TYR", 19:"VAL", 20:"UNK",
}

# ── sidechain build table ──────────────────────────────────────────────────────
# Bond lengths (Å) and bond angles (°) are ideal protein geometry values from:
#   Engh, R.A. & Huber, R. (1991). Accurate bond and angle parameters for
#   X-ray protein structure refinement. Acta Cryst. A47, 392–400.
#   (Updated in: Int. Tables for Crystallography, Vol. F, Ch. 18.3, 2001.)
#
# Each entry: (atom_name, (ref_a, ref_b, ref_c), bond_len_Å, bond_angle_deg, dihedral)
# dihedral encoding:
#   int i             → chi_angles[i]
#   float f           → fixed f radians
#   (int i, float f)  → chi_angles[i] + f        (i >= 0)
#                     → -chi_angles[-i-1] + f     (i < 0: negate that chi)
# For sp2 aromatic CG, CD1 and CD2 are on opposite sides of the CB–CG axis in
# the ring plane, so dihedral(CA,CB,CG,CD2) = chi2 + π → use (1, _PI).
_SC_BUILD: dict[str, list[tuple]] = {
    "CYS": [
        ("SG",  ("N","CA","CB"), 1.808, 113.8, 0),
    ],
    "SER": [
        ("OG",  ("N","CA","CB"), 1.417, 111.2, 0),
    ],
    "THR": [
        ("OG1", ("N","CA","CB"), 1.433, 109.6, 0),
        ("CG2", ("N","CA","CB"), 1.521, 111.5, (0,  2.094)),  # chi1 + 120°
    ],
    "VAL": [
        ("CG1", ("N","CA","CB"), 1.521, 115.0, 0),
        ("CG2", ("N","CA","CB"), 1.521, 115.0, (0, -2.094)),  # chi1 − 120°
    ],
    "ILE": [
        ("CG1", ("N","CA","CB"),   1.540, 115.0, 0),
        ("CG2", ("N","CA","CB"),   1.521, 111.5, (0,  2.094)),  # chi1 + 120°
        ("CD1", ("CA","CB","CG1"), 1.521, 113.8, 1),
    ],
    "LEU": [
        ("CG",  ("N","CA","CB"),  1.520, 113.8, 0),
        ("CD1", ("CA","CB","CG"), 1.521, 115.0, 1),
        ("CD2", ("CA","CB","CG"), 1.521, 115.0, (1, -2.094)),  # chi2 − 120°
    ],
    "MET": [
        ("CG",  ("N","CA","CB"),  1.520, 113.8, 0),
        ("SD",  ("CA","CB","CG"), 1.803, 113.0, 1),
        ("CE",  ("CB","CG","SD"), 1.791, 100.9, 2),
    ],
    "PRO": [
        ("CG",  ("N","CA","CB"),  1.516, 104.5, 0),
        ("CD",  ("CA","CB","CG"), 1.516, 103.0, 1),
    ],
    "ASN": [
        ("CG",  ("N","CA","CB"),  1.520, 113.8, 0),
        ("OD1", ("CA","CB","CG"), 1.231, 120.8, 1),
        ("ND2", ("CA","CB","CG"), 1.328, 116.4, (1, _PI)),
    ],
    "ASP": [
        ("CG",  ("N","CA","CB"),  1.520, 113.8, 0),
        ("OD1", ("CA","CB","CG"), 1.249, 120.8, 1),
        ("OD2", ("CA","CB","CG"), 1.249, 120.8, (1, _PI)),
    ],
    "GLN": [
        ("CG",  ("N","CA","CB"),  1.520, 113.8, 0),
        ("CD",  ("CA","CB","CG"), 1.516, 111.3, 1),
        ("OE1", ("CB","CG","CD"), 1.231, 120.7, 2),
        ("NE2", ("CB","CG","CD"), 1.328, 116.5, (2, _PI)),
    ],
    "GLU": [
        ("CG",  ("N","CA","CB"),  1.520, 113.8, 0),
        ("CD",  ("CA","CB","CG"), 1.516, 111.3, 1),
        ("OE1", ("CB","CG","CD"), 1.249, 120.7, 2),
        ("OE2", ("CB","CG","CD"), 1.249, 120.7, (2, _PI)),
    ],
    "LYS": [
        ("CG",  ("N","CA","CB"),  1.520, 113.8, 0),
        ("CD",  ("CA","CB","CG"), 1.516, 111.3, 1),
        ("CE",  ("CB","CG","CD"), 1.516, 111.3, 2),
        ("NZ",  ("CG","CD","CE"), 1.489, 111.9, 3),
    ],
    "ARG": [
        ("CG",  ("N","CA","CB"),  1.520, 113.8, 0),
        ("CD",  ("CA","CB","CG"), 1.516, 111.3, 1),
        ("NE",  ("CB","CG","CD"), 1.460, 111.9, 2),
        ("CZ",  ("CG","CD","NE"), 1.329, 124.2, 3),
        ("NH1", ("CD","NE","CZ"), 1.326, 120.0, 0.0),
        ("NH2", ("CD","NE","CZ"), 1.326, 120.0, _PI),
    ],
    "HIS": [
        ("CG",  ("N","CA","CB"),   1.497, 113.8, 0),
        ("ND1", ("CA","CB","CG"),  1.378, 122.7, 1),
        ("CD2", ("CA","CB","CG"),  1.354, 130.4, (1, _PI)),
        ("CE1", ("CB","CG","ND1"), 1.321, 108.5, _PI),
        ("NE2", ("CB","CG","CD2"), 1.371, 107.4, _PI),
    ],
    "PHE": [
        ("CG",  ("N","CA","CB"),    1.502, 113.8, 0),
        ("CD1", ("CA","CB","CG"),   1.384, 120.7, 1),
        ("CD2", ("CA","CB","CG"),   1.384, 120.7, (1, _PI)),
        ("CE1", ("CB","CG","CD1"),  1.384, 120.0, _PI),
        ("CE2", ("CB","CG","CD2"),  1.384, 120.0, _PI),
        ("CZ",  ("CG","CD1","CE1"), 1.384, 120.0, 0.0),
    ],
    "TYR": [
        ("CG",  ("N","CA","CB"),    1.502, 113.8, 0),
        ("CD1", ("CA","CB","CG"),   1.384, 120.7, 1),
        ("CD2", ("CA","CB","CG"),   1.384, 120.7, (1, _PI)),
        ("CE1", ("CB","CG","CD1"),  1.384, 120.0, _PI),
        ("CE2", ("CB","CG","CD2"),  1.384, 120.0, _PI),
        ("CZ",  ("CG","CD1","CE1"), 1.384, 120.0, 0.0),
        ("OH",  ("CD1","CE1","CZ"), 1.362, 119.9, _PI),
    ],
    "TRP": [
        ("CG",  ("N","CA","CB"),     1.498, 113.8, 0),
        ("CD1", ("CA","CB","CG"),    1.365, 126.9, 1),
        ("CD2", ("CA","CB","CG"),    1.409, 126.8, (1, _PI)),
        ("NE1", ("CB","CG","CD1"),   1.374, 107.6, _PI),
        ("CE2", ("CB","CG","CD2"),   1.403, 107.2, _PI),
        ("CE3", ("CB","CG","CD2"),   1.398, 133.9, 0.0),
        ("CZ2", ("CG","CD2","CE2"),  1.394, 122.4, _PI),
        ("CZ3", ("CG","CD2","CE3"),  1.382, 118.6, _PI),
        ("CH2", ("CD2","CE2","CZ2"), 1.368, 117.5, 0.0),
    ],
    "ALA": [],
    "GLY": [],
}


def _resolve_dihedral(entry_dihedral, chi_angles: np.ndarray) -> float:
    """
    Resolve a build-sequence dihedral entry to a float in radians.
    int i            → chi_angles[i]
    float f          → f (fixed)
    (int i, float f) → chi_angles[i] + f   (i >= 0)
                     → -chi_angles[-i-1] + f  (i < 0: negate that chi angle)
    """
    d = entry_dihedral
    if isinstance(d, int):
        return float(chi_angles[d])
    if isinstance(d, tuple):
        chi_idx, offset = d
        if chi_idx < 0:
            return -float(chi_angles[-chi_idx - 1]) + float(offset)
        return float(chi_angles[chi_idx]) + float(offset)
    return float(d)


# ── NeRF atom placement ────────────────────────────────────────────────────────

def _place_atom(
    a: np.ndarray, b: np.ndarray, c: np.ndarray,
    bond_length: float, bond_angle_deg: float, dihedral: float,
) -> np.ndarray:
    """
    NeRF (Natural Extension Reference Frame): place atom D bonded to c, given
    three reference atoms (a, b, c), bond length c→D (Å), bond angle b–c–D (°),
    and dihedral a–b–c–D (rad).

    Builds a right-handed orthonormal frame at c aligned to the b→c bond, then
    expresses D in that frame using the bond geometry:

        D = c  −  bond_length·cos(θ) · along_bond
               +  bond_length·sin(θ) · [ cos(φ)·in_plane_perp
                                        + sin(φ)·plane_normal ]
    """
    bond_angle = math.radians(bond_angle_deg)

    # Unit vector along the b→c bond — the axis that D extends from
    along_bond = c - b
    along_bond_len = np.linalg.norm(along_bond)
    if along_bond_len < 1e-8:
        return c.copy()
    along_bond = along_bond / along_bond_len

    # Normal to the plane spanned by a, b, c
    plane_normal = np.cross(b - a, along_bond)
    plane_normal_len = np.linalg.norm(plane_normal)
    if plane_normal_len < 1e-8:
        # a, b, c are collinear — pick any perpendicular as a fallback
        fallback = np.array([0., 0., 1.]) if abs(along_bond[2]) < 0.9 else np.array([1., 0., 0.])
        plane_normal = np.cross(along_bond, fallback)
        plane_normal_len = np.linalg.norm(plane_normal)
    plane_normal = plane_normal / plane_normal_len

    # Third axis: in the a–b–c plane, perpendicular to along_bond
    in_plane_perp = np.cross(plane_normal, along_bond)

    return (
        c + bond_length * (
            -math.cos(bond_angle) * along_bond
            + math.sin(bond_angle) * (
                math.cos(dihedral) * in_plane_perp
                - math.sin(dihedral) * plane_normal
            )
        )
    ).astype(np.float32)


def reconstruct_sidechain(
    resname: str,
    chi_angles: np.ndarray,          # (N_CHI,) in radians
    bb_atoms: dict[str, np.ndarray], # must contain at least N, CA, CB
) -> dict[str, np.ndarray]:
    """
    Reconstruct sidechain atom positions from chi angles via NeRF.
    Returns {atom_name: xyz} including the supplied backbone atoms.
    """
    placed = dict(bb_atoms)
    for new_atom, refs, bond_length, bond_angle_deg, dihedral_spec in _SC_BUILD.get(resname, []):
        a = placed.get(refs[0])
        b = placed.get(refs[1])
        c = placed.get(refs[2])
        if a is None or b is None or c is None:
            break
        placed[new_atom] = _place_atom(a, b, c, bond_length, bond_angle_deg,
                                       _resolve_dihedral(dihedral_spec, chi_angles))
    return placed


# ── model ──────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: str) -> qFitLatent:
    """
    Rebuild a qFitLatent with the same architecture the checkpoint was trained
    with. Every shape-determining hyperparameter (single/pair widths, IPA head
    count, channels-per-head, IPA depth, and the chi-head mixture size k) is
    inferred from the state dict, so any of the trained runs loads without the
    caller having to remember its config.
    """
    sd  = torch.load(ckpt_path, map_location="cpu")["model"]
    d_s = sd["aa_embed.weight"].shape[1]
    d_z = sd["rel_pos_embed.weight"].shape[1]
    H   = sd["ipa_blocks.0.ipa.head_weight"].shape[0]
    c   = sd["ipa_blocks.0.ipa.q_s.weight"].shape[0] // H
    # number of IPA blocks = highest ipa_blocks.<i>. index present + 1
    n_ipa = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("ipa_blocks."))
    # each chi head's final Linear emits 3*k (pi, mu, kappa per component)
    k = sd["chi_head.heads.0.2.weight"].shape[0] // 3
    model = qFitLatent(d_single=d_s, d_pair=d_z, n_heads=H, c=c, n_ipa=n_ipa, k=k)
    model.load_state_dict(sd)
    return model.eval().to(device)


def load_confidence_model(ckpt_path: Path, device: str) -> ConfidenceModel:
    """
    Rebuild a ConfidenceModel the same way load_model() rebuilds a qFitLatent
    — hyperparameters inferred from the checkpoint's state dict.
    """
    sd  = torch.load(ckpt_path, map_location="cpu")["model"]
    d_s = sd["aa_embed.weight"].shape[1]
    d_z = sd["rel_pos_embed.weight"].shape[1]
    H   = sd["ipa_blocks.0.ipa.head_weight"].shape[0]
    c   = sd["ipa_blocks.0.ipa.q_s.weight"].shape[0] // H
    n_ipa = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("ipa_blocks."))
    model = ConfidenceModel(d_single=d_s, d_pair=d_z, n_heads=H, c=c, n_ipa=n_ipa)
    model.load_state_dict(sd)
    return model.eval().to(device)


@torch.no_grad()
def run_confidence(
    model: ConfidenceModel, sample: dict, device: str,
) -> torch.Tensor:
    """
    Forward pass: returns predicted per-residue W1 (Å) — the confidence
    model's estimate of how far qFitLatent's chi-mixture prediction will land
    from the true qFit ensemble, expm1'd back out of the log1p space it was
    trained in.
    """
    pred_log_w1 = model(
        sample["aa_tokens"].to(device),
        sample["R"].to(device),
        sample["t"].to(device),
    )
    return torch.expm1(pred_log_w1).clamp(min=0).cpu()


@torch.no_grad()
def run_inference(
    model: qFitLatent, sample: dict, device: str, K: int = 8, to_cpu: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass: returns the predicted per-residue mixture (pi, mu, kappa).

    pi  [N, K]      mixing weights of the top-K joint chi modes
    mu  [N, K, D]   von Mises means    (D = N_CHI = 4, radians)
    kappa [N, K, D] von Mises concentrations

    to_cpu=False keeps the result on `device` (useful for benchmarking the GPU
    forward pass without forcing a device→host copy).
    """
    pi, mu, kappa = model(
        sample["aa_tokens"].to(device),
        sample["R"].to(device),
        sample["t"].to(device),
        K=K,
    )
    if to_cpu:
        return pi.cpu(), mu.cpu(), kappa.cpu()
    return pi, mu, kappa


# ── PDB output ─────────────────────────────────────────────────────────────────

def _pdb_atom_name(name: str) -> str:
    return f" {name:<3s}" if len(name) < 4 else f"{name:<4s}"


def write_predicted_pdb(
    path: Path,
    pdb_path: Path,
    sample: dict,
    pi: torch.Tensor,
    mu: torch.Tensor,
    kappa: torch.Tensor,
    pi_thresh: float = 0.10,
) -> None:
    """
    Write a multiconformer PDB from predicted GMM chi angles.

    Backbone + CB atoms come from the primary (blank/A) conformer of the
    original PDB.  Sidechain atoms are reconstructed via pNeRF from each GMM
    component mean; components with mixing weight > pi_thresh are written as
    altlocs A/B/C/…
    """
    records = parse_pdb(pdb_path)
    keys    = sorted(records)
    aa      = sample["aa_tokens"].numpy()

    lines, serial = [], 1
    for i, key in enumerate(keys):
        resname  = AA_IDX.get(int(aa[i]), "UNK")
        chain_id = key[0]
        resseq   = key[1]
        alts     = records[key]["altlocs"]

        # Backbone + CB: blank altloc, from primary conformer
        bb_atoms: dict[str, np.ndarray] = {}
        for atom_name in ("N", "CA", "C", "O", "CB"):
            xyz = get_xyz(alts, atom_name)
            if xyz is not None:
                bb_atoms[atom_name] = xyz.copy()
                lines.append(
                    f"ATOM  {serial:5d} {_pdb_atom_name(atom_name)} "
                    f"{resname:3s} {chain_id}{resseq:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    f"  1.00  0.00\n"
                )
                serial += 1

        build_seq = _SC_BUILD.get(resname, [])
        if not build_seq or "CA" not in bb_atoms or "CB" not in bb_atoms:
            continue

        weights = pi[i].numpy()
        comps   = np.where(weights > pi_thresh)[0]
        if len(comps) == 0:
            comps = [int(weights.argmax())]

        for rank, k_comp in enumerate(comps):
            alt        = chr(ord("A") + rank)
            occ        = float(weights[k_comp])
            chi_angles = mu[i, k_comp].numpy()   # (N_CHI,)

            # kappa -> approx circular std (radians) for the b-factor column
            _sig = kappa[i, k_comp].clamp(min=1e-3).rsqrt()
            comp_sigma = float(_sig.mean()) if _sig.numel() > 1 else float(_sig)

            placed = reconstruct_sidechain(resname, chi_angles, bb_atoms)

            for atom_name, xyz in placed.items():
                if atom_name in ("N", "CA", "C", "O", "CB"):
                    continue
                lines.append(
                    f"ATOM  {serial:5d} {_pdb_atom_name(atom_name)}{alt}"
                    f"{resname:3s} {chain_id}{resseq:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    f"{occ:6.2f}{comp_sigma:6.2f}\n"
                )
                serial += 1

    path.write_text("".join(lines) + "END\n")


def write_confidence_csv(
    path: Path, pdb_path: Path, sample: dict, pred_w1: torch.Tensor,
) -> None:
    """
    Per-residue confidence, one row per residue in `sample`: the predicted W1
    (Å) between this prediction and the true qFit ensemble.
    """
    import csv
    records = parse_pdb(pdb_path)
    keys    = sorted(records)
    aa      = sample["aa_tokens"].numpy()
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["chain", "seq", "resname", "pred_w1"])
        for i, (chain_id, resseq) in enumerate(keys):
            writer.writerow(
                [chain_id, resseq, AA_IDX.get(int(aa[i]), "UNK"),
                 f"{float(pred_w1[i]):.4f}"]
            )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb",    type=Path, required=True,
                    help="Input PDB (backbone-only is sufficient)")
    ap.add_argument("--ckpt",   type=Path, default=DEFAULT_CKPT,
                    help=f"Model checkpoint (default: {DEFAULT_CKPT})")
    ap.add_argument("--out",    type=Path, default=None,
                    help="Output PDB path (default: <pdb_stem>_predicted.pdb)")
    ap.add_argument("--thresh", type=float, default=0.10,
                    help="Min mixing weight to write as altloc (default: 0.10)")
    ap.add_argument("--K",      type=int, default=8,
                    help="Number of joint chi modes to predict (default: 8)")
    ap.add_argument("--confidence_ckpt", type=Path, default=DEFAULT_CONFIDENCE_CKPT,
                    help="Confidence-model checkpoint predicting per-residue "
                         f"W1 (default: {DEFAULT_CONFIDENCE_CKPT}; skipped if absent)")
    ap.add_argument("--no_confidence", action="store_true",
                    help="Skip the confidence prediction even if the checkpoint exists")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Torch device (default: cuda if available else cpu)")
    args = ap.parse_args()

    from qfit_latent.data.data import ground_truth

    device = args.device
    model  = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt.name}  [{device}]")

    sample = ground_truth(args.pdb, max_len=None)
    if sample is None:
        print("failed to parse PDB"); return

    pi, mu, kappa = run_inference(model, sample, device, K=args.K)

    out = args.out or args.pdb.parent / f"{args.pdb.stem}_predicted.pdb"
    write_predicted_pdb(out, args.pdb, sample, pi, mu, kappa, pi_thresh=args.thresh)
    print(f"wrote → {out}  ({pi.shape[0]} residues)")

    # The confidence model (scripts/train_confidence.py) — how far (Å) is this
    # prediction likely to be from the true qFit ensemble? Trained separately
    # from the base model, so it is skipped quietly if there is no checkpoint.
    if not args.no_confidence and args.confidence_ckpt.exists():
        conf_model = load_confidence_model(args.confidence_ckpt, device)
        pred_w1 = run_confidence(conf_model, sample, device)
        conf_out = out.parent / f"{out.stem}_confidence.csv"
        write_confidence_csv(conf_out, args.pdb, sample, pred_w1)
        print(f"wrote → {conf_out}  (mean predicted W1 = {pred_w1.mean():.3f} Å)")


if __name__ == "__main__":
    main()
