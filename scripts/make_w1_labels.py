# created by clay
'''
Precompute per-residue W1 labels for the confidence model: the Wasserstein
distance between a frozen qFitLatent checkpoint's predicted chi mixture
(inference-mode beam search) and the true qFit conformer ensemble. These are
the regression targets scripts/train_confidence.py trains against.

Usage:
    python3 scripts/make_w1_labels.py --ckpt checkpoints/vmm_ar_k8/latest.pt
'''
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

torch.set_num_threads(1)  # tiny per-protein tensors -- see validation/confidence_external_contacts.py

from qfit_latent.data.data import parse_pdb, load_subset_paths, N_CHI_PER_AA
from validate import load_model, featurize, per_residue_metrics


def label_one_pdb(model, p, device):
    records = parse_pdb(p)
    keys, aa_tokens, R, t = featurize(records)
    if len(keys) < 3:
        return []
    with torch.no_grad():
        pi, mu, _ = model(torch.from_numpy(aa_tokens).to(device),
                          torch.from_numpy(R).to(device),
                          torch.from_numpy(t).to(device))
    w1, _ = per_residue_metrics(records, keys, aa_tokens, pi, mu)
    return [
        dict(pdb=p.stem, chain=key[0], seq=key[1], w1=float(w1[i]))
        for i, key in enumerate(keys)
        if N_CHI_PER_AA[int(aa_tokens[i])] > 0 and w1[i] == w1[i]  # finite
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/vmm_ar_k8/latest.pt")
    ap.add_argument("--train_csv", default="qfit_latent/data/train_subset.csv")
    ap.add_argument("--val_csv", default="qfit_latent/data/val_subset.csv")
    ap.add_argument("--structures_dir", default="../qfit_data/structures")
    ap.add_argument("--out", default="qfit_latent/data/w1_labels.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = load_subset_paths(Path(args.train_csv), Path(args.structures_dir))
    if args.val_csv:
        paths += load_subset_paths(Path(args.val_csv), Path(args.structures_dir))
    print(f"device: {device} | {len(paths)} pdbs | ckpt: {args.ckpt}")

    model = load_model(Path(args.ckpt), device)

    rows = []
    for i, p in enumerate(paths):
        try:
            rows += label_one_pdb(model, p, device)
        except Exception as e:
            print(f"  [{i}] {p.stem}: SKIPPED ({e})")
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(paths)}] {len(rows)} labeled residues so far")

    import csv
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["pdb", "chain", "seq", "w1"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} residue labels from {len(paths)} pdbs -> {out_path}")


if __name__ == "__main__":
    main()
