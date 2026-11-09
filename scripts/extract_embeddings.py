# created by clay
'''
Extract per-residue dynamics embeddings (the latent mean mu_z) from qFit
multiconformer PDBs with a trained checkpoint.

Writes one .npz per structure with aligned per-residue arrays:
    emb      [N, d_latent]  latent mean mu_z, the dynamics representation
    logvar   [N, d_latent]  latent log-variance
    chain, resseq, resname, aa_token, resolved
resseq is the author residue number, -1 for positions in the canonical sequence
that the structure does not model (only present when --seq-dir has its sequence).

    python scripts/extract_embeddings.py --checkpoint model.pt --pdb 1abc.pdb --out emb/
'''

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qfitlatent.featurize import DEFAULT_SEQ_DIR
from qfitlatent.inference import embed_pdb, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="trained train.py checkpoint")
    parser.add_argument("--pdb", nargs="+", required=True, help="qFit multiconformer PDB(s)")
    parser.add_argument("--out", default="embeddings", help="output directory")
    parser.add_argument("--seq-dir", default=str(DEFAULT_SEQ_DIR),
                        help="canonical sequences from fetch_sequences.py (optional)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdb_path in args.pdb:
        arrays = embed_pdb(model, pdb_path, args.seq_dir, args.device)
        out_path = out_dir / f"{Path(pdb_path).stem}.npz"
        np.savez(out_path, **arrays)
        print(f"{pdb_path}: {arrays['emb'].shape[0]} residues -> {out_path}")


if __name__ == "__main__":
    main()
