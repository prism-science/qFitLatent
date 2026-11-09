# qFitLatent

A variational autoencoder for qFit multiconformer crystal structures. It compresses each residue's observed side-chain ensemble (altlocs and occupancies) into a small SE(3)-invariant latent and reconstructs that ensemble as a distribution over χ angles. The per-residue latent mean $\mu_z$ is a **dynamics representation**: it is learned from the conformational heterogeneity in experimental density, for use in downstream models.

---

## Architecture

### Inputs (per residue, `qfitlatent/featurize.py`)

| Symbol | Shape | Description |
|--------|-------|-------------|
| $a_i$ | $(N,)$ | amino acid token, $a_i \in \{0, \ldots, 20\}$ |
| $R_i, t_i$ | $(N, 3, 3), (N, 3)$ | backbone frame from $N, C_\alpha, C$ of the dominant altloc |
| $\phi_i, \psi_i$ | $(N,)$ | backbone dihedrals |
| $\chi_{ia}, o_{ia}$ | $(N, 6, 4), (N, 6)$ | up to 6 altlocs: $\sin/\cos$ of 4 χ angles + occupancy |

### Model (`qfitlatent/model.py`)

- **Ensemble encoder.** A set transformer over each residue's altloc tokens: self-attention, then attention pooling with 4 learned seeds. It is permutation invariant across altlocs. Hidden or unmodeled residues get a learned mask token.
- **Trunk.** The single track is the aa embedding plus the ensemble code ($d_s = 256$). The pair track holds relative position, $C_\alpha$–$C_\alpha$ distance and the outer product ($d_z = 64$). These feed 8 IPA blocks with triangle updates.
- **Bottleneck.** A per-residue Gaussian latent $z \in \mathbb{R}^{32}$, sampled in training and set to $\mu_z$ at inference.
- **Decoder** (`direct_head.py`). From $z$, the aa embedding and $\sin/\cos(\phi, \psi)$, it predicts a $K = 16$ von Mises mixture over the χ torus:

$$p(\chi_i \mid z_i) = \sum_{k=1}^{K} w_{ik} \prod_{d=1}^{D_i} \mathrm{vM}\!\left(\chi_{d} \mid \mu_{ikd}, \kappa_{ikd}\right)$$

### Loss (`qfitlatent/losses.py`)

During training, 15% of residues are hidden and must be reconstructed from context:

$$\mathcal{L} = \mathrm{NLL}_{\mathrm{all}} + \beta\,\mathrm{KL} + \lambda_H H(w) + \lambda_{\mathrm{OT}}\,\mathrm{OT} + \lambda_{\mathrm{occ}}\,\mathrm{OCC} + \lambda_m \mathrm{NLL}_{\mathrm{masked}}$$

- **NLL:** the occupancy-weighted mixture NLL of the observed altlocs. π-symmetric χ are folded.
- **KL:** the term against $\mathcal{N}(0, I)$.
- **$H(w)$:** mixture-weight entropy.
- **OT:** Sinkhorn transport between modes and altlocs on the torus.
- **OCC:** matching mode weights to altloc occupancies.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

Training takes qFit multiconformer PDBs, given as a directory or as list files with one path per line.

```bash
# optional: fetch canonical sequences from RCSB so unmodeled residues are placed correctly
python scripts/fetch_sequences.py --structures-dir data/structures

python scripts/train.py --train-list train.txt --val-list val.txt \
  --d-latent 32 --beta 0.02 --entropy 0.3 --sinkhorn 1.0 --occ-match 5 \
  --mask-frac 0.15 --val-mask-frac 0.15 --epochs 12 --kl-warmup 2 \
  --checkpoint runs/model.pt --log runs/model.csv
```

The released model was trained in two parts:

1. 12 epochs from scratch with the command above.
2. Three 4-epoch stages, each warm-started from the previous one with `--init-from <checkpoint> --masked-weight 1 --epochs 4 --kl-warmup 1`.

Use `--cache-dir` to cache features and `--workers` to featurize in parallel. A GPU is recommended. The best validation epoch is saved as `runs/model.best.pt`.

---

## Inference: extract embeddings

```bash
python scripts/extract_embeddings.py --checkpoint runs/model.best.pt --pdb 1abc.pdb --out embeddings/
```

The input is a qFit multiconformer PDB. Every chain is encoded with its altloc ensembles visible. The script writes `embeddings/1abc.npz` with these aligned per-residue arrays:

| key | shape | description |
|-----|-------|-------------|
| `emb` | $[N, 32]$ | latent mean $\mu_z$ (the dynamics embedding) |
| `logvar` | $[N, 32]$ | latent log-variance |
| `chain`, `resseq`, `resname` | $[N]$ | residue identity (`resseq = -1` for unmodeled positions) |
| `aa_token`, `resolved` | $[N]$ | amino acid token and whether the residue is modeled |

If `--seq-dir` holds the chain's canonical sequence from `fetch_sequences.py`, unmodeled positions are included and encoded from context. Otherwise only the modeled residues are used. The same is available from Python:

```python
from qfitlatent.inference import load_model, embed_pdb
model = load_model("runs/model.best.pt")
arrays = embed_pdb(model, "1abc.pdb")
```
