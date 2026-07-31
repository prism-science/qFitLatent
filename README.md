# qFitLatent

SE(3)-invariant model that predicts per-residue sidechain torsion distributions from backbone structure and sequence, trained on qFit multiconformer crystal structures.

---

## Architecture

### Inputs

| Symbol | Shape | Description |
|--------|-------|-------------|
| $a_i$ | $(N,)$ | amino acid tokens, $a_i \in \{0, \ldots, 20\}$ |
| $R_i$ | $(N, 3, 3)$ | backbone rotation matrix (to local frame, primary altloc) |
| $t_i$ | $(N, 3)$ | $C_\alpha$ position (frame origin) |

Local frames are constructed from $N$, $C_\alpha$, $C$ atoms of the primary (blank) altloc:

$$\hat{x} = \frac{C - C_\alpha}{\| C - C_\alpha \|}, \quad \hat{z} = \frac{\hat{x} \times (N - C_\alpha)}{\| \hat{x} \times (N - C_\alpha) \|}, \quad \hat{y} = \hat{z} \times \hat{x}$$

so that $R_i = [\hat{x} \; \hat{y} \; \hat{z}]$ is a proper orthonormal frame that puts the $C_\alpha$ at the origin, the $C$ along the positive x-axis, and the $N$ in the xy-plane with positive y.

### Output head (autoregressive von Mises mixture)

The final single representation $s_i$ conditions an **autoregressive** chain over the four sidechain $\chi$ angles. Each $\chi_d$ has its own $k$-component von Mises mixture whose parameters depend on $s_i$ **and** the previous angles $\chi_{\lt d}$ (fed in as $\sin/\cos$ features):

$$p(\chi_i \mid s_i) = \prod_{d=1}^{D_\chi} \sum_{k=1}^{K} \pi_{dk}(s_i, \chi_{\lt d})\; \mathrm{vM}\!\left(\chi_d \mid \mu_{dk}, \kappa_{dk}\right)$$

with $D_\chi = 4$ and, per head, $\pi = \mathrm{softmax}(\cdot)$, $\mu = \pi\tanh(\cdot) \in (-\pi, \pi]$, $\kappa = \mathrm{softplus}(\cdot) + \kappa_{\min} \gt 0$ (the circular analogue of inverse variance). A single rotameric mode is therefore a *sequence* of component choices $(k_1, \ldots, k_4)$, so one mode can natively express correlated $\chi$ (e.g. ARG $g^+/g^+/t/g^+$) — unlike the earlier factorised mixture, whose components used independent per-$\chi$ parameters. The von Mises form (replacing an earlier wrapped Gaussian) removes the variance-floor hack and gives a properly normalized density on the torus.

---

## Training

**Data.** qFit multiconformer PDB files. Named altlocs (A, B, $\ldots$) with crystallographic occupancies provide the discrete ground-truth ensemble; the blank altloc provides the input backbone frame.

**Loss.** Occupancy-weighted negative log-likelihood of the observed altloc $\chi$ angles, with the density factored autoregressively and **teacher-forced**: the conditioning $\chi_{a,\lt d}$ at step $d$ uses altloc $a$'s own observed earlier angles. For a single residue with $A$ observed altlocs:

$$\mathcal{L} = - \sum_{a=1}^{A} o_a \sum_{d=1}^{D} \log \sum_{k=1}^{K} \pi_{dk}(\chi_{a,\lt d})\, \mathrm{vM}\!\left(\chi_{ad} \mid \mu_{dk}, \kappa_{dk}\right), \qquad \mathrm{vM}(\chi \mid \mu, \kappa) = \frac{\exp(\kappa \cos(\chi - \mu))}{2\pi I_0(\kappa)}$$

averaged over the valid residue set $\mathcal{R} = \{ i : n_{\mathrm{obs},i} \geq 2,\ d_{\mathrm{eff},i} \geq 1 \}$ (at least two observed altlocs and one defined $\chi$). Here $o_a$ is the normalized occupancy ($\sum_a o_a = 1$), $D$ the number of defined $\chi$ for the residue (per-$\chi$ validity mask), and $I_0$ the order-0 modified Bessel function — its log-normalizer is evaluated stably as $\log I_0(\kappa) = \log\,\mathrm{i0e}(\kappa) + \kappa$ (`torch.special.i0e`). The residual $\chi_{ad} - \mu_{dk}$ is taken on the circle; for $\pi$-symmetric $\chi$ (ASP $\chi_2$, GLU $\chi_3$, PHE $\chi_2$, TYR $\chi_2$) it is folded to $(-\pi/2, \pi/2]$ via $\tfrac{1}{2}\,\mathrm{atan2}(\sin 2\Delta, \cos 2\Delta)$.

**Optimization.** AdamW with $\eta = 3 \times 10^{-4}$, cosine annealing to $\eta / 100$ over the full training run. Batch size 1 (one full protein per step).

---

## Inference

```bash
python scripts/inference.py --pdb backbone.pdb --ckpt checkpoints/vmm_ar_k8/latest.pt --out pred.pdb
```

Only the backbone ($N, C_\alpha, C$) and sequence are used, so a **backbone-only PDB is a valid input**. The autoregressive head is unrolled by ancestral beam search to the top-$K$ joint $\chi$ modes; components with mixing weight above `--thresh` (default $0.10$) are written as altlocs A/B/C…, with sidechain atoms rebuilt from the predicted $\chi$ via NeRF. The checkpoint's architecture (including $k$) is inferred on load, so any trained run works. See `qfl_evaluation/benchmark/` for forward-pass latency benchmarking.

### Confidence model

`ConfidenceModel` scores a prediction *before* you have anything to compare it to: it predicts the per-residue $W_1$ (Å) between qFitLatent's $\chi$ mixture and the true qFit ensemble, from backbone and sequence alone. It is trained separately from the base model and is entirely optional — `inference.py` writes `<out>_confidence.csv` when a checkpoint is present and skips quietly when it isn't.

```bash
# 1. label: measure the real W1 of a frozen checkpoint's own predictions
python scripts/make_w1_labels.py --ckpt checkpoints/vmm_ar_k8/latest.pt \
    --out qfit_latent/data/w1_labels.csv

# 2. train against those labels
python scripts/train_confidence.py --w1_csv qfit_latent/data/w1_labels.csv \
    --checkpoint_dir checkpoints/confidence
```

The model reuses qFitLatent's IPA trunk and swaps the $\chi$-mixture head for a small MLP (`ConfidenceHead`), AlphaFold-pLDDT style — there is no mixture to condition on, so it sees only the backbone frames and sequence. Regression is on $\log(1+W_1)$ under a smooth-L1 loss, which keeps a handful of very flexible, large-$W_1$ residues from dominating; `inference.py` `expm1`s the prediction back into Å. Because the targets come from a *frozen* base checkpoint's own predictions, the labels are specific to that checkpoint — rerunning `make_w1_labels.py` is required if the base model is retrained.

### Per-residue embeddings

```bash
python scripts/extract_embeddings.py --pdb backbone.pdb --ckpt checkpoints/vmm_ar_k8/latest.pt --out emb.npz
```

Writes the single representation $s_i$ from the final IPA block — the SE(3)-invariant per-residue vector the chi head reads — as an `.npz` with `emb` $[N, d_s]$ plus aligned `chain` / `resseq` / `resname` / `aa_token` arrays. Load with `np.load(..., allow_pickle=True)`.

---

## Default hyperparameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| single representation | $d_s$ | $128$ |
| pair representation | $d_z$ | $32$ |
| IPA blocks | $L$ | $8$ |
| attention heads | $H$ | $4$ |
| head dim | $c$ | $8$ |
| query/key points | $N_{qk}$ | $4$ |
| value points | $N_v$ | $8$ |
| von Mises components | $k$ | $5$ |
| chi dimensions | $D_\chi$ | $4$ |
| concentration floor | $\kappa_{\min}$ | $0.01$ |
| dropout | $p$ | $0.2$ |
| optimizer | --- | AdamW |
| learning rate | $\eta$ | $3 \times 10^{-4}$ |
| schedule | --- | cosine to $\eta / 100$ |
| batch size | --- | $1$ |
