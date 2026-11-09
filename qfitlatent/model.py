# created by clay
'''
A variational autoencoder over qFit multiconformer sidechain dynamics, with an
IPA backbone.

Goal: a smooth representation space for the conformational dynamics carried by
the qFit altlocs. Every residue's observed ensemble is encoded (through the IPA
trunk, which conditions each residue on its structural environment) into a small
per-residue latent z. The decoder (DirectChiHead) places free conformers on the chi
torus from z plus the residue's amino acid and backbone (phi, psi), so z is forced to
encode the residue's dynamics-beyond-backbone. A KL term to N(0, I) keeps the latent space
smooth and samplable. The whole ensemble is reconstructed (not a single masked
residue), so this is a reconstruction autoencoder, not an imputation model.

A separate downstream step (post-training latent-space math) maps a static
(top-altloc) encoding to its multiconformer counterpart in this same space, so a
single-conformer structure can be converted to a predicted ensemble.
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from .featurize import N_CHI
from .environment import NeighbourAtomEncoder
from .direct_head import DirectChiHead
from .rotamers import chi_valid
# the objectives live in losses.py; re-exported so `from .model import ...` still works
from .losses import (  # noqa: F401
    KAPPA_MIN, CONF_BINS, CONF_MAX, RSCC_BINS, RSCC_MAX, _SYM_CHI, ensemble_nll,
    kl_standard_normal, occ_entropy, mixture_entropy, sinkhorn_ot,
    occupancy_match_loss, slot_cross_entropy, stray_mass_loss, matched_atom_loss,
    confidence_loss, pick_mask_indices,
)

class qFitLatent(nn.Module):
    '''
    SE(3) invariant variational autoencoder for a representation space of qFit
    conformational dynamics.

    Inputs (all from featurize.py, one protein chain):
        - aa_tokens: (N)
        - R: (N, 3, 3), t: (N, 3)
        - chi_angles: (N, A, N_CHI), occupancies: (N, A), chi_mask: (N, N_CHI)
        - phi: (N), psi: (N), bb_mask: (N, 2)
        - hide: (N) residues encoded from context only (unresolved; no ensemble)
        - sc_atoms: (N, A, MAXATOM, 3) altloc atoms, read only when env_neighbours > 0

    Outputs:
        log_p: (N, K) log weights of the K predicted conformers
        mu: (N, K, N_CHI), sigma: (N, K, N_CHI) conformer chi means / std devs
        valid: (N, N_CHI) chi angles the library parameterizes
        mu_z: (N, d_latent), logvar_z: (N, d_latent) posterior over the latent
        conf: (N, CONF_BINS) log distribution over binned per-residue W1 error
        prscc: (N, RSCC_BINS) log distribution over the binned side-chain RSCC of
            the reconstruction against the experimental map (pRSCC)

    Pair with ensemble_nll(...) for reconstruction and kl_standard_normal(mu_z,
    logvar_z) for the VAE regularizer. z (= mu_z at eval) is the dynamics
    representation. confidence_head.expected(conf) gives the predicted W1 per
    residue; confidence_loss(...) trains it against the actual W1. Likewise
    rscc_head.expected(prscc) is the predicted side-chain RSCC, trained with
    confidence_loss(..., max_value=RSCC_MAX) against the RSCC of a frozen model's
    reconstructions (scripts/build_rscc_targets.py).
    '''

    # initialize with hyperparams
    def __init__(
        self,
        n_aa = 21,
        d_single = 128,
        d_pair = 32,
        d_latent = 16,
        n_ipa = 8,
        n_heads = 8,
        n_geom_attn_qpts = 4,
        n_geom_attn_vpts = 8,
        c = 8,
        dropout = 0.0,
        n_conf = 16,        # number of free conformers the decoder predicts
        env_neighbours = 0, # >0: encode the side-chain atoms of this many neighbours
        prscc_input = "latent",  # pRSCC head input: "latent" (decoder features),
                                 # "trunk" (+ trunk features), "trunk_mix" (+ mixture
                                 # entropy and top weight)
    ):
        super().__init__() # inherit nn
        # set representation sizes
        self.d_s = d_single
        self.d_z = d_pair
        self.d_lat = d_latent

        # set architecture sizes:
        # discrete vars:
        # embedding table for 21 amino acids to the residue rep dimension
        self.aa_embed = nn.Embedding(n_aa, d_single) # [21, d_single]
        # embedding table for primary seq diatance (capped at +- 64 distance)
        self.rel_pos_embed = nn.Embedding(129, d_pair) # [129, d_pair]
        # continous vars:
        # linear layer for pairwise distance projection CA-CA
        self.dist_embed = nn.Linear(1, d_pair, bias=False) # [1, d_pair]
        # projection of a single res representation to pair dim
        self.env_proj = nn.Linear(d_single, d_pair) # [d_single, d_pair]
        # projection of the pair representation to d_pair
        self.env_pair_proj = nn.Linear(d_pair, d_pair) # [d_pair, d_pair]

        # IPA block(s) hyperparameters
        ipa_hyperparams = dict(
            n_heads = n_heads,
            n_geom_attn_qpts = n_geom_attn_qpts,
            n_geom_attn_vpts = n_geom_attn_vpts,
            c=c
        )
        # build the IPA layers
        self.ipa_blocks = nn.ModuleList(
            [IPABlock(d_single, d_pair, dropout=dropout, **ipa_hyperparams)
            for _ in range(n_ipa)]
        )

        # encoder for the observed qFit ensembles (unresolved residues get a token)
        self.ensemble_encoder = EnsembleEncoder(d_single)
        # optional local atomic environment (neighbouring side-chain atoms)
        self.env_encoder = (NeighbourAtomEncoder(d_single, env_neighbours)
                            if env_neighbours > 0 else None)

        # variational bottleneck: compress the trunk features to a small per-residue
        # latent and expand back. besides the residue's amino acid and backbone, the
        # decoder sees ONLY this latent, so its ensemble must flow through z - forcing
        # z to encode the residue's conformational dynamics (and nothing the amino acid
        # and backbone already explain). KL to N(0, I) keeps the space smooth.
        self.to_latent = nn.Linear(d_single, 2 * d_latent)
        self.from_latent = nn.Sequential(
            nn.Linear(d_latent, d_single), nn.GELU(), nn.Linear(d_single, d_single),
        )

        # decoder readout (driven by the latent, not the trunk features directly):
        # n_conf free conformers placed anywhere on the chi torus, so ensembles can
        # sit between the rotamer library's wells
        self.n_conf = n_conf
        self.direct_head = DirectChiHead(d_single, n_conf, n_aa=n_aa)
        # AlphaFold-style confidence: predicts the model's own per-residue W1 error
        self.confidence_head = ConfidenceHead(d_single)
        # pRSCC: predicts how well the reconstructed side chain fits the density
        self.prscc_input = prscc_input
        d_prscc = {"latent": d_single, "trunk": 2 * d_single,
                   "trunk_mix": 2 * d_single + 2}[prscc_input]
        self.rscc_head = ConfidenceHead(d_prscc, n_bins=RSCC_BINS, max_value=RSCC_MAX)
        # ensemble-heterogeneity head: predicts the residue's conformational entropy
        # FROM the latent z (gradient flows into z, unlike the confidence head). This
        # forces z to carry ensemble-specific information the top-altloc input cannot
        # supply, so the dynamics encoding stops collapsing onto the static one.
        self.hetero_head = nn.Sequential(
            nn.Linear(d_latent, d_latent), nn.GELU(), nn.Linear(d_latent, 1))

    # initialize the pair representation track
    def pair_init(self, s, t, pad_mask):
        # batch and number of residues
        B, N = s.shape[0], s.shape[1]
        idx = torch.arange(N, device=t.device)

        # the relative position distances (clamped to +-64)
        rel_pos = (idx[:, None] - idx[None, :]).clamp(-64, 64) + 64
        # embed the relative positions (shared across the batch)
        z = self.rel_pos_embed(rel_pos)                  # [N, N, d_pair]
        z = z.unsqueeze(0).expand(B, -1, -1, -1)         # [B, N, N, d_pair]
        # get the pairwise distances of CA's (L2 Norm) keep dim for proj
        dist = (t[:, :, None] - t[:, None, :]).norm(dim=-1, keepdim=True)  # [B,N,N,1]
        # add the distance embeddings to the rel pos embeddings
        z = z + self.dist_embed(dist / 10.0) # angstroms to nm
        # project the single representations
        env = self.env_proj(s) # [B, N, d_single] to [B, N, d_pair]
        # add the pair projection products to the pair rep
        z = z + self.env_pair_proj(env[:, :, None, :] * env[:, None, :, :])
        # zero pairs touching a padded residue so padding never enters the track
        pair_pad = (pad_mask[:, :, None] & pad_mask[:, None, :]).unsqueeze(-1)
        return z * pair_pad  # [B, N, N, d_pair]

    # a forward pass
    def forward(
        self,
        aa_tokens,      # [B, N]
        R,              # [B, N, 3, 3]
        t,              # [B, N, 3]
        chi_angles,     # [B, N, A, N_CHI]
        occupancies,    # [B, N, A]
        chi_mask,       # [B, N, N_CHI]
        phi,            # [B, N]
        psi,            # [B, N]
        bb_mask,        # [B, N, 2]
        hide,           # [B, N] bool, residues with no ensemble to encode
                        #   (unresolved, padding) - get a learned mask token
        pad_mask=None,  # [B, N] bool, True for real residues (None = all real)
        sc_atoms=None,  # [B, N, A, MAXATOM, 3] altloc atoms (env encoder only)
    ):
        B, N = aa_tokens.shape
        if pad_mask is None:
            pad_mask = torch.ones(B, N, dtype=torch.bool, device=aa_tokens.device)

        # embed sequence tokens plus the visible ensembles, prepare pair track
        s = self.aa_embed(aa_tokens)
        s = s + self.ensemble_encoder(chi_angles, occupancies, chi_mask, hide)
        if self.env_encoder is not None:
            s = s + self.env_encoder(R, t, sc_atoms, occupancies, aa_tokens,
                                     pad_mask & ~hide)
        z = self.pair_init(s, t, pad_mask)

        # iterate through blocks and pass info; pair track is now updated
        # in-place by each block via the triangle multiplication
        for block in self.ipa_blocks:
            if self.training: # load from checkpoint and resume
                s, z = _checkpoint(block, s, z, R, t, pad_mask, use_reentrant=False)
            else: # forward pass
                s, z = block(s, z, R, t, pad_mask)

        # chi validity is a per-residue, 1-D op: flatten the batch, then restore
        flat = lambda x: x.reshape(B * N, *x.shape[2:])
        valid = chi_valid(flat(aa_tokens), flat(chi_mask)).view(B, N, N_CHI)

        # variational bottleneck: trunk features -> per-residue latent posterior.
        # sample during training (reparameterization); use the mean at eval so the
        # representation and its reconstruction are deterministic.
        mu_z, logvar_z = self.to_latent(s).chunk(2, dim=-1)
        # keep the latent log-variance in a sane range so exp() can't overflow to inf
        # (matters when warm-starting a low-beta latent into a new objective)
        logvar_z = logvar_z.clamp(-10.0, 10.0)
        if self.training:
            z = mu_z + torch.randn_like(mu_z) * torch.exp(0.5 * logvar_z)
        else:
            z = mu_z
        h = self.from_latent(z)              # decoder features, driven by z alone

        log_p, mu, sigma = self.direct_head(h, aa_tokens, phi, psi, bb_mask)
        # AF-style confidence: predicted distribution over per-residue W1 bins.
        # stop-gradient on the features so the confidence loss trains only the head
        # and never perturbs the reconstruction it is scoring.
        conf = self.confidence_head(h.detach())   # [B, N, CONF_BINS] log p(W1 bin)
        prscc = self.rscc_head(self.prscc_features(s, h, log_p).detach())
        # predicted per-residue conformational entropy; grad flows into z on purpose
        hetero = self.hetero_head(z).squeeze(-1)  # [B, N]
        return log_p, mu, sigma, valid, mu_z, logvar_z, conf, hetero, prscc

    def prscc_features(self, s, h, log_p):
        '''pRSCC head input: decoder features h, optionally with the trunk features s
        and the predicted mixture's entropy and top weight.'''
        if self.prscc_input == "latent":
            return h
        if self.prscc_input == "trunk":
            return torch.cat([s, h], -1)
        p = log_p.exp()
        entropy = -(p * log_p.clamp_min(-1e4)).sum(-1, keepdim=True)
        return torch.cat([s, h, entropy, p.amax(-1, keepdim=True)], -1)


class _SetAttnBlock(nn.Module):
    '''
    Self-attention block over a set of tokens with NO positional encoding, so it is
    permutation equivariant (Set-Transformer SAB). Lets a residue's altlocs attend to
    each other, so the pooled code can represent "there are two distinct conformers
    here", not just their average.
    '''
    def __init__(self, d, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x, key_padding_mask):
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        return x


class EnsembleEncoder(nn.Module):
    '''
    Permutation-invariant SET encoder over a residue's observed qFit altlocs.

    The previous encoder pooled the altlocs by an occupancy-weighted SUM. That blurs a
    multi-conformer residue's distinct rotamers into a single averaged point (two
    altlocs at 𝝌 +/-60 average, in sin/cos space, to one low-concentration mode) -
    destroying the very multi-conformer structure the decoder is scored against, and
    doing so BEFORE the variational bottleneck. Opening the bottleneck therefore fixed
    single- but not multi-conformer reconstruction: the multi-conformer signal was
    already gone one layer upstream.

    This keeps ONE token per altloc, lets the altlocs attend to each other (self-
    attention, no positional encoding -> permutation invariant over altloc order), then
    pools them with learned-seed multihead attention (Set-Transformer PMA). The seed
    queries can attend selectively, so the pooled code preserves the number, positions
    and populations of the conformers instead of averaging them away. Each altloc token
    still carries its own occupancy, so the populations survive into z as well.

    Residues with no observed ensemble (`hide`) get a learned mask token, so their
    latent is inferred from structural context rather than a missing ensemble.

    Inputs / Output shapes unchanged (rank-agnostic; altloc axis is dim -2 of
    chi_angles): chi_angles [..., A, N_CHI], occupancies [..., A], chi_mask
    [..., N_CHI], hide [...] -> e [..., d_single].
    '''
    def __init__(self, d_single, n_chi=N_CHI, n_seeds=4, n_heads=4):
        super().__init__() # inherit module
        self.d = d_single
        self.token = nn.Linear(2 * n_chi + 1, d_single)     # per-altloc embedding
        self.sab = _SetAttnBlock(d_single, n_heads)         # altlocs see each other
        self.seed = nn.Parameter(torch.randn(n_seeds, d_single) * 0.02)
        self.pma = nn.MultiheadAttention(d_single, n_heads, batch_first=True)
        self.pma_norm = nn.LayerNorm(d_single)
        self.out = nn.Linear(n_seeds * d_single, d_single)
        # start silent: an untrained encoder feeds no ensemble signal to the trunk, so
        # the decoder begins from the amino acid and backbone alone
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # learned stand-in for a hidden ensemble
        self.mask_token = nn.Parameter(torch.zeros(d_single))

    def forward(self, chi_angles, occupancies, chi_mask, hide):
        lead = chi_angles.shape[:-2]                         # [..., A, N_CHI] -> [...]
        A = chi_angles.shape[-2]
        # zero out chi the residue does not have, then one feature vector per altloc
        m = chi_mask.unsqueeze(-2).to(chi_angles.dtype)      # [..., 1, N_CHI]
        feats = torch.cat(
            [torch.sin(chi_angles) * m, torch.cos(chi_angles) * m,
             occupancies[..., None]],
            dim=-1,
        )                                                    # [..., A, 2*n_chi + 1]
        x = self.token(feats).reshape(-1, A, self.d)         # [M, A, d]
        valid = (occupancies > 0).reshape(-1, A)             # [M, A] real altloc slots
        # guarantee >=1 unmasked key per row so the attention softmax never sees an
        # all -inf row (which yields nan); fully-empty rows are hidden residues whose
        # output is overwritten by the mask token below anyway
        empty = ~valid.any(-1)
        valid = valid.clone()
        valid[empty, 0] = True
        kpm = ~valid                                         # True = ignore (padding)
        x = self.sab(x, kpm)
        M = x.shape[0]
        q = self.seed.unsqueeze(0).expand(M, -1, -1)         # [M, n_seeds, d]
        pooled, _ = self.pma(q, x, x, key_padding_mask=kpm, need_weights=False)
        pooled = self.pma_norm(pooled)                       # [M, n_seeds, d]
        e = self.out(pooled.reshape(M, -1)).reshape(*lead, self.d)
        return torch.where(hide[..., None], self.mask_token.expand_as(e), e)


class ConfidenceHead(nn.Module):
    '''
    AlphaFold-style per-residue confidence head. From the decoder features it
    predicts a distribution over a binned per-residue quality score, exactly as AF2
    predicts binned lDDT (pLDDT) / aligned error (PAE). Used twice: for the W1
    between the predicted and observed ensembles (bins over [0, pi] radians) and for
    the side-chain RSCC of the reconstruction (pRSCC, bins over [0, 1]). The scalar
    score is the expectation sum_b p_b * bin_center_b.

    Trained (via confidence_loss) against the actual, detached per-residue value, so
    it never perturbs the reconstruction it is scoring.
    '''
    def __init__(self, d_single, n_bins=CONF_BINS, max_value=CONF_MAX, hidden=None):
        super().__init__()
        hidden = hidden or d_single // 2
        self.net = nn.Sequential(
            nn.Linear(d_single, hidden), nn.GELU(), nn.Linear(hidden, n_bins))
        centers = (torch.arange(n_bins) + 0.5) * (max_value / n_bins)
        self.register_buffer("bin_centers", centers)   # [n_bins]
        self.n_bins, self.max_value = n_bins, max_value

    def forward(self, h):
        return torch.log_softmax(self.net(h), dim=-1)  # [..., n_bins] log p(bin)

    def expected(self, log_p_bins):
        '''Expected score: sum_b p_b * center_b.'''
        return (log_p_bins.exp() * self.bin_centers).sum(-1)



class IPA(nn.Module):
    '''
    Invariant point attention to make use of standard multi head
    self attention  plus geometric attention allowing the model to 
    learn a combined representation with structure and sequence priors

    For a query residue i and key residue j:

    a_ij = standard attention + geometric attention + pair bias
    a_ij = q_i * k_j / sqrt(c) +
           (-𝛾_h / 2) * Sum(pts) dist(qi_pts, kj_pts) + 
           b_ij

    Inputs:
        - s: sequence representations from embedding or previous layer
        - z: pair representation
        - R: local frame rotation
        - t: local frame translation

    Outputs:
        - s: updated features from attn mechanism
    '''

    def __init__(
        self, 
        d_single, 
        d_pair, 
        n_heads = 8, 
        n_geom_attn_qpts = 4, 
        n_geom_attn_vpts = 8, 
        c = 16
    ):
        super().__init__() # inherit module
        # set vars
        self.H, self.n_qk, self.n_v, self.c = n_heads, n_geom_attn_qpts, n_geom_attn_vpts, c

        # the mechanism of IPA (project representations and combine pairwise)
        # standard
        self.q_s = nn.Linear(d_single, n_heads*c, bias=False)
        self.k_s = nn.Linear(d_single, n_heads*c, bias=False)
        self.v_s = nn.Linear(d_single, n_heads*c, bias=False)
        # geometric
        self.q_pts = nn.Linear(d_single, n_heads * n_geom_attn_qpts * 3, bias=False)
        self.k_pts = nn.Linear(d_single, n_heads * n_geom_attn_qpts * 3, bias=False)
        self.v_pts = nn.Linear(d_single, n_heads * n_geom_attn_vpts * 3, bias=False)

        # the pair bias
        self.b_ij = nn.Linear(d_pair, n_heads, bias=False)
        # the head weights
        self.head_weight = nn.Parameter(torch.zeros(n_heads))

        # output dimension and projection back to single rep dim
        d_out = n_heads*c + n_heads * n_geom_attn_vpts * (3 + 1)
        self.out = nn.Linear(d_out, d_single)

    # global transformations (local to global); batched over [B, N, ...]
    # (batch subscript is 'g' to avoid colliding with the rotation axis 'b')
    def _global(self, pts, R, t):
        return torch.einsum("gnab,gnhpb->gnhpa", R, pts) + t[:, :, None, None, :]

    # forward pass, batched over B chains padded to N with pad_mask [B, N]
    def forward(self, s, z, R, t, pad_mask):
        B, N = s.shape[0], s.shape[1]
        H, nq, nv, c = self.H, self.n_qk, self.n_v, self.c

        # project seqeunce representations and reshape for multihead attn
        # standard attn
        q = self.q_s(s).view(B, N, H, c)
        k = self.k_s(s).view(B, N, H, c)
        v = self.v_s(s).view(B, N, H, c)
        # global attention
        q_p = self._global(self.q_pts(s).view(B, N, H, nq, 3), R, t)
        k_p = self._global(self.k_pts(s).view(B, N, H, nq, 3), R, t)
        v_p = self._global(self.v_pts(s).view(B, N, H, nv, 3), R, t)

        # add attention terms
        attn = torch.einsum("bihc,bjhc->bijh", q, k) / math.sqrt(c)
        diff = q_p[:, :, None] - k_p[:, None, :]
        dist = diff.pow(2).sum(-1).sum(-1) # squared distances
        attn = attn - 0.5 * F.softplus(self.head_weight)[None, None, None] * dist
        attn = attn + self.b_ij(z)
        # mask padded keys j so real residues never attend to padding
        attn = attn.masked_fill(~pad_mask[:, None, :, None], float("-inf"))
        attn = F.softmax(attn, dim=2) # normalize over the j residues

        # project the multihead scalar attention onto v and reshape back to dim
        o_scalar = torch.einsum("bijh,bjhc->bihc", attn, v).reshape(B, N, -1)

        # aggreagate the point attention and transform back to local frame of i
        v_agg = torch.einsum("bijh,bjhpd->bihpd", attn, v_p)
        v_local = torch.einsum("gnba,gnhpb->gnhpa", R, v_agg - t[:, :, None, None, :])
        # flatten of local points
        o_point = v_local.reshape(B, N, H*nv*3)
        # get norms of coords
        o_n = v_local.norm(dim=-1).reshape(B, N, H*nv)

        # return scalar attn features, local attention points, their magnitudes
        return self.out(torch.cat([o_scalar, o_point, o_n], dim=-1))
    
class IPABlock(nn.Module):
    '''
    A full IPA block that passes the protein through multihead IPA and a feed 
    forward nn to represent the residues with understanding of each other. The
    IPA backbone allows the residues to understand global distnace and 
    orientation, the standard attention allows the sequence representations to
    talk to one another, and the ffn with gelu allows the attention mechanism to 
    capture nonlinear relationships.

    Inputs:
        - s: sequence representations from embedding or previous layer
        - z: pair representation
        - R: local frame rotation
        - t: local frame translation

    Outputs:
        - s: updated features from attn and ffn
    '''
    def __init__(self, d_single, d_pair, dropout, **ipa_args):
        super().__init__() # inherit
        # pass through layernorms, ipa, then ff, applying dropout
        self.norm1 = nn.LayerNorm(d_single)
        self.ipa = IPA(d_single, d_pair, **ipa_args)
        self.norm2 = nn.LayerNorm(d_single)
        self.ff = nn.Sequential(
            nn.Linear(d_single, d_single*4),
            nn.GELU(),
            nn.Linear(d_single*4, d_single)
        )
        self.dropout = nn.Dropout(dropout)
        # AF-style triangle multiplicative update on the pair rep
        self.triangle = TriangleUpdate(d_pair)

    # a pass through the full IPA with dropout
    def forward(self, s, z, R, t, pad_mask):
        s = s + self.dropout(self.ipa(self.norm1(s), z, R, t, pad_mask))
        s = s + self.dropout(self.ff(self.norm2(s)))
        # update the pair representation with one outgoing triangle multiply
        z = z + self.dropout(self.triangle(z, pad_mask))
        return s, z


class TriangleUpdate(nn.Module):
    '''
    AlphaFold-style triangle multiplicative update (outgoing direction). For
    each pair (i,j), aggregates information from all (i,k) and (j,k) edges
    via an outer contraction over k:

        z_ij ← Out( Sum_k a_ik * b_jk )

    This injects structural triangle-inequality priors into the pair track
    so it can carry geometric relations between residues, rather than being
    a static bias.

    Inputs:
        - z: [B, N, N, d_pair] pair representation
        - pad_mask: [B, N] bool, True for real residues

    Outputs:
        - delta: [B, N, N, d_pair] update to add residually to z
    '''
    def __init__(self, d_pair):
        super().__init__() # inherit
        # layernorm the pair input before projecting
        self.norm = nn.LayerNorm(d_pair)
        # the two triangle edge projections (a for i-k, b for j-k)
        self.a = nn.Linear(d_pair, d_pair)
        self.b = nn.Linear(d_pair, d_pair)
        # output projection after the triangle contraction
        self.out = nn.Linear(d_pair, d_pair)

    def forward(self, z, pad_mask):
        z_n = self.norm(z)
        a, b = self.a(z_n), self.b(z_n)             # [B, N, N, d_pair]
        # zero the contracted index k where it is padding, so the sum over k
        # only accumulates real edges
        km = pad_mask[:, None, :, None].to(a.dtype)  # [B, 1, N, 1] over the k axis
        a, b = a * km, b * km
        # outgoing triangle: sum over k of a_ik * b_jk -> [B, N, N, d_pair]
        delta = self.out(torch.einsum("bikd,bjkd->bijd", a, b))
        # keep padded pairs at zero so the residual add never revives padding
        pair_pad = (pad_mask[:, :, None] & pad_mask[:, None, :]).unsqueeze(-1)
        return delta * pair_pad