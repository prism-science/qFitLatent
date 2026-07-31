# created by clay
''' 
The model to learn qfit multiconformer dynamics with an
IPA backbone
'''

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _checkpoint
import torch.nn.functional as F
import math
from .data.data import N_CHI, N_CHI_PER_AA

# global parameters
# minimum von mises concentration to keep kappa > 0 and i0e well behaved
KAPPA_MIN = 1e-2

class qFitLatent(nn.Module):
    ''' 
    SE(3) invariant model for learning protein dynamics
    from qfit multiconformer models
    
    Inputs:
        - aa_tokens: (N)
        - R: (N, 3, 3)
        - t: (N, 3)
        - chi_obs (optional): (N, A, d_chi) observed altloc chi for teacher
          forcing the autoregressive head during training; if None we run
          inference-mode beam search

    Outputs (autoregressive head — see ChiARHead):
        training (chi_obs given):  pi/mu/kappa (N, A, d_chi, k)
        inference (chi_obs=None):  pi (N, K), mu/kappa (N, K, d_chi)
    '''

    # initialize with hyperparams
    def __init__(
        self,
        n_aa = 21,
        d_single = 128,
        d_pair = 32,
        n_ipa = 8,
        n_heads = 8,
        n_geom_attn_qpts = 4,
        n_geom_attn_vpts = 8,
        c = 8,
        k = 5,
        dropout = 0.0,
    ):
        super().__init__() # inherit nn 
        # set representation sizes
        self.d_s = d_single
        self.d_z = d_pair

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

        # autoregressive chi-angle vM mixture head (per-chi conditional on chi_<d)
        self.chi_head = ChiARHead(
            d_single,
            k=k,
            d_chi=N_CHI,
        )

    # initialize the pair representation track
    def pair_init(self, s, t):
        # number of residues
        N = s.shape[0]
        # sequential indexes
        idx = torch.arange(N, device=t.device)

        # the relative position distances (clamped to +-64)
        rel_pos = (idx[:, None] - idx[None, :]).clamp(-64, 64) + 64
        # embed the relative positions
        z = self.rel_pos_embed(rel_pos) # [N, N, 129] to [N, N, d_pair]
        # get the pairwise distances of CA's (L2 Norm) keep dim for proj
        dist = (t[:, None] - t[None, :]).norm(dim=-1, keepdim=True)
        # add the distance embeddings to the rel pos embeddings
        z = z + self.dist_embed(dist / 10.0) # angstroms to nm
        # project the single representations
        env = self.env_proj(s) # [N, d_single] to [N, d_pair]
        # add the pair projection products to the pair rep
        z = z + self.env_pair_proj(env[:, None, :]*env[None, :, :])
        return z # return the pairwise representation [N, N, d_pair]
    
    def chi_mask(self, aa_tokens):
        counts = torch.tensor(N_CHI_PER_AA, device=aa_tokens.device)
        thresholds = torch.arange(N_CHI, device=aa_tokens.device)
        return thresholds[None, :] < counts[aa_tokens, None] # [N, N_CHI] bool
    
    # a forward pass. chi_obs is required for training (teacher-forced AR);
    # at inference (chi_obs=None) we beam-search ancestrally for the top K
    # joint modes — same return-shape convention as the old head.
    def forward(self, aa_tokens, R, t, chi_obs=None, K=8):
        # embed sequence tokens and prepare pair track
        s = self.aa_embed(aa_tokens)
        z = self.pair_init(s, t)

        # iterate through blocks and pass info; pair track is now updated
        # in-place by each block via the triangle multiplication
        for block in self.ipa_blocks:
            if self.training: # load from checkpoint and resume
                s, z = _checkpoint(block, s, z, R, t, use_reentrant=False)
            else: # forward pass
                s, z = block(s, z, R, t)

        chi_mask = self.chi_mask(aa_tokens)
        if chi_obs is not None:
            # training: teacher-forced AR -> pi/mu/kappa [N, A, D, k]
            return self.chi_head(s, chi_obs, chi_mask)
        # inference: beam search -> pi [N,K], mu/kappa [N, K, D]
        return self.chi_head.predict(s, chi_mask, K=K)


class ConfidenceModel(qFitLatent):
    '''
    Predicts per-residue confidence for a qFitLatent checkpoint: how far (in
    A, Wasserstein distance) its predicted chi mixture is likely to land from
    the true qFit conformer ensemble -- trained separately, in addition to
    the base model, against W1 labels computed from a frozen checkpoint's own
    predictions (scripts/make_w1_labels.py). Same backbone + sequence input
    and IPA trunk as qFitLatent (AlphaFold-pLDDT style confidence head, just
    without a chi mixture to condition on).
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.chi_head  # confidence head replaces the chi mixture head
        self.conf_head = ConfidenceHead(self.d_s)

    # aa_tokens: (N); R: (N,3,3); t: (N,3) -> pred_log_w1: (N,)
    def forward(self, aa_tokens, R, t):
        s = self.aa_embed(aa_tokens)
        z = self.pair_init(s, t)
        for block in self.ipa_blocks:
            if self.training:
                s, z = _checkpoint(block, s, z, R, t, use_reentrant=False)
            else:
                s, z = block(s, z, R, t)
        return self.conf_head(s)


class ConfidenceHead(nn.Module):
    ''' Small MLP predicting a scalar per-residue target from the single rep s. '''
    def __init__(self, d_single, d_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_single, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)  # (N,)


class ChiARHead(nn.Module):
    '''
    Autoregressive von Mises mixture over the four sidechain chi angles.
    Each chi dim d has its own k-component conditional vM mixture whose
    parameters depend on the IPA hidden h AND the previous chi values
    chi_<d via sin/cos features:

        p(𝝌 | h) = ∏_d Σ_k π_dk(h, χ_<d) · vM(χ_d | μ_dk, κ_dk)

    A single rotameric mode is now a *sequence* of component choices
    (k_1, k_2, k_3, k_4) — so one mode can natively express correlated
    chi (e.g. ARG g+/g+/t/g+) instead of the factorised mixture's
    independent-per-dim within-component restriction. 

    Training (teacher forcing) — forward(h, chi_obs):
        conditioning at chi_d uses the OBSERVED altloc chi_<d, so the
        returned params are shape [N, A, D, k] per altloc per chi per
        component. Loss sums log-likelihood over D autoregressively.
    Inference — predict(h, K):
        ancestral beam search; returns top-K joint modes [N, K, D] in the
        same shape as the old head, so the multiconformer writer keeps
        working.
    '''
    def __init__(self, d_single, k = 5, d_chi = N_CHI, d_cond = 64):
        super().__init__()
        self.k = k
        self.d_chi = d_chi
        # one small MLP per chi dim; input is h plus sin/cos of all previous
        # chi (2 features each) -> output 3*k for (pi, mu, kappa) each k-dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_single + 2*d, d_cond),
                nn.SiLU(),
                nn.Linear(d_cond, 3*k),
            )
            for d in range(d_chi)
        ])

    # split a [..., 3k] head output into vM mixture params with the right
    # activations: softmax(pi), pi*tanh(mu), softplus(kappa)+KAPPA_MIN
    def _split(self, out):
        pi_logits, mu_raw, ka_raw = out.chunk(3, dim=-1)
        pi    = torch.softmax(pi_logits.float(), dim=-1)
        mu    = math.pi * torch.tanh(mu_raw.float())
        kappa = F.softplus(ka_raw.float()) + KAPPA_MIN
        return pi, mu, kappa

    # training: teacher-forced autoregressive pass. chi_obs supplies the
    # conditioning per altloc; output broadcasts over the altloc axis A.
    def forward(self, h, chi_obs, chi_mask=None):
        # h: [N, d_single]; chi_obs: [N, A, D]; chi_mask: [N, D]
        N, A, D = chi_obs.shape
        sin_chi = torch.sin(chi_obs.float())        # [N, A, D]
        cos_chi = torch.cos(chi_obs.float())        # [N, A, D]
        h_exp = h.unsqueeze(1).expand(N, A, h.shape[-1])  # [N, A, d_single]

        pis, mus, kappas = [], [], []
        for d in range(D):
            if d == 0:
                cond = h_exp                                    # [N, A, d_s]
            else:
                # sin/cos of chi_0..d-1 concatenated  [N, A, 2d]
                prev = torch.cat([sin_chi[:, :, :d],
                                  cos_chi[:, :, :d]], dim=-1)
                cond = torch.cat([h_exp, prev], dim=-1)         # [N, A, d_s+2d]
            pi_d, mu_d, ka_d = self._split(self.heads[d](cond)) # [N, A, k] each
            pis.append(pi_d); mus.append(mu_d); kappas.append(ka_d)

        pi    = torch.stack(pis,    dim=2)          # [N, A, D, k]
        mu    = torch.stack(mus,    dim=2)
        kappa = torch.stack(kappas, dim=2)

        # hide chi dims that dont exist in this residue
        if chi_mask is not None:
            m = chi_mask[:, None, :, None].to(torch.bool)       # [N, 1, D, 1]
            mu    = mu * m.to(mu.dtype)
            kappa = torch.where(m, kappa, torch.ones_like(kappa))

        return pi, mu, kappa

    # inference: ancestral beam search through the AR chain. returns top-K
    # joint modes in the same shape as the old head so downstream code (the
    # multiconformer pdb writer, w1 eval) works unchanged.
    @torch.no_grad()
    def predict(self, h, chi_mask=None, K=8):
        # h: [N, d_single]; returns pi [N, K], mu/kappa [N, K, D]
        N = h.shape[0]
        k = self.k
        D = self.d_chi

        # step 0: k candidate modes, no conditioning
        pi0, mu0, ka0 = self._split(self.heads[0](h))           # [N, k]
        cand_mu     = mu0.unsqueeze(-1)                         # [N, k, 1]
        cand_kappa  = ka0.unsqueeze(-1)                         # [N, k, 1]
        cand_log_pi = pi0.clamp(min=1e-8).log()                 # [N, k]

        for d in range(1, D):
            C = cand_mu.shape[1]
            # condition on each candidate's chi so far (use the mode mu)
            prev = torch.cat([torch.sin(cand_mu),
                              torch.cos(cand_mu)], dim=-1)      # [N, C, 2d]
            h_exp = h.unsqueeze(1).expand(N, C, h.shape[-1])
            cond  = torch.cat([h_exp, prev], dim=-1)            # [N, C, d_s+2d]
            pi_d, mu_d, ka_d = self._split(self.heads[d](cond)) # [N, C, k]

            # expand each candidate by k children; joint log_pi accumulates
            new_log_pi = (cand_log_pi.unsqueeze(-1)
                          + pi_d.clamp(min=1e-8).log())         # [N, C, k]
            new_log_pi = new_log_pi.reshape(N, C * k)
            new_mu = torch.cat([
                cand_mu.unsqueeze(2).expand(N, C, k, d),
                mu_d.unsqueeze(-1),
            ], dim=-1).reshape(N, C * k, d + 1)
            new_kappa = torch.cat([
                cand_kappa.unsqueeze(2).expand(N, C, k, d),
                ka_d.unsqueeze(-1),
            ], dim=-1).reshape(N, C * k, d + 1)

            # beam prune: keep top K joint modes by log_pi
            keep = min(K, new_log_pi.shape[1])
            top_log_pi, top_idx = new_log_pi.topk(keep, dim=-1)
            idx_d = top_idx.unsqueeze(-1).expand(-1, -1, new_mu.shape[-1])
            cand_mu     = new_mu.gather(1, idx_d)
            cand_kappa  = new_kappa.gather(1, idx_d)
            cand_log_pi = top_log_pi

        # renormalise joint probabilities across the kept K modes
        pi = torch.softmax(cand_log_pi, dim=-1)                 # [N, K]
        if chi_mask is not None:
            m = chi_mask[:, None, :].to(torch.bool)             # [N, 1, D]
            cand_mu    = cand_mu * m.to(cand_mu.dtype)
            cand_kappa = torch.where(m.expand_as(cand_kappa),
                                     cand_kappa,
                                     torch.ones_like(cand_kappa))
        return pi, cand_mu, cand_kappa
    
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

    # global transformations (local to global)
    def _global(self, pts, R, t):
        return torch.einsum("nab,nhpb->nhpa", R, pts) + t[:, None, None, :]
    
    # forward pass
    def forward(self, s, z, R, t):
        N, H, nq, nv, c = s.shape[0], self.H, self.n_qk, self.n_v, self.c

        # project seqeunce representations and reshape for multihead attn
        # standard attn
        q = self.q_s(s).view(N, H, c)
        k = self.k_s(s).view(N, H, c)
        v = self.v_s(s).view(N, H, c)
        # global attention
        q_p = self._global(self.q_pts(s).view(N, H, nq, 3), R, t)
        k_p = self._global(self.k_pts(s).view(N, H, nq, 3), R, t)
        v_p = self._global(self.v_pts(s).view(N, H, nv, 3), R, t)

        # add attention terms
        attn = torch.einsum("ihc,jhc->ijh", q, k) / math.sqrt(c)
        diff = q_p[:, None] - k_p[None, :]
        dist = diff.pow(2).sum(-1).sum(-1) # squared distances
        attn = attn - 0.5 * F.softplus(self.head_weight)[None, None] * dist
        attn = F.softmax(attn + self.b_ij(z), dim=1) # normalize over the j residues

        # project the multihead scalar attention onto v and reshape back to dim
        o_scalar = torch.einsum("ijh,jhc->ihc", attn, v).reshape(N, -1)

        # aggreagate the point attention and transform back to local frame of i
        v_agg = torch.einsum("ijh,jhpd->ihpd", attn, v_p)
        v_local= torch.einsum("nba,nhpb->nhpa", R, v_agg - t[:, None, None, :])
        # flatten of local points
        o_point = v_local.reshape(N, H*nv*3)
        # get norms of coords
        o_n = v_local.norm(dim=-1).reshape(N, H*nv)

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
    def forward(self, s, z, R, t):
        s = s + self.dropout(self.ipa(self.norm1(s), z, R, t))
        s = s + self.dropout(self.ff(self.norm2(s)))
        # update the pair representation with one outgoing triangle multiply
        z = z + self.dropout(self.triangle(z))
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
        - z: [N, N, d_pair] pair representation

    Outputs:
        - delta: [N, N, d_pair] update to add residually to z
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

    def forward(self, z):
        z_n = self.norm(z)
        # outgoing triangle: sum over k of a_ik * b_jk -> [N, N, d_pair]
        return self.out(torch.einsum("ikd,jkd->ijd", self.a(z_n), self.b(z_n)))