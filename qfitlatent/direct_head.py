# created by clay
'''
Direct chi readout, the model's decoder: predicts each residue's conformer mixture
(weights, chi means, chi spreads) with no Dunbrack rotamer library.

It replaced the library shift head (archive/rotamer_shift_heads.py), which reweighted
the library's rotamer slots and moved their means by at most +/-45 deg, so its
components stayed near the library wells. This head places K free components anywhere
on the chi torus, so ensembles can sit between the wells. The library lookup handed
the shift head the residue's amino acid and backbone (phi, psi) outside the latent
bottleneck; this head gets those two inputs directly.
'''

import math

import torch
import torch.nn as nn

from .featurize import N_CHI


def torus_anchors(n_comp, n_chi):
    '''
    Quasi-uniform starting angles [n_comp, n_chi] on the chi torus (Kronecker
    sequence), so the untrained components cover every region instead of piling up
    and leaving distant altlocs with no component near them (the dead-component
    failure of the archived library-anchored conformer head).
    '''
    steps = torch.tensor([math.sqrt(p) for p in (2, 3, 5, 7, 11, 13)][:n_chi])
    k = torch.arange(n_comp, dtype=torch.float32)[:, None] + 0.5
    return 2 * math.pi * torch.frac(k * steps[None, :])


class DirectChiHead(nn.Module):
    '''
    K free von Mises conformers per residue, predicted directly:

        mu_kc    = atan2(y_kc, x_kc)                 (full circle, no library mean)
        sigma_kc = exp(lo + (hi - lo) * sigmoid(s_kc))   in [min_deg, max_deg]
        log w_k  = log_softmax(w_k)

    from [h, aa embedding, sin/cos(phi, psi)]. (x, y) start on the unit circle at
    torus_anchors, the spreads at init_deg and the weights uniform. Every loss and
    evaluation reads the mixture over its component axis (log_p [., K], mu / sigma
    [., K, N_CHI]).

    Inputs:
        - h: [..., d_single] decoder features (from the latent)
        - aa_tokens: [...] long
        - phi, psi: [...] radians; bb_mask: [..., 2] bool (angle defined)

    Outputs:
        - log_p: [..., K], mu: [..., K, N_CHI], sigma: [..., K, N_CHI]
    '''
    def __init__(self, d_single, n_comp=16, n_aa=21, n_chi=N_CHI,
                 min_deg=2.0, max_deg=90.0, init_deg=30.0):
        super().__init__() # inherit module
        self.K, self.C = n_comp, n_chi
        self.aa_embed = nn.Embedding(n_aa, d_single)
        self.bb_proj = nn.Linear(6, d_single)
        self.net = nn.Sequential(
            nn.LayerNorm(d_single), nn.Linear(d_single, d_single), nn.GELU(),
            nn.Linear(d_single, d_single), nn.GELU(),
        )
        self.weight_logits = nn.Linear(d_single, n_comp)
        self.chi_xy = nn.Linear(d_single, n_comp * n_chi * 2)
        self.spread = nn.Linear(d_single, n_comp * n_chi)

        # uniform weights, anchored means, broad spreads at init
        nn.init.zeros_(self.weight_logits.weight)
        nn.init.zeros_(self.weight_logits.bias)
        anchors = torus_anchors(n_comp, n_chi)                     # [K, C]
        with torch.no_grad():
            self.chi_xy.weight.mul_(0.1)
            self.chi_xy.bias.copy_(
                torch.stack([anchors.cos(), anchors.sin()], -1).flatten())
        self.lo, self.hi = math.log(math.radians(min_deg)), math.log(math.radians(max_deg))
        frac = (math.log(math.radians(init_deg)) - self.lo) / (self.hi - self.lo)
        nn.init.zeros_(self.spread.weight)
        nn.init.constant_(self.spread.bias, math.log(frac / (1 - frac)))

    def forward(self, h, aa_tokens, phi, psi, bb_mask):
        m = bb_mask.to(h.dtype)
        bb = torch.stack([phi.sin() * m[..., 0], phi.cos() * m[..., 0],
                          psi.sin() * m[..., 1], psi.cos() * m[..., 1],
                          m[..., 0], m[..., 1]], dim=-1)           # [..., 6]
        x = self.net(h + self.aa_embed(aa_tokens) + self.bb_proj(bb))

        shape = (*h.shape[:-1], self.K, self.C)
        xy = self.chi_xy(x).view(*shape, 2)
        mu = torch.atan2(xy[..., 1], xy[..., 0])
        log_sigma = self.lo + (self.hi - self.lo) * torch.sigmoid(self.spread(x).view(shape))
        log_p = torch.log_softmax(self.weight_logits(x).float(), dim=-1)
        return log_p, mu, log_sigma.exp()
