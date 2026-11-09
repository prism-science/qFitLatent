# created by clay
'''
Local atomic environment of each residue: the side-chain atoms of its nearest
visible neighbours, seen from the residue's own backbone frame.

The ensemble encoder only gives each residue its own chi angles and occupancies,
so a residue learns about the side chains packed around it only indirectly, through
the IPA trunk's attention over those per-residue codes. This encoder hands the trunk
the neighbouring atoms themselves: for residue i, every side-chain heavy atom of
every altloc of its K nearest visible residues (by CA distance), expressed in i's
local frame, embedded with its chemistry (residue type x atom slot) and a radial
basis of its distance, and pooled with the altloc occupancy as weight. Hidden
residues (masked or unresolved) contribute no atoms, and a residue never sees its
own atoms, so the masked target cannot leak into its own context.
'''

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .sidechain import MAXATOM, tables

N_BACKBONE = 4      # N, CA, C, O lead every residue's atom order; frames cover them


class NeighbourAtomEncoder(nn.Module):
    '''
    Occupancy-weighted point-set encoder over neighbouring side-chain atoms.

    Inputs: R [B,N,3,3], t [B,N,3] backbone frames; sc_atoms [B,N,A,MAXATOM,3]
    altloc heavy atoms (global frame); occupancies [B,N,A]; aa_tokens [B,N];
    visible [B,N] residues whose atoms may be shown. Output: [B,N,d_single].

    The output projection is zero initialized, so adding this encoder to a trained
    model leaves its predictions unchanged until training uses the new signal.
    '''
    def __init__(self, d_single, n_neighbours=12, d_atom=32, n_rbf=16, max_dist=12.0):
        super().__init__()
        self.k, self.max_dist = n_neighbours, max_dist
        self.atom_type = nn.Embedding(21 * MAXATOM, d_atom)    # residue type x slot
        self.geom = nn.Linear(3 + n_rbf, d_atom)
        self.atom_mlp = nn.Sequential(nn.GELU(), nn.Linear(d_atom, d_atom), nn.GELU(),
                                      nn.Linear(d_atom, d_atom))
        self.out = nn.Linear(d_atom, d_single)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("rbf_centers", torch.linspace(0.0, max_dist, n_rbf))
        self.rbf_width = max_dist / n_rbf
        # side-chain slots per residue type (UNK gets none)
        sc_slots = torch.cat([tables()[2], torch.zeros(1, MAXATOM, dtype=torch.bool)])
        self.register_buffer("sc_slots", sc_slots[:, N_BACKBONE:])

    def neighbours(self, t, visible):
        '''Indices [B,N,K] of each residue's K nearest visible residues and a mask of
        the real ones (fewer than K visible residues leaves padded slots).'''
        B, N = visible.shape
        k = min(self.k, N - 1)
        dist = torch.cdist(t, t)                                    # [B,N,N]
        hidden = ~visible[:, None, :] | torch.eye(N, dtype=torch.bool, device=t.device)
        dist = dist.masked_fill(hidden, float("inf"))
        d_k, idx = dist.topk(k, dim=-1, largest=False)
        return idx, torch.isfinite(d_k)

    def encode(self, R, t, atoms, occ, aa_nb, idx, nb_ok):
        '''Pooled environment features [B,N,d_atom] from gathered neighbour atoms.'''
        # neighbour atoms in residue i's local frame: R_i^T (x - t_i)
        local = torch.einsum("bnji,bnkamj->bnkami", R,
                             atoms - t[:, :, None, None, None, :])  # [B,N,K,A,M,3]
        r = local.norm(dim=-1)
        rbf = torch.exp(-((r[..., None] - self.rbf_centers) / self.rbf_width) ** 2)
        slot = torch.arange(N_BACKBONE, MAXATOM, device=t.device)
        chem = self.atom_type(aa_nb[..., None] * MAXATOM + slot)    # [B,N,K,M,d]
        h = self.atom_mlp(self.geom(torch.cat([local / 10.0, rbf], -1))
                          + chem[:, :, :, None])                    # [B,N,K,A,M,d]
        present = (atoms != 0).any(-1) & self.sc_slots[aa_nb][:, :, :, None]
        weight = (occ[..., None] * present * (r < self.max_dist)
                  * nb_ok[..., None, None]).to(h.dtype)             # [B,N,K,A,M]
        pooled = (weight[..., None] * h).sum(dim=(2, 3, 4))         # [B,N,d]
        return pooled / nb_ok.sum(-1, keepdim=True).clamp_min(1).to(h.dtype)

    def forward(self, R, t, sc_atoms, occupancies, aa_tokens, visible):
        B, N = aa_tokens.shape
        if N < 2:
            return self.out.bias.expand(B, N, -1)
        idx, nb_ok = self.neighbours(t, visible)
        # only the altloc slots any residue in the batch uses, side-chain atoms only
        n_alt = max(int((occupancies > 0).sum(-1).max()), 1)
        flat = (torch.arange(B, device=t.device)[:, None, None] * N + idx).reshape(-1)
        gather = lambda x: x.reshape(B * N, *x.shape[2:])[flat].reshape(
            B, N, idx.shape[-1], *x.shape[2:])
        atoms = gather(sc_atoms[:, :, :n_alt, N_BACKBONE:])         # [B,N,K,A,M,3]
        occ = gather(occupancies[:, :, :n_alt])                     # [B,N,K,A]
        aa_nb = gather(aa_tokens.clamp(max=20))                     # [B,N,K]
        args = (R, t, atoms, occ, aa_nb, idx, nb_ok)
        # the per-atom features are large; recompute them in backward instead of storing
        pooled = (checkpoint(self.encode, *args, use_reentrant=False)
                  if self.training else self.encode(*args))
        return self.out(pooled)
