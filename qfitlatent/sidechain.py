# created by clay
'''
Differentiable side-chain kinematics for the atom-space (Angstrom) reconstruction
loss.

Each residue's side chain is a tree rooted at the backbone (CB's parent is CA, and
so on). Rotating chi_c turns every atom in the subtree below the chi_c bond about
that bond. Given a residue's OBSERVED heavy-atom template and its observed chi, we
can therefore place the atoms at ANY target chi by rotating the moving subset of
each chi in turn - a fully differentiable function of the target chi (torch), so a
matched atom-RMSD loss can backprop into the predicted chi.

Using the residue's own observed atoms as the template means we never need ideal
bond lengths/angles: the geometry is exactly the crystallographic one, and chi=chi
reproduces it exactly. The only static per-residue-type data is the bond tree
(atom -> parent), from which the atom order, the per-chi moving-atom mask and the
chi rotation-axis atoms are derived.
'''

import torch

from .featurize import N_CHI, CHI_ATOMS, _AA

BACKBONE = ["N", "CA", "C", "O"]

# side-chain heavy-atom bond tree per residue type: atom -> parent (parent may be a
# backbone atom). Ring-closing bonds are omitted; a ring's atoms are all descendants
# of the ring's entry atom, so they still rotate together as one rigid subtree.
SC_TREE = {
    "ARG": {"CB": "CA", "CG": "CB", "CD": "CG", "NE": "CD", "CZ": "NE",
            "NH1": "CZ", "NH2": "CZ"},
    "ASN": {"CB": "CA", "CG": "CB", "OD1": "CG", "ND2": "CG"},
    "ASP": {"CB": "CA", "CG": "CB", "OD1": "CG", "OD2": "CG"},
    "CYS": {"CB": "CA", "SG": "CB"},
    "GLN": {"CB": "CA", "CG": "CB", "CD": "CG", "OE1": "CD", "NE2": "CD"},
    "GLU": {"CB": "CA", "CG": "CB", "CD": "CG", "OE1": "CD", "OE2": "CD"},
    "HIS": {"CB": "CA", "CG": "CB", "ND1": "CG", "CD2": "CG", "CE1": "ND1",
            "NE2": "CD2"},
    "ILE": {"CB": "CA", "CG1": "CB", "CG2": "CB", "CD1": "CG1"},
    "LEU": {"CB": "CA", "CG": "CB", "CD1": "CG", "CD2": "CG"},
    "LYS": {"CB": "CA", "CG": "CB", "CD": "CG", "CE": "CD", "NZ": "CE"},
    "MET": {"CB": "CA", "CG": "CB", "SD": "CG", "CE": "SD"},
    "PHE": {"CB": "CA", "CG": "CB", "CD1": "CG", "CD2": "CG", "CE1": "CD1",
            "CE2": "CD2", "CZ": "CE1"},
    "PRO": {"CB": "CA", "CG": "CB", "CD": "CG"},
    "SER": {"CB": "CA", "OG": "CB"},
    "THR": {"CB": "CA", "OG1": "CB", "CG2": "CB"},
    "TRP": {"CB": "CA", "CG": "CB", "CD1": "CG", "CD2": "CG", "NE1": "CD1",
            "CE2": "NE1", "CE3": "CD2", "CZ2": "CE2", "CZ3": "CE3", "CH2": "CZ2"},
    "TYR": {"CB": "CA", "CG": "CB", "CD1": "CG", "CD2": "CG", "CE1": "CD1",
            "CE2": "CD2", "CZ": "CE1", "OH": "CZ"},
    "VAL": {"CB": "CA", "CG1": "CB", "CG2": "CB"},
}

# atom order per residue: backbone then side-chain atoms with parents before children
def _atom_order(resname):
    order = list(BACKBONE)
    tree = SC_TREE.get(resname, {})
    remaining = list(tree)
    while remaining:                       # topological: emit an atom once its parent is placed
        progressed = False
        for a in list(remaining):
            if tree[a] in order:
                order.append(a); remaining.remove(a); progressed = True
        if not progressed:                 # dangling parent (shouldn't happen) - emit as-is
            order += remaining; break
    return order


ATOM_ORDER = {a: _atom_order(a) for a in _AA if a in SC_TREE}
MAXATOM = max((len(v) for v in ATOM_ORDER.values()), default=0)
N_AA = len(_AA)


def _descendants(tree, root):
    '''All atoms in the subtree rooted at `root` (inclusive).'''
    out, stack = set(), [root]
    while stack:
        x = stack.pop()
        out.add(x)
        stack += [a for a, p in tree.items() if p == x]
    return out


# static tables: per-aa atom count, per-chi moving-atom mask, per-chi axis atom indices
_atom_count = torch.zeros(N_AA, dtype=torch.long)
_moving = torch.zeros(N_AA, MAXATOM, N_CHI, dtype=torch.bool)
_axis = torch.zeros(N_AA, N_CHI, 2, dtype=torch.long)     # (from_idx, to_idx) into atom order
_sc_mask = torch.zeros(N_AA, MAXATOM, dtype=torch.bool)   # atom slot is a real side-chain heavy atom

for ai, resname in enumerate(_AA):
    order = ATOM_ORDER.get(resname)
    if not order:
        continue
    idx = {name: i for i, name in enumerate(order)}
    _atom_count[ai] = len(order)
    _sc_mask[ai, len(BACKBONE):len(order)] = True          # exclude backbone from RMSD
    tree = SC_TREE[resname]
    for c, quad in enumerate(CHI_ATOMS.get(resname, [])):
        b, cc = quad[1], quad[2]                            # rotation bond b--cc
        _axis[ai, c] = torch.tensor([idx[b], idx[cc]])
        for name in _descendants(tree, cc):                # subtree below the bond rotates
            _moving[ai, idx[name], c] = True


def tables(device=None):
    '''Move the static geometry tables to a device (cached on the tensors themselves).'''
    return (_moving.to(device), _axis.to(device), _sc_mask.to(device),
            _atom_count.to(device))


def rebuild(atoms, chi_ref, chi_tgt, aa_tokens):
    '''
    Place a residue's side-chain atoms at target chi, given its observed template.

    Rotates, for each chi in turn, the atoms in that chi's moving subtree about the
    (current) chi bond by (chi_tgt - chi_ref). Differentiable in chi_tgt.

    Inputs (batched over leading dims ...):
        atoms:     [..., MAXATOM, 3] observed template coordinates (Angstrom)
        chi_ref:   [..., N_CHI] the template's own chi (radians)
        chi_tgt:   [..., N_CHI] target chi to place the atoms at (radians)
        aa_tokens: [...] long, amino-acid index (selects the geometry tables)

    Output:
        [..., MAXATOM, 3] atoms placed at chi_tgt (backbone/invalid slots unchanged)
    '''
    moving, axis, _, _ = tables(atoms.device)
    mv = moving[aa_tokens]                 # [..., MAXATOM, N_CHI]
    ax = axis[aa_tokens]                   # [..., N_CHI, 2]
    x = atoms
    for c in range(N_CHI):
        i_from = ax[..., c, 0]             # [...]
        i_to = ax[..., c, 1]
        b = torch.gather(x, -2, i_from[..., None, None].expand(*i_from.shape, 1, 3))[..., 0, :]
        cc = torch.gather(x, -2, i_to[..., None, None].expand(*i_to.shape, 1, 3))[..., 0, :]
        u = cc - b
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)          # rotation axis [...,3]
        delta = chi_tgt[..., c] - chi_ref[..., c]              # [...]
        cos = torch.cos(delta)[..., None, None]
        sin = torch.sin(delta)[..., None, None]
        p = x - cc[..., None, :]                               # about the bond atom cc
        uu = u[..., None, :]                                   # [...,1,3]
        dot = (p * uu).sum(-1, keepdim=True)
        rot = p * cos + torch.cross(uu.expand_as(p), p, dim=-1) * sin + uu * dot * (1 - cos)
        rotated = cc[..., None, :] + rot
        m = mv[..., c][..., None]                              # [...,MAXATOM,1]
        x = torch.where(m, rotated, x)
    return x
