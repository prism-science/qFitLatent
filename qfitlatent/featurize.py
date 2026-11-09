# created by clay
'''
Per-chain featurization for qFit multiconformer models, built on atomworks.

qFit PDBs carry no canonical sequence, so gaps (unmodeled residues) are recovered
by aligning the structure-observed sequence to the RCSB sequence fetched by
scripts/fetch_sequences.py. Every canonical position gets an amino-acid token;
resolved positions also get an AF-style rigid frame (R, t) and qFit chi
angles/occupancies per altloc. Unresolved positions get identity R, zero t,
resolved_mask = False.
'''

import argparse
from pathlib import Path

import numpy as np
import torch
import biotite.structure as struc
from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal
from atomworks.io.utils.io_utils import load_any
from atomworks.ml.utils.geometry import rigid_from_3_points

DEFAULT_SEQ_DIR = Path(__file__).resolve().parent.parent / "data" / "seqeuences"
N_CHI, MAX_ALTLOCS = 4, 6

# --- concepts: amino-acid tokens and IUPAC chi-angle atom quadruples ----------
_AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
_ONE = "ARNDCQEGHILKMFPSTWYV"
THREE_TO_ONE = {**{t: o for t, o in zip(_AA, _ONE)}, "MSE": "M"}
ONE_TO_THREE = {o: t for t, o in zip(_AA, _ONE)}
ONE_TO_IDX = {o: i for i, o in enumerate(_ONE)}
UNK = 20

CHI_ATOMS = {
    "ARG": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")],
    "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "LYS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "PRO": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "VAL": [("N", "CA", "CB", "CG1")],
}
_PROT_ALPHABET = set(ProteinSequence.alphabet)
_SUB = SubstitutionMatrix.std_protein_matrix()


def load_canonical(pdbid, chain, seq_dir=DEFAULT_SEQ_DIR):
    '''Read the fetched canonical sequence for one chain, or None if absent.'''
    path = Path(seq_dir) / f"{pdbid}_{chain}.fasta"
    if not path.exists():
        return None
    return "".join(l.strip() for l in path.read_text().splitlines()
                   if not l.startswith(">"))


def _align(canonical, observed):
    '''Yield (canonical_idx, observed_idx | None) for each canonical position.'''
    san = lambda s: "".join(c if c in _PROT_ALPHABET else "X" for c in s)
    aln = align_optimal(ProteinSequence(san(canonical)), ProteinSequence(san(observed)),
                        _SUB, gap_penalty=(-10, -1), terminal_penalty=False)[0]
    for ci, oi in aln.trace:
        if ci != -1:
            yield int(ci), (None if oi == -1 else int(oi))


def _coord(res, name, alt):
    '''Coordinate of an atom in a given altloc, falling back to the shared (" ").'''
    for a in (alt, " "):
        m = (res.atom_name == name) & (res.altloc_id == a)
        if m.any():
            return res.coord[m][0]
    return None


def _chi_and_occ(res):
    '''
    qFit chi angles + normalized occupancies + per-altloc heavy-atom coordinates for
    one residue. Atoms are laid out in the residue type's canonical order (backbone
    then side chain, see sidechain.ATOM_ORDER) so the atom-space loss can rebuild and
    RMSD them; atom_mask marks the real side-chain heavy atoms present.
    '''
    from .sidechain import ATOM_ORDER, MAXATOM   # lazy: sidechain imports this module

    chi = np.zeros((MAX_ALTLOCS, N_CHI), np.float32)
    occ = np.zeros(MAX_ALTLOCS, np.float32)
    atoms = np.zeros((MAX_ALTLOCS, MAXATOM, 3), np.float32)
    atom_mask = np.zeros(MAXATOM, bool)
    resname = res.res_name[0]
    defs = CHI_ATOMS.get(resname, [])
    if not defs:
        return chi, occ, atoms, atom_mask
    order = ATOM_ORDER.get(resname, [])

    alts = [a for a in sorted(set(res.altloc_id)) if a != " "] or [" "]
    records = []
    for alt in alts:
        angles, ok = np.zeros(N_CHI, np.float32), True
        for i, quad in enumerate(defs):
            pts = [_coord(res, n, alt) for n in quad]
            if any(p is None for p in pts):
                ok = False
                break
            angles[i] = struc.dihedral(*pts)
        if ok:
            xyz = np.zeros((MAXATOM, 3), np.float32)
            present = np.zeros(MAXATOM, bool)
            for k, name in enumerate(order):
                p = _coord(res, name, alt)
                if p is not None:
                    xyz[k] = p
                    present[k] = True
            named = res.occupancy[res.altloc_id == alt]
            records.append((angles, float(named.max()) if len(named) else 1.0,
                            xyz, present))

    records.sort(key=lambda r: -r[1])
    records = records[:MAX_ALTLOCS]
    total = sum(o for _, o, _, _ in records) or 1.0
    for j, (angles, o, xyz, present) in enumerate(records):
        chi[j], occ[j], atoms[j] = angles, o / total, xyz
        if j == 0:
            atom_mask = present
    return chi, occ, atoms, atom_mask


def _n_chi(one_letter):
    return len(CHI_ATOMS.get(ONE_TO_THREE.get(one_letter, ""), []))


def _backbone_dihedrals(bb, L):
    '''
    phi/psi per canonical position from neighbouring backbone atoms.

    phi_i = C(i-1) - N(i) - CA(i) - C(i), psi_i = N(i) - CA(i) - C(i) - N(i+1).
    Neighbours are canonical indices, so a gap or a chain terminus simply leaves
    the angle undefined (bb_mask False, value 0.0) rather than bridging the break.
    '''
    phi = np.zeros(L, np.float32)
    psi = np.zeros(L, np.float32)
    bb_mask = np.zeros((L, 2), bool)

    for i, (n, ca, c) in bb.items():
        prev = bb.get(i - 1)
        if prev is not None:
            phi[i] = struc.dihedral(prev[2], n, ca, c)
            bb_mask[i, 0] = True
        nxt = bb.get(i + 1)
        if nxt is not None:
            psi[i] = struc.dihedral(n, ca, c, nxt[0])
            bb_mask[i, 1] = True

    return phi, psi, bb_mask


def featurize_chain(chain, canonical, return_context=False):
    '''
    Featurize one protein-chain AtomArray against its canonical sequence.

    With return_context=True also returns (residues, canon_to_obs): the per-residue
    AtomArrays in observed order and a {canonical_index: observed_index} map, so
    callers (e.g. validate.py) can trace a model prediction back to real atoms.
    '''
    residues = list(struc.residue_iter(chain))
    observed = "".join(THREE_TO_ONE.get(r.res_name[0], "X") for r in residues)
    if canonical is None:
        canonical = observed
        pairs = [(i, i) for i in range(len(observed))]
    else:
        pairs = list(_align(canonical, observed))

    from .sidechain import MAXATOM               # lazy: sidechain imports this module
    L = len(canonical)
    aa_tokens = np.array([ONE_TO_IDX.get(c, UNK) for c in canonical], np.int64)
    R = np.broadcast_to(np.eye(3, dtype=np.float32), (L, 3, 3)).copy()
    t = np.zeros((L, 3), np.float32)
    resolved = np.zeros(L, bool)
    chi = np.zeros((L, MAX_ALTLOCS, N_CHI), np.float32)
    occ = np.zeros((L, MAX_ALTLOCS), np.float32)
    sc_atoms = np.zeros((L, MAX_ALTLOCS, MAXATOM, 3), np.float32)
    sc_atom_mask = np.zeros((L, MAXATOM), bool)
    chi_mask = np.zeros((L, N_CHI), bool)
    for i, c in enumerate(canonical):
        chi_mask[i, :_n_chi(c)] = True

    # backbone atoms kept per canonical index so phi/psi can use neighbours
    bb = {}

    for ci, oi in pairs:
        if oi is None:
            continue
        res = residues[oi]
        n, ca, cc = (_coord(res, a, "A") for a in ("N", "CA", "C"))
        if ca is not None:
            t[ci] = ca
        if n is not None and ca is not None and cc is not None:
            rot, _ = rigid_from_3_points(*(torch.tensor(np.asarray(p)) for p in (n, ca, cc)))
            R[ci] = rot.numpy()
            bb[ci] = (n, ca, cc)
        chi[ci], occ[ci], sc_atoms[ci], sc_atom_mask[ci] = _chi_and_occ(res)
        resolved[ci] = True

    phi, psi, bb_mask = _backbone_dihedrals(bb, L)

    feats = {
        "aa_tokens": torch.from_numpy(aa_tokens),
        "R": torch.from_numpy(R),
        "t": torch.from_numpy(t),
        "resolved_mask": torch.from_numpy(resolved),
        "chi_angles": torch.from_numpy(chi),
        "occupancies": torch.from_numpy(occ),
        "sc_atoms": torch.from_numpy(sc_atoms),
        "sc_atom_mask": torch.from_numpy(sc_atom_mask),
        "chi_mask": torch.from_numpy(chi_mask),
        "phi": torch.from_numpy(phi),
        "psi": torch.from_numpy(psi),
        "bb_mask": torch.from_numpy(bb_mask),
    }
    if return_context:
        canon_to_obs = {ci: oi for ci, oi in pairs if oi is not None}
        return feats, residues, canon_to_obs
    return feats


def _protein(path):
    '''
    Load a PDB and return the amino-acid-only AtomArray (single model, all altlocs).

    include_bonds=False because we never use connectivity, and biotite's bond
    parser rejects files whose atom serial numbers are not strictly increasing -
    which is common in qFit multiconformer output once altlocs are renumbered.
    '''
    aa = load_any(path, file_type="pdb", extra_fields=["occupancy"], altloc="all",
                  model=1, include_bonds=False)
    return aa[struc.filter_amino_acids(aa)]


def list_protein_chains(path):
    '''Protein chain letters in a file (used to build the per-chain dataset index).'''
    prot = _protein(path)
    return list(dict.fromkeys(prot.chain_id.tolist()))


def featurize_pdb(path, seq_dir=DEFAULT_SEQ_DIR):
    '''Featurize every protein chain in a qFit PDB. Returns {chain: feature_dict}.'''
    path = Path(path)
    pdbid = path.stem.lower()
    prot = _protein(path)
    out = {}
    for chain_id in dict.fromkeys(prot.chain_id.tolist()):
        chain = prot[prot.chain_id == chain_id]
        out[chain_id] = featurize_chain(chain, load_canonical(pdbid, chain_id, seq_dir))
    return out


def featurize_pdb_ctx(path, chain_id, seq_dir=DEFAULT_SEQ_DIR):
    '''
    Featurize a single named chain and also return the structural context
    (residues, canonical->observed map) needed to rebuild atoms. Returns
    (features, residues, canon_to_obs).
    '''
    path = Path(path)
    prot = _protein(path)
    chain = prot[prot.chain_id == chain_id]
    if chain.array_length() == 0:
        raise ValueError(f"chain {chain_id!r} not found or has no amino acids")
    canonical = load_canonical(path.stem.lower(), chain_id, seq_dir)
    return featurize_chain(chain, canonical, return_context=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", "-f", required=True, help="Input qFit PDB file")
    parser.add_argument("--seq-dir", default=str(DEFAULT_SEQ_DIR))
    args = parser.parse_args()
    for chain, f in featurize_pdb(args.file, seq_dir=args.seq_dir).items():
        L, n = f["aa_tokens"].shape[0], int(f["resolved_mask"].sum())
        print(f"chain {chain}: L={L} resolved={n} gaps={L - n}")
        for name, tensor in f.items():
            print(f"    {name:14s} {tuple(tensor.shape)} {tensor.dtype}")
