# created by clay
'''
The script to parse pdb files, extract altloc information and create a 
dataset to pass to a model to learn qfit altloc dynamics
'''

import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torch
import csv

# some global variables
# amino acid tokens
AA3 = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}
# number of chi angles per amino acid
N_CHI_PER_AA = (
    0,  # 0  ALA
    4,  # 1  ARG
    2,  # 2  ASN
    2,  # 3  ASP
    1,  # 4  CYS
    3,  # 5  GLN
    3,  # 6  GLU
    0,  # 7  GLY
    2,  # 8  HIS
    2,  # 9  ILE
    2,  # 10 LEU
    4,  # 11 LYS
    3,  # 12 MET
    2,  # 13 PHE
    2,  # 14 PRO
    1,  # 15 SER
    1,  # 16 THR
    2,  # 17 TRP
    2,  # 18 TYR
    1,  # 19 VAL
    0,  # 20 UNK
)
# total number of chi angles
N_CHI = 4
# Inverse map: AA index → 3-letter code (column-name ordering for CSV).
IDX_TO_AA3 = {v: k for k, v in AA3.items()}
AA_NAMES = [IDX_TO_AA3[i] for i in range(20)]
N_AA = 20
# IUPAC chi-angle atom quadruples per residue type (chi1 … chi4).
# Each tuple is (a1, a2, a3, a4) defining the dihedral a1-a2-a3-a4.
CHI_ATOMS: dict[str, list[tuple[str, str, str, str]]] = {
    "ALA": [],
    "ARG": [("N","CA","CB","CG"), ("CA","CB","CG","CD"),
            ("CB","CG","CD","NE"), ("CG","CD","NE","CZ")],
    "ASN": [("N","CA","CB","CG"), ("CA","CB","CG","OD1")],
    "ASP": [("N","CA","CB","CG"), ("CA","CB","CG","OD1")],
    "CYS": [("N","CA","CB","SG")],
    "GLN": [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","OE1")],
    "GLU": [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","OE1")],
    "GLY": [],
    "HIS": [("N","CA","CB","CG"), ("CA","CB","CG","ND1")],
    "ILE": [("N","CA","CB","CG1"), ("CA","CB","CG1","CD1")],
    "LEU": [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "LYS": [("N","CA","CB","CG"), ("CA","CB","CG","CD"),
            ("CB","CG","CD","CE"), ("CG","CD","CE","NZ")],
    "MET": [("N","CA","CB","CG"), ("CA","CB","CG","SD"), ("CB","CG","SD","CE")],
    "PHE": [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "PRO": [("N","CA","CB","CG"), ("CA","CB","CG","CD")],
    "SER": [("N","CA","CB","OG")],
    "THR": [("N","CA","CB","OG1")],
    "TRP": [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "TYR": [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "VAL": [("N","CA","CB","CG1")],
}
SYM_CHI = {
    3: (1,),   # ASP
    6: (2,),   # GLU
    13: (1,),  # PHE
    18: (1,),  # TYR
}
SYM_TBL = torch.zeros(N_AA+1, N_CHI, dtype=torch.bool)
for a, chis in SYM_CHI.items():
    for c in chis:
        SYM_TBL[a, c] = True
# maximum number of altlocs to read from qfit 
MAX_ALTLOCS = 6

class qFitDataset(Dataset):
    '''
    The Dataset class for the qFit altloc data storing the chi angles of 
    multiconformer models and their occupancies
    '''
    def __init__(
        self,
        structures_dir = None,
        paths = None,
        seed = 42,
        max_len = None
    ):

        if paths is not None:
            self.paths = list(paths)
        else:
            self.paths = sorted(Path(structures_dir).glob("*.pdb"))
                                
        self.max_len = max_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return ground_truth(self.paths[idx], max_len = self.max_len)

class ConfidenceDataset(Dataset):
    '''
    Dataset for the confidence model: same backbone+sequence input features
    as qFitDataset, paired with per-residue W1 labels (the Wasserstein
    distance between qFitLatent's predicted chi mixture and the true qFit
    ensemble) precomputed by scripts/make_w1_labels.py.
    '''
    def __init__(self, paths, w1_table, max_len=None):
        self.paths = list(paths)
        self.w1_table = w1_table
        self.max_len = max_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        gt = ground_truth(path, max_len=self.max_len)
        if gt is None:
            return None
        keys = sorted(parse_pdb(path))
        labels = self.w1_table.get(path.stem.lower(), {})
        w1 = torch.tensor(
            [labels.get(key, float("nan")) for key in keys], dtype=torch.float32
        )
        if not torch.isfinite(w1).any():
            return None
        return {"aa_tokens": gt["aa_tokens"], "R": gt["R"], "t": gt["t"], "w1": w1}


def load_w1_table(csv_path):
    '''
    Load scripts/make_w1_labels.py's output: pdb -> {(chain, seq): w1}
    '''
    table = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            key = (row["chain"], int(row["seq"]))
            table.setdefault(row["pdb"].lower(), {})[key] = float(row["w1"])
    return table


def ground_truth(path, max_len):
    '''
    Get the ground truth altloc information from the qfit multiconformer model
    '''
    data = parse_pdb(path)
    keys = sorted(data)
    N = len(keys)
    if max_len and N > max_len:
        return None
    
    # set up the features
    aa_tokens = np.zeros(N, dtype=np.int64) # amino acid tokenization
    R = np.zeros((N, 3, 3), dtype=np.float32) # rotation frames
    t = np.zeros((N, 3), dtype=np.float32) # translation frames (CA)
    chi_angles = np.zeros((N, MAX_ALTLOCS, N_CHI), dtype=np.float32)
    occupancies = np.zeros((N, MAX_ALTLOCS), dtype=np.float32)
    chi_mask = np.zeros((N, N_CHI), dtype=bool)

    for i, key in enumerate(keys):
        residue = data[key]
        altlocs = residue["altlocs"] # the altlocs
        aa_tokens[i] = AA3.get(residue["resname"], 20) # the tokenized sequence

        ca_xyz = get_xyz(altlocs, "CA") # alpha carbon of first altloc 
        c_xyz = get_xyz(altlocs, "C") # bb carbon of first altloc 
        n_xyz = get_xyz(altlocs, "N") # alpha carbon of first altloc 

        t[i] = ca_xyz if ca_xyz is not None else 0.0 # frame translation vector

        # if missing a backbone atom skip otherwise build R rotation matrix
        if n_xyz is None or c_xyz is None or ca_xyz is None:
            R[i] = np.eye(3, dtype=np.float32)
        else:
            R[i] = build_frame(ca_xyz, n_xyz, c_xyz)

        chi_angles[i], occupancies[i], chi_mask[i] = get_chi_info(
            altlocs, residue["resname"]
        )

    return {
        "aa_tokens": torch.from_numpy(aa_tokens),
        "R": torch.from_numpy(R),
        "t": torch.from_numpy(t),
        "chi_angles": torch.from_numpy(chi_angles),
        "occupancies": torch.from_numpy(occupancies),
        "chi_mask": torch.from_numpy(chi_mask)
    }

# parse a pdb file and get relevant information
def parse_pdb(path, max_length = None):
    '''
    Parse a pdb file and get the relevant altloc information stored in nested
    dictionaries
    '''
    output = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"): # skip non atom lines
                continue
            try:
                atom  = line[12:16].strip() # atom name
                alt   = line[16].strip() or " " # altloc identifier for this atom
                res   = line[17:20].strip() # residue for this atom
                chain = line[21] # chain name for this atom
                seq   = int(line[22:26]) # residue number
                xyz   = np.array([float(line[30:38]), float(line[38:46]),
                                  float(line[46:54])], dtype=np.float32) #coords
                occ   = float(line[54:60]) # occupancy for this atom
            except ValueError:
                continue
            key  = (chain, seq) # key for a residue e.g. (A, 32)
            if key not in output: 
                # if it hits max length return
                if max_length and len(output) >= max_length:
                    return output
                # add to output with blank altlocs if not in there already
                output[key] = {"resname": res, "altlocs": {}}
            # if already in there, then add info
            output[key]["altlocs"].setdefault(alt, {})[atom] = (xyz, occ)
    return output # nested dict of altloc information 

def load_subset_paths(subset_csv, structures_dir):
    '''
    Return PDB paths for chains listed in a csv
    '''
    seen = set()
    paths = []
    missing = 0
    with open(subset_csv) as fh:
        for row in csv.DictReader(fh):
            pdb_code = row["pdb_chain"].rsplit("_", 1)[0]
            if pdb_code in seen:
                continue
            seen.add(pdb_code)
            p = structures_dir / f"{pdb_code}.pdb"
            if p.exists():
                paths.append(p)
            else:
                missing += 1
    if missing:
        print(f"WARNING: {missing} PDB files not found in {structures_dir}")
    print(f"{len(paths)} PDB files from {subset_csv.name}")
    return paths

def get_chi_info(altlocs, resname, max=MAX_ALTLOCS):
    '''
    Iterate through the altloc data and extract the chi angles, occupancies and 
    masks
    '''
    chi_defs = CHI_ATOMS.get(resname, [])
    n_chi = len(chi_defs)

    chi_angles = np.zeros((max, N_CHI), dtype=np.float32)
    occupancies = np.zeros(max, dtype=np.float32)
    chi_mask = np.zeros(N_CHI, dtype=bool)
    chi_mask[:n_chi] = True

    if n_chi == 0:
        return chi_angles, occupancies, chi_mask
    
    altloc_data = []
    for alt, atoms in sorted(altlocs.items()): # loop through altlocs
        if alt == " ": # skip blank altlocs
            continue
        
        # get the occupancy from the 
        occ = next((o for _, o in atoms.values()), None)
        if occ is None: 
            continue

        angles = np.zeros(N_CHI, dtype=np.float32)
        valid = True

        # qfit pdbs encode altlocs in delta form: the blank altloc holds
        # shared atoms (typically backbone) and named altlocs A/B/C only
        # carry the atoms that genuinely differ between conformers. only fall
        # back to the blank altloc here, never to another named altloc, so
        # we never silently mix atoms from conformer A into conformer B's chi.
        # ~28% of recorded chi values were previously contaminated this way.
        blank = altlocs.get(" ", {})
        def get(name):
            if name in atoms:
                return atoms[name][0]
            if name in blank:
                return blank[name][0]
            return None

        for ci, (a1n, a2n, a3n, a4n) in enumerate(chi_defs):
            p1, p2, p3, p4 = get(a1n), get(a2n), get(a3n), get(a4n)
            if p1 is None or p2 is None or p3 is None or p4 is None:
                valid = False
                break
            angles[ci] = dihedral(p1, p2, p3, p4)

        if valid:
            altloc_data.append((angles, occ))

        
    if not altloc_data:
        return chi_angles, occupancies, chi_mask
    
    altloc_data.sort(key=lambda x: -x[1])
    n = min(len(altloc_data), MAX_ALTLOCS)
    raw_occs = []
    for j, (angles, occ) in enumerate(altloc_data[:n]):
        chi_angles[j] = angles
        raw_occs.append(occ)

    raw_occs = np.array(raw_occs, dtype=np.float32)
    raw_occs /= raw_occs.sum()
    occupancies[:n] = raw_occs

    return chi_angles, occupancies, chi_mask

def get_xyz(altlocs, atom):
    ''' 
    Get the xyz coordinate of a given atom from either blank altloc or A
    '''
    for alt in (" ", "A", *sorted(altlocs)):
        if alt in altlocs and atom in altlocs[alt]:
            return altlocs[alt][atom][0]
        
    return None

def build_frame(ca, n, c):
    '''
    Built rotation frame putting CA at 0,0,0, CA-C is x axis and N in xy plane,
    z is normal to the CA-C-N plpane
    '''
    x = c - ca;  x /= np.linalg.norm(x) + 1e-8
    bc = n - ca; bc /= np.linalg.norm(bc) + 1e-8
    z = np.cross(x, bc)
    z_norm = np.linalg.norm(z)
    z = np.array([0., 0., 1.]) if z_norm < 1e-6 else z / z_norm
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1).astype(np.float32)

def dihedral(a1, a2, a3 , a4):
    '''
    Dihedral angle calculation from 4 points
    '''
    b1 = a2 - a1
    b2 = a3 - a2
    b3 = a4 - a3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_n = b2 / (np.linalg.norm(b2) + 1e-8)
    m1 = np.cross(n1, b2_n)
    return float(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))
                
                