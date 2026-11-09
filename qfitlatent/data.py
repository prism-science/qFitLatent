# created by clay
'''
Dataset for qFit altloc data. Featurization (parsing, alignment, gap filling,
chi/frame extraction) lives in featurize.py; this module only turns a set of PDB
paths into per-protein-chain samples for a DataLoader.
'''

from pathlib import Path

from torch.utils.data import Dataset


class qFitDataset(Dataset):
    '''
    One sample per protein chain: builds an index of (path, chain) up front and
    featurizes lazily in __getitem__ via featurize.featurize_pdb.
    '''
    def __init__(
        self,
        structures_dir = None,
        paths = None,
        seed = 42,
        max_len = None,
        seq_dir = None
    ):

        if paths is not None:
            self.paths = list(paths)
        else:
            self.paths = sorted(Path(structures_dir).glob("*.pdb"))

        self.max_len = max_len
        self.seq_dir = seq_dir

        # one sample per protein chain: build an index of (path, chain).
        # a single unreadable structure must not kill a whole training run, so
        # unparseable files are skipped with a warning
        from qfitlatent.featurize import list_protein_chains
        self.index = []
        self.skipped = []
        for p in self.paths:
            try:
                chains = list_protein_chains(p)
            except Exception as e:
                self.skipped.append((p, f"{type(e).__name__}: {e}"))
                continue
            for chain in chains:
                self.index.append((p, chain))
        if self.skipped:
            print(f"WARNING: skipped {len(self.skipped)} unreadable structure(s); "
                  f"first: {Path(self.skipped[0][0]).name} -> {self.skipped[0][1][:80]}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        from qfitlatent.featurize import DEFAULT_SEQ_DIR, featurize_pdb
        path, chain = self.index[idx]
        feats = featurize_pdb(path, seq_dir=self.seq_dir or DEFAULT_SEQ_DIR)
        sample = feats[chain]
        if self.max_len and sample["aa_tokens"].shape[0] > self.max_len:
            return None
        return sample


# --- preloading: featurize once, in parallel, into a resumable sharded cache ---

def structure_paths(structures_dir=None, list_file=None):
    '''PDB paths from a directory (*.pdb) or a list file (one path per line).'''
    if list_file:
        return [Path(line.strip()) for line in Path(list_file).read_text().splitlines()
                if line.strip()]
    return sorted(Path(structures_dir).glob("*.pdb"))


def featurize_structure(job):
    '''
    Every protein chain of one structure as (name, sample) pairs, each carrying a
    boolean `score` mask of the residues the loss is measured on (resolved, >=1
    parameterized chi, >=1 occupied altloc). Unreadable files and chains with
    nothing to score are dropped; one bad structure never stops a run.

    Tensors are returned as numpy arrays: torch would hand them back to the parent
    through shared memory, which runs out once tens of thousands accumulate.
    '''
    import torch
    from qfitlatent.featurize import DEFAULT_SEQ_DIR, featurize_pdb
    from qfitlatent.model import pick_mask_indices
    from qfitlatent.rotamers import chi_valid
    path, seq_dir, max_len = job
    try:
        chains = featurize_pdb(path, seq_dir=seq_dir or DEFAULT_SEQ_DIR)
    except Exception as e:
        return [], f"{Path(path).name}: {type(e).__name__}: {e}"
    out = []
    for chain, sample in chains.items():
        if max_len and sample["aa_tokens"].shape[0] > max_len:
            continue
        valid = chi_valid(sample["aa_tokens"], sample["chi_mask"])
        score = torch.zeros(len(sample["aa_tokens"]), dtype=torch.bool)
        score[pick_mask_indices(sample["resolved_mask"], sample["occupancies"], valid)] = True
        if score.any():
            sample["chi_valid"], sample["score"] = valid, score
            out.append((f"{Path(path).stem}_{chain}",
                        {k: v.numpy() for k, v in sample.items()}))
    return out, None


def preload(paths, seq_dir=None, max_len=None, workers=1, progress_every=2000):
    '''Featurize structures (in `workers` processes) -> [(name, sample)] on CPU.'''
    from multiprocessing import get_context
    import torch
    jobs = [(str(p), seq_dir, max_len) for p in paths]
    samples, failed = [], []
    with get_context("spawn").Pool(workers) as pool:
        for k, (chains, err) in enumerate(pool.imap(featurize_structure, jobs, chunksize=8)):
            samples.extend((name, {k: torch.from_numpy(v) for k, v in d.items()})
                           for name, d in chains)
            if err:
                failed.append(err)
            if progress_every and (k + 1) % progress_every == 0:
                print(f"    featurized {k + 1}/{len(jobs)} structures "
                      f"({len(samples)} chains)", flush=True)
    if failed:
        print(f"    WARNING: skipped {len(failed)} unreadable structure(s); "
              f"first: {failed[0][:100]}")
    return samples


def load_or_preload(cache, paths, seq_dir=None, max_len=None, workers=1, shard_size=2000):
    '''
    Featurized samples (on CPU), cached so repeated runs skip featurization.

    `cache` ending in .pt is a single cache file (the original format). Any other
    path is a directory of shards of `shard_size` structures each, written as they
    finish, so an interrupted preload of a large set resumes where it stopped.
    The cache is keyed only by path; delete it if the structures or featurizer change.
    '''
    import torch
    if cache is None:
        return preload(paths, seq_dir, max_len, workers)
    cache = Path(cache)
    if cache.suffix == ".pt":
        if cache.exists():
            print(f"  loading cached features from {cache}")
            return torch.load(cache, map_location="cpu", weights_only=False)
        samples = preload(paths, seq_dir, max_len, workers)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples, cache)
        print(f"  cached features to {cache}")
        return samples
    cache.mkdir(parents=True, exist_ok=True)
    samples = []
    for start in range(0, len(paths), shard_size):
        shard = cache / f"shard_{start // shard_size:04d}.pt"
        if not shard.exists():
            print(f"  featurizing structures {start}-{start + shard_size} -> {shard.name}",
                  flush=True)
            torch.save(preload(paths[start:start + shard_size], seq_dir, max_len, workers,
                               progress_every=0), shard)
        samples.extend(torch.load(shard, map_location="cpu", weights_only=False))
    print(f"  loaded {len(samples)} chains from {cache}")
    return samples


# --- pRSCC targets (scripts/build_rscc_targets.py) ------------------------------

def attach_rscc_targets(samples, csv_paths):
    '''
    Add each chain's pRSCC targets from build_rscc_targets.py CSVs as [N] tensors:
    rscc_group (mask group, -1 = no target), rscc_masked and rscc_visible (nan = no
    target). Chains without any target are dropped. Returns the kept samples.
    '''
    import csv
    import torch
    targets = {}
    for path in csv_paths:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                targets.setdefault(f"{row['pdb']}_{row['chain']}", []).append(row)
    kept = []
    for name, sample in samples:
        rows = targets.get(name)
        if not rows:
            continue
        n = len(sample["aa_tokens"])
        group = torch.full((n,), -1, dtype=torch.long)
        masked = torch.full((n,), float("nan"))
        visible = torch.full((n,), float("nan"))
        for row in rows:
            i = int(row["index"])
            group[i] = int(row["group"])
            masked[i], visible[i] = float(row["rscc_masked"]), float(row["rscc_visible"])
        sample.update(rscc_group=group, rscc_masked=masked, rscc_visible=visible)
        kept.append((name, sample))
    return kept
