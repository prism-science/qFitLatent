# created by clay
'''
Fetch canonical per-chain sequences from RCSB for every PDB file in the
structures directory and cache them as one FASTA per chain in the sequences
directory.

The qFit PDB files carry no SEQRES / REMARK 465 and their author numbering does
not correspond to a canonical sequence, so we pull the deposited entity sequence
from RCSB. featurize.py later aligns the structure-observed sequence to these to
recover which canonical positions are resolved vs. unmodeled.
'''

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request
from pathlib import Path

RCSB_FASTA_URL = "https://www.rcsb.org/fasta/entry/{pdbid}"


def pdb_ids_in(structures_dir):
    '''Return the sorted set of 4-char PDB ids found under structures_dir.'''
    ids = set()
    for p in Path(structures_dir).rglob("*.pdb"):
        ids.add(p.stem.lower())
    return sorted(ids)


def fetch_fasta(pdbid, timeout=30):
    '''Fetch the raw multi-record FASTA text for a PDB id from RCSB.'''
    url = RCSB_FASTA_URL.format(pdbid=pdbid.upper())
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def parse_fasta_chains(text):
    '''
    Parse an RCSB FASTA into {chain_letter: sequence}.

    Headers look like ">1A0B_1|Chain A|..." or ">XXXX_2|Chains A, B|..."; the
    same sequence is written once per chain letter it covers.
    '''
    chains = {}
    header = None
    seq_parts = []

    def flush():
        if header is None:
            return
        seq = "".join(seq_parts)
        # the chain field is the 2nd pipe-delimited token, e.g. "Chains A, B"
        fields = header.split("|")
        if len(fields) < 2:
            return
        chain_field = fields[1]
        chain_field = chain_field.replace("Chains", "").replace("Chain", "")
        for chain in chain_field.split(","):
            chain = chain.strip()
            if chain:
                chains[chain] = seq

    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line.strip())
    flush()
    return chains


def chain_files_exist(out_dir, pdbid):
    '''True if at least one chain FASTA already exists for this id.'''
    return any(Path(out_dir).glob(f"{pdbid}_*.fasta"))


def write_chains(out_dir, pdbid, chains):
    '''Write one FASTA per chain: {pdbid}_{chain}.fasta.'''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for chain, seq in chains.items():
        path = out_dir / f"{pdbid}_{chain}.fasta"
        path.write_text(f">{pdbid}_{chain}\n{seq}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures-dir", default="data/structures")
    parser.add_argument("--out-dir", default="data/seqeuences")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if chain files already exist")
    parser.add_argument("--sleep", type=float, default=0.2,
                        help="Seconds each worker waits between requests")
    parser.add_argument("--workers", type=int, default=1,
                        help="concurrent requests")
    args = parser.parse_args()

    ids = pdb_ids_in(args.structures_dir)
    if not ids:
        print(f"No .pdb files found under {args.structures_dir}", file=sys.stderr)
        return

    todo = [pdbid for pdbid in ids
            if args.force or not chain_files_exist(args.out_dir, pdbid)]
    n_skipped = len(ids) - len(todo)

    def fetch_one(pdbid):
        """Fetch and write one entry; returns True on success."""
        try:
            chains = parse_fasta_chains(fetch_fasta(pdbid))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"WARNING: failed to fetch {pdbid}: {e}", file=sys.stderr)
            return False
        finally:
            time.sleep(args.sleep)
        if not chains:
            print(f"WARNING: no chains parsed for {pdbid}", file=sys.stderr)
            return False
        write_chains(args.out_dir, pdbid, chains)
        return True

    # each worker sleeps between its own requests, so the request rate is
    # roughly workers / sleep per second
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(fetch_one, todo))
    n_fetched = sum(results)
    n_failed = len(results) - n_fetched
    print(f"\nfetched {n_fetched}, skipped {n_skipped}, failed {n_failed} "
          f"(of {len(ids)} ids)")


if __name__ == "__main__":
    main()
