# created by clay
'''
Load a trained checkpoint and embed qFit multiconformer structures.

The embedding of a residue is the latent mean mu_z (the dynamics representation),
computed in autoencoding mode: every resolved residue's altloc ensemble is visible
and only unresolved (unmodeled) positions get the mask token.
'''

from pathlib import Path

import numpy as np
import torch

from .featurize import DEFAULT_SEQ_DIR, _AA, featurize_pdb_ctx, list_protein_chains
from .model import qFitLatent

MODEL_INPUTS = ("aa_tokens", "R", "t", "chi_angles", "occupancies", "chi_mask",
                "phi", "psi", "bb_mask")


def load_model(path, device="cpu"):
    '''Build the model from a train.py checkpoint's saved args and load its weights.'''
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    model = qFitLatent(d_single=args.get("d_single", 256),
                            d_pair=args.get("d_pair", 64),
                            d_latent=args.get("d_latent", 16),
                            n_ipa=args.get("n_ipa", 8),
                            n_conf=args.get("n_conf", 16),
                            env_neighbours=args.get("env_neighbours", 0),
                            prscc_input=args.get("prscc_input", "latent"))
    missing, _ = model.load_state_dict(checkpoint["model"], strict=False)
    # the hetero and pRSCC heads are optional; anything else missing is a mismatch
    missing = [key for key in missing if not key.startswith(("hetero_head", "rscc_head"))]
    if missing:
        raise RuntimeError(f"checkpoint does not match model.py; missing {missing[:4]}")
    return model.to(device).eval()


@torch.no_grad()
def embed_chain(model, features, device="cpu"):
    '''Latent mean and log-variance for one featurized chain: ([N, d], [N, d]).'''
    inputs = [features[key][None].to(device) for key in MODEL_INPUTS]
    hide = ~features["resolved_mask"][None].to(device)
    sc_atoms = features["sc_atoms"][None].to(device)
    outputs = model(*inputs, hide, sc_atoms=sc_atoms)
    mu_z, logvar_z = outputs[4], outputs[5]
    return mu_z[0].cpu(), logvar_z[0].cpu()


def residue_labels(features, residues, canon_to_obs):
    '''Per canonical position: author residue number (-1 if unmodeled) and name.'''
    resseq, resname = [], []
    for index, token in enumerate(features["aa_tokens"].tolist()):
        observed = canon_to_obs.get(index)
        if observed is None:
            resseq.append(-1)
            resname.append(_AA[token] if token < len(_AA) else "UNK")
        else:
            residue = residues[observed]
            resseq.append(int(residue.res_id[0]))
            resname.append(str(residue.res_name[0]))
    return resseq, resname


def embed_pdb(model, pdb_path, seq_dir=DEFAULT_SEQ_DIR, device="cpu"):
    '''
    Embed every protein chain of a qFit PDB. Returns a dict of aligned per-residue
    arrays: emb [N, d], logvar [N, d], chain, resseq, resname, aa_token, resolved.
    '''
    columns = {key: [] for key in ("emb", "logvar", "chain", "resseq", "resname",
                                   "aa_token", "resolved")}
    for chain_id in list_protein_chains(pdb_path):
        features, residues, canon_to_obs = featurize_pdb_ctx(pdb_path, chain_id, seq_dir)
        mu_z, logvar_z = embed_chain(model, features, device)
        resseq, resname = residue_labels(features, residues, canon_to_obs)
        columns["emb"].append(mu_z.numpy())
        columns["logvar"].append(logvar_z.numpy())
        columns["chain"] += [chain_id] * len(resseq)
        columns["resseq"] += resseq
        columns["resname"] += resname
        columns["aa_token"].append(features["aa_tokens"].numpy())
        columns["resolved"].append(features["resolved_mask"].numpy())
    return {
        "emb": np.concatenate(columns["emb"]),
        "logvar": np.concatenate(columns["logvar"]),
        "chain": np.array(columns["chain"]),
        "resseq": np.array(columns["resseq"]),
        "resname": np.array(columns["resname"]),
        "aa_token": np.concatenate(columns["aa_token"]),
        "resolved": np.concatenate(columns["resolved"]),
    }
