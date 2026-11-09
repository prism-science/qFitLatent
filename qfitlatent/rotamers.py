# created by clay
'''
Backbone-dependent Dunbrack rotamer prior lookup.

Loads the tensors built by scripts/build_rotamer_prior.py and, for each residue,
returns the library's rotamer distribution conditioned on its backbone (phi, psi).
This is the distribution the model shifts toward the observed qFit altloc
distribution; mu/sigma are the library's chi means and standard deviations.
'''

import os
from pathlib import Path

import torch

# the prior bundle to load; override with the ROTAMER_PRIOR env var to swap in a
# variant (e.g. a sharpened-sigma "finer" library) without touching callers
DEFAULT_PRIOR_PATH = Path(os.environ.get(
    "ROTAMER_PRIOR",
    str(Path(__file__).resolve().parent.parent / "external" / "rotamer_prior.pt")))
N_BINS = 36
LOG_ZERO = -1e30

_CACHE = {}


def load_prior(path=DEFAULT_PRIOR_PATH, device=None):
    '''Load (and cache) the prior bundle, optionally moved to a device.'''
    key = (str(path), str(device))
    if key not in _CACHE:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if device is not None:
            bundle = {k: (v.to(device) if torch.is_tensor(v) else v)
                      for k, v in bundle.items()}
        _CACHE[key] = bundle
    return _CACHE[key]


def n_chi_per_aa(path=DEFAULT_PRIOR_PATH):
    '''
    Chi angles per amino acid as parameterized by the library.

    Note this can be smaller than the geometric chi count (PRO: the library only
    treats chi1 as rotameric), so it is the authoritative mask for anything that
    reads library mu/sigma.
    '''
    return load_prior(path)["n_chi"]


def _bin(angle_rad):
    '''Backbone angle (radians) -> library bin index, matching the .lib grid.'''
    deg = torch.rad2deg(angle_rad)
    return (((deg + 180.0) / 10.0).floor().long()) % N_BINS


def lookup(aa_tokens, phi, psi, bb_mask, path=DEFAULT_PRIOR_PATH):
    '''
    Per-residue rotamer prior conditioned on backbone.

    Inputs:
        aa_tokens: [N] long, 0..19 (20/UNK is treated as having no rotamers)
        phi, psi:  [N] float radians
        bb_mask:   [N, 2] bool, whether phi / psi are defined

    Returns:
        log_prior: [N, S] log p(rotamer | aa, phi, psi), -1e30 on invalid slots
        mu:        [N, S, 4] library chi means (radians)
        sigma:     [N, S, 4] library chi standard deviations (radians)
        slot_mask: [N, S] bool, which slots are real rotamers for that residue

    Residues whose phi or psi is undefined (chain termini, gap edges) fall back to
    the backbone-marginal distribution for that amino acid rather than indexing a
    meaningless bin.
    '''
    bundle = load_prior(path, device=aa_tokens.device)
    log_prob, mu_t, sigma_t = bundle["log_prob"], bundle["mu"], bundle["sigma"]
    slot_mask_t = bundle["slot_mask"]

    # UNK (20) has no rotamer table; clamp the index and mask it out below
    aa = aa_tokens.clamp(max=log_prob.shape[0] - 1)
    valid_aa = aa_tokens < log_prob.shape[0]

    pb, sb = _bin(phi), _bin(psi)
    log_prior = log_prob[aa, pb, sb]          # [N, S]
    mu = mu_t[aa, pb, sb]                     # [N, S, 4]
    sigma = sigma_t[aa, pb, sb]               # [N, S, 4]

    # backbone-marginal fallback where phi or psi is undefined
    undefined = ~(bb_mask[:, 0] & bb_mask[:, 1])
    if undefined.any():
        prob_all = log_prob.exp()                             # [20, 36, 36, S]
        marg = prob_all.mean(dim=(1, 2))                      # [20, S]
        marg = marg / marg.sum(-1, keepdim=True).clamp_min(1e-12)
        log_marg = marg.clamp_min(1e-30).log()
        # marginal chi means/sigmas: probability-weighted over backbone bins
        w = prob_all / prob_all.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        mu_marg = (mu_t * w[..., None]).sum(dim=(1, 2))       # [20, S, 4]
        sigma_marg = (sigma_t * w[..., None]).sum(dim=(1, 2))

        idx = undefined.nonzero(as_tuple=True)[0]
        a = aa[idx]
        log_prior = log_prior.clone()
        mu = mu.clone()
        sigma = sigma.clone()
        log_prior[idx] = log_marg[a]
        mu[idx] = mu_marg[a]
        sigma[idx] = sigma_marg[a].clamp_min(1e-3)

    slot_mask = slot_mask_t[aa] & valid_aa[:, None]
    log_prior = log_prior.masked_fill(~slot_mask, LOG_ZERO)
    return log_prior, mu, sigma, slot_mask


# chi angles per amino acid as the library parameterizes them (featurize token
# order). Smaller than the geometric count for PRO (chi1 only); fixed, so the model
# does not need the library bundle at runtime.
LIBRARY_N_CHI = torch.tensor([0, 4, 2, 2, 1, 3, 3, 0, 2, 2,
                              2, 4, 3, 2, 1, 1, 1, 2, 2, 1])


def chi_valid(aa_tokens, chi_mask):
    '''
    Intersect the geometric chi mask from featurize with what the library actually
    parameterizes, so no likelihood is ever evaluated against an unpopulated
    mu/sigma slot (this is what keeps PRO chi2 out of the loss).
    '''
    n_chi = LIBRARY_N_CHI.to(aa_tokens.device)
    aa = aa_tokens.clamp(max=n_chi.shape[0] - 1)
    counts = torch.where(aa_tokens < n_chi.shape[0], n_chi[aa],
                         torch.zeros_like(aa))
    arange = torch.arange(chi_mask.shape[-1], device=chi_mask.device)
    return chi_mask & (arange[None, :] < counts[:, None])
