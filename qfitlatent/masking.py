# created by clay
'''
Which scored residues of a chain are hidden behind the mask token.

  random    each scored residue independently
  sequence  clusters of residues close along the chain
  spatial   clusters of residues close in space (CA-CA distance)

A cluster is a random visible seed plus its nearest scored residues, `cluster_size`
in total, so the sequence and spatial strategies differ only in the distance used.
Clusters are added until the chain's budget of `mask_frac` of its scored residues is
spent, and may overlap and merge. The cluster size does not depend on `mask_frac`.

featurize.py reads the backbone from each residue's highest-occupancy altloc, but
a residue with no CA at all is left at the origin (t = 0), so for the spatial
strategy its position is interpolated along the chain; otherwise all such residues
would count as each other's neighbours.
'''

import numpy as np
import torch

MASK_STRATEGIES = ("random", "sequence", "spatial")


def hidden_budget(n_scored, mask_frac, generator=None):
    '''Residues to hide: mask_frac * n_scored, rounded at random so that short chains
    still hide mask_frac of their residues on average.'''
    return int(mask_frac * n_scored + torch.rand(1, generator=generator).item())


def interpolate_missing_ca(ca):
    '''[N, 3] CA with origin entries interpolated from located residues along the chain.'''
    located = (ca != 0).any(-1)
    if located.all() or not located.any():
        return ca
    indices = np.arange(len(ca))
    filled = [np.interp(indices, indices[located.numpy()], ca[located, axis].numpy())
              for axis in range(3)]
    return torch.from_numpy(np.stack(filled, -1)).to(ca.dtype)


def cluster_positions(ca, strategy):
    '''Coordinates whose Euclidean distance defines "close" [N, D].'''
    if strategy == "sequence":
        return torch.arange(len(ca), dtype=ca.dtype)[:, None]
    return interpolate_missing_ca(ca)


def grow_clusters(positions, budget, cluster_size, generator=None):
    '''Hide `budget` of the [M, D] positions, as clusters. Returns [M] bool.'''
    hidden = torch.zeros(len(positions), dtype=torch.bool)
    n_hidden = 0
    while n_hidden < budget:
        visible = (~hidden).nonzero(as_tuple=True)[0]
        seed = visible[torch.randint(len(visible), (1,), generator=generator)]
        distances = (positions - positions[seed]).norm(dim=-1)
        distances[seed] = -1.0      # coincident residues must not push out the seed
        cluster = distances.argsort()[:cluster_size]
        added = cluster[~hidden[cluster]][:budget - n_hidden]
        hidden[added] = True
        n_hidden += len(added)
    return hidden


def draw_hidden(score, ca, strategy, mask_frac, cluster_size=12, generator=None):
    '''
    The hidden residues of one chain [N] bool, a subset of `score` [N] bool. `ca` [N, 3]
    is on the same (CPU) device as `generator`.
    '''
    if strategy == "random":
        return score & (torch.rand(len(score), generator=generator) < mask_frac)
    scored = score.nonzero(as_tuple=True)[0]
    positions = cluster_positions(ca, strategy)[scored]
    budget = hidden_budget(len(scored), mask_frac, generator)
    hidden = torch.zeros_like(score)
    hidden[scored[grow_clusters(positions, budget, cluster_size, generator)]] = True
    return hidden
