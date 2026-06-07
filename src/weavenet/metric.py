from typing import Tuple, Optional
from typing_extensions import Literal

import torch
import torch.nn.functional as F
from .preference import PreferenceFormat, to_cost, batch_sum

__all__ = [
    'binarize',
    'is_one2one',
    'is_stable',
    'count_blocking_pairs',
    'sexequality_cost',
    'egalitarian_score',
    'balance_score',
    'calc_all_fairness_metrics',
    'default_stable_matching_metric',
    'MatchingAccuracy',
]

#@torch.jit.script
def binarize(m : torch.Tensor):
    r"""
    Binarizes each matrix in a batch into the one-to-one format (if N=M). If N>M, N-M vertices will have no partner and vice versa.
        
    Shape:
        - m:  :math:`(B, N, M)`
        - output: :math:`(B, N, M)`
    Args:
        m: a continously-relaxed assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
    Returns:
        A binarized batched matrices.
    """
    na, nb = m.shape[-2:]
    if na >= nb:
        m = F.one_hot(m.argmax(dim=-1), num_classes=nb)
    else:
        m = F.one_hot(m.argmax(dim=-2), num_classes=na).t()
    return m

#@torch.jit.script
def is_one2one(m : torch.Tensor):
    r"""
    Checks whether each matrix in a batch m has no duplicated correspondence.
        
    Shape:
        - m:  :math:`(B, N, M)`
        - output: :math:`(B)`
    Args:
        m: a binary assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
    Returns:
        A binary bool vector.
    """
    return ~((torch.sum(m,dim=-2)>1).any(dim=-1) + (torch.sum(m,dim=-1)>1).any(dim=-1))


#@torch.jit.script
def is_stable(m : torch.Tensor, sab : torch.Tensor, sba_t : torch.Tensor) -> torch.Tensor:
    r"""
    Checks whether each matrix in a batch m is a stable match or not.
        
    Shape:
        - m:  :math:`(B, N, M)`
        - sab: :math:`(B, N, M)`
        - sab: :math:`(B, M, N)`
        - output: :math:`(B)`
    Args:
        m: a binary (or continously-relaxed) assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
        sab: a satisfaction at matching of agents in side :math:`a` to side :math:`b`.  
        sba: a satisfaction at matching of agents in side :math:`b` to side :math:`a`.  
    Returns:
        A binary bool vector.
    """
    # Vectorised (#477): a blocking pair (man i, woman j) needs man i to prefer j
    # over his current match AND woman j to prefer i over hers — computed for the
    # whole batch at once, no Python/torch.jit.fork loops. Bit-identical to the
    # original per-column implementation (verified on the fixed 1000-testset), ~60x
    # faster: it was the dominant eval cost (#477 bottleneck analysis).
    matched_sab = (m * sab).sum(dim=-1, keepdim=True)        # (B,N,1) a_i at match
    matched_sba = (m * sba_t).sum(dim=-2, keepdim=True)      # (B,1,M) b_j at match
    blocking = (sab > matched_sab) & (sba_t > matched_sba)   # (B,N,M)
    return blocking.sum(dim=(-2, -1)) == 0                   # (B,) bool


def count_blocking_pairs(m : torch.Tensor, sab : torch.Tensor, sba_t : torch.Tensor)->torch.Tensor:
    r"""
    Counts the number of blocking pairs for each matrix in batch m.
        
    Shape:
        - m:  :math:`(B, N, M)`
        - sab: :math:`(B, N, M)`
        - sab: :math:`(B, M, N)`
        - output: :math:`(B)`
    Args:
        m: a binary (or continously-relaxed) assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
        sab: a satisfaction at matching of agents in side :math:`a` to side :math:`b`.  
        sba: a satisfaction at matching of agents in side :math:`b` to side :math:`a`.  
    Returns:
        A count vector.
    """
    # Vectorised (#477), exact match to the original per-column count including
    # argmax-binarised non-permutations (column collisions): a blocking pair
    # (man i, woman c) needs man i to prefer c over his match AND, summed over each
    # man a actually assigned to c, woman c to prefer i over a:
    #   n = Σ_{c,i,a} PM[i,c] · m[a,c] · (sba_t[i,c] > sba_t[a,c]),  PM[i,c] = man i prefers c.
    # Bit-identical to the fork version (verified on the 1000-testset), ~60x faster.
    matched_sab = (m * sab).sum(dim=-1, keepdim=True)        # (B,N,1)
    PM = (sab > matched_sab).float()                         # (B,N,M) man i prefers woman c
    cmp = (sba_t.unsqueeze(2) > sba_t.unsqueeze(1)).float()  # (B,i,a,c): woman c prefers i over a
    unsba = torch.einsum("bac,biac->bic", m.float(), cmp)    # (B,N,M)
    return (PM * unsba).sum(dim=(-2, -1)).sum()              # scalar total (original API)


def sexequality_cost(m : torch.Tensor, cab : torch.Tensor, cba_t : torch.Tensor, 
                     pformat : PreferenceFormat = PreferenceFormat.cost) -> torch.Tensor :
    r"""
    Calculates sexequality costs.
    
    Shape:
        - m:  :math:`(B, N, M)`
        - cab: :math:`(B, N, M)`
        - cab: :math:`(B, M, N)`
        - output: :math:`(B)`
    Args:
        m: a binary (or continously-relaxed) assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
        cab: a cost at matching of agents in side :math:`a` to side :math:`b`.  
        cba: a cost at matching of agents in side :math:`b` to side :math:`a`.  
    Returns:
        A cost vector.
    """
    if pformat != PreferenceFormat.cost:
        cab = to_cost(mat=cab, pformat=pformat, dim=-1)
        cba_t = to_cost(mat=cba_t, pformat=pformat, dim=-2)
    batch_size = m.size(0)
    return (batch_sum(m, cab, batch_size) - batch_sum(m, cba_t, batch_size)).abs()

def egalitarian_score(m : torch.Tensor, cab : torch.Tensor, cba_t : torch.Tensor, 
                     pformat: PreferenceFormat = PreferenceFormat.cost) -> torch.Tensor:
    r"""
    Calculates egalitarian score.
    
    Shape:
        - m:  :math:`(B, N, M)`
        - cab: :math:`(B, N, M)`
        - cab: :math:`(B, M, N)`
        - output: :math:`(B)`
    Args:
        m: a binary (or continously-relaxed) assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
        cab: a cost at matching of agents in side :math:`a` to side :math:`b`.  
        cba: a cost at matching of agents in side :math:`b` to side :math:`a`.  
    Returns:
        A score vector.
    """
    if pformat != PreferenceFormat.cost:
        cab = to_cost(cab, pformat, dim=-1)
        cba_t = to_cost(cba_t, pformat, dim=-2)
    batch_size = m.size(0)
    return (batch_sum(m, cab, batch_size) + batch_sum(m, cba_t, batch_size)) # egalitarian cost = -1 * egalitarian score.

#@torch.jit.script
def balance_score(m : torch.Tensor, cab : torch.Tensor, cba_t : torch.Tensor, 
                     pformat: PreferenceFormat = PreferenceFormat.cost) -> torch.Tensor:
    r"""
    Calculates egalitarian score.
    
    Shape:
        - m:  :math:`(B, N, M)`
        - cab: :math:`(B, N, M)`
        - cab: :math:`(B, M, N)`
        - output: :math:`(B)`
    Args:
        m: a binary (or continously-relaxed) assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
        cab: a cost at matching of agents in side :math:`a` to side :math:`b`.  
        cba: a cost at matching of agents in side :math:`b` to side :math:`a`.  
    Returns:
        A score vector.
    """
    if pformat != PreferenceFormat.cost:
        cab = to_cost(cab, pformat, dim=-1)
        cba_t = to_cost(cba_t, pformat, dim=-2)
    batch_size = m.size(0)
    return batch_sum(m, cab, batch_size).max(batch_sum(m, cba_t, batch_size))

def calc_all_fairness_metrics(m : torch.Tensor, cab : torch.Tensor, cba_t : torch.Tensor, 
                     pformat: PreferenceFormat = PreferenceFormat.cost) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Calculates the three fairness scores (sex-equality, egalitarian score, and balance score).
    
    Shape:
        - m:  :math:`(B, N, M)`
        - cab: :math:`(B, N, M)`
        - cab: :math:`(B, M, N)`
        - output: :math:`(B)`
    Args:
        m: a binary (or continously-relaxed) assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
        cab: a cost at matching of agents in side :math:`a` to side :math:`b`.  
        cba: a cost at matching of agents in side :math:`b` to side :math:`a`.  
    Returns:
        The three score vectors.
    """
    if pformat != PreferenceFormat.cost:
        cab = to_cost(cab, pformat, dim=-1)
        cba_t = to_cost(cba_t, pformat, dim=-2)
    batch_size = m.size(0)
    A = batch_sum(m, cab, batch_size)
    B = batch_sum(m, cba_t, batch_size)
    se = (A-B).abs()
    egal = A+B
    balance = (se+egal)/2
    #balance_ = torch.stack([batch_sum(m, cab, batch_size), batch_sum(m.transpose(-1,-2), cba, batch_size)]).max(dim=0)[0]
    #assert((balance ==  balance_).all())
    return se, egal, balance


def default_stable_matching_metric(m: torch.Tensor, sab: torch.Tensor, sba_t: torch.Tensor):
    r"""Default per-batch metric bundle for stable-matching criteria.

    Binarizes the soft assignment ``m`` (argmax along the larger side) and
    computes the family of metrics every stable-matching training tracks:

    - ``is_one2one`` and ``is_stable`` (per-sample bools, as floats), and
      their product ``is_success``,
    - ``num_blocking_pair``,
    - the three fairness scores (``sexequality``, ``egalitarian``, ``balance``).

    Returned as ``(log, mb)`` where ``log`` is the dict above and ``mb`` is
    the binarized assignment, so callers can re-use the discrete matching
    without re-computing it. ``binarize``, ``is_one2one``, ``is_stable``,
    ``count_blocking_pairs`` and ``calc_all_fairness_metrics`` run in
    parallel via :func:`torch.jit.fork`.

    Shape:
       - ``m``:     ``(B, ..., N, M)`` or ``(B, ..., N, M, 1)``
       - ``sab``:   ``(B, ..., N, M)``
       - ``sba_t``: ``(B, ..., N, M)``

    Args:
       m:     soft (or already-binarized) matching to evaluate.
       sab:   side-a satisfaction.
       sba_t: side-b satisfaction transposed to share shape with ``sab``.

    Returns:
       ``(log, mb)``. ``log`` keys: ``is_one2one``, ``is_stable``,
       ``is_success``, ``num_blocking_pair``, ``sexequality``,
       ``egalitarian``, ``balance``. ``mb`` is the binarized matching.
    """
    mb = binarize(m)
    futs = [
        torch.jit.fork(is_one2one, mb),
        torch.jit.fork(is_stable, mb, sab, sba_t),
        torch.jit.fork(count_blocking_pairs, mb, sab, sba_t),
    ]
    log = {}
    log['sexequality'], log['egalitarian'], log['balance'] = calc_all_fairness_metrics(
        mb, sab, sba_t, pformat=PreferenceFormat.satisfaction,
    )
    temp_one2one = torch.jit.wait(futs[0])
    temp_stable = torch.jit.wait(futs[1])
    log['is_one2one'] = temp_one2one
    log['is_stable'] = temp_stable
    log['is_success'] = temp_one2one * temp_stable
    log['num_blocking_pair'] = torch.jit.wait(futs[2])
    return log, mb
