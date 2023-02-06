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

def __iss(m : torch.Tensor, sab : torch.Tensor, sba : torch.Tensor, c:int) -> torch.Tensor:
    sab_selected = sab[:,c:c+1].expand(sab.shape) # to keep dimension, sab[:,c] is implemented as sab[:,c:c+1]
    #sab_selected = sab_selected.repeat_interleave(M,dim=1)

    unsab = (m*torch.clamp(sab_selected-sab,min=0)).mean(dim=1)

    sba_selected = sba[c:c+1,:].expand(sba.shape) # keep dimension.
    #sba_selected = sba_selected.repeat_interleave(N,dim=0)
    _sba = sba_selected.t()
    _m = m[:,c:c+1].expand(m.shape)
    #_m = _m.repeat_interleave(N,dim=1)
    unsba = (_m*torch.clamp(sba_selected-_sba,min=0)).mean(dim=0)
    envy = (unsab*unsba).sum()
    return envy<=0
    
def _is_stable(m : torch.Tensor, sab : torch.Tensor, sba : torch.Tensor) -> torch.Tensor:
    M = sba.shape[0]
    futs = [torch.jit.fork(__iss, m, sab, sba, c) for c in range(M)]
    return torch.stack([torch.jit.wait(fut) for fut in futs]).all()


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
    sba = sba_t.transpose(-1,-2)
    futs = [torch.jit.fork(_is_stable,_m,_sab,_sba) for _m,_sab,_sba in zip(m, sab, sba)]
    return torch.stack([torch.jit.wait(fut) for fut in futs])
    #return torch.tensor([_is_stable(_m,_sab,_sba) for _m,_sab,_sba in zip(m, sab, sba)], dtype=torch.bool, device=m.device)

def __cbp(m : torch.Tensor, sab : torch.Tensor, sba : torch.Tensor, c:int)->torch.Tensor:
    sab_selected = sab[:,c:c+1].expand(sab.shape)
    unsab_target = (m*(sab_selected-sab)>0).sum(dim=1) 
    sba_selected = sba[c:c+1,:].expand(sba.shape) 
    _sba = sba_selected.t()
    _m = m[:,c:c+1].expand(m.shape)
    unsba_target = (_m*(sba_selected-_sba)>0).sum(dim=0) 
    n = (unsab_target * unsba_target).sum()
    return n

def _count_blocking_pairs(m : torch.Tensor, sab : torch.Tensor, sba : torch.Tensor)->torch.Tensor:
    M = sba.shape[0]

    n_blocking_pair = 0
    futs = [torch.jit.fork(__cbp, m, sab, sba, c) for c in range(M)]
    return torch.stack([torch.jit.wait(fut) for fut in futs]).sum()


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
    sba = sba_t.transpose(-1,-2)
    futs = [torch.jit.fork(_count_blocking_pairs, _m,_sab,_sba) for _m,_sab,_sba in zip(m, sab, sba)]
    return torch.stack([torch.jit.wait(fut) for fut in futs]).sum()
    #return torch.tensor([_count_blocking_pairs(_m,_sab,_sba) for _m,_sab,_sba in zip(m, sab, sba)], dtype=torch.float32, device=m.device)


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
        cba = to_cost(mat=cba, pformat=pformat, dim=-2)
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
