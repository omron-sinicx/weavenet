import numpy as np
import torch
from enum import IntEnum

class PreferenceFormat(IntEnum):
    r"""An Enum class to categorize the format of preference.
    
    """
    satisfaction = 1 #: assume satisfaction is normalized in range(0.1, 1.0) by the numpy.linspace function. greater is more preferred.
    rank = 2 #: assume rank is shown as integer. smaller is more preferred.
    cost = 3 #: assume cost is shown as interger. smaller is more preffered (same as rank???).


def batch_sum(matching : torch.Tensor, mat : torch.Tensor, batch_size : int)->torch.Tensor:
    r"""summate elements of **mat** filtered by **matching** for each sample in a batch.
    
    Shape:
        - matching: (B, \ldots, N, M)
        - mat: (B, N, M)
        - output: (B, \ldots)
    Args:
        matching: (continuous) matching results
        mat: summation targets
    Return:
        :math:`\sum_{ij} (\text{matching}*\text{mat})` for each sample and channel in a batch   
    
    """    
    dim_diff = matching.dim() - mat.dim()
    if dim_diff > 0:
        shape_mat = list(mat.shape)
        shape_mat = shape_mat[:1] + [1] * dim_diff + shape_mat[1:]
        mat = mat.view(shape_mat)
    assert(matching.dim()==mat.dim())
    
    return (mat * matching).view(matching.shape[:-2] + (-1,)).sum(dim=-1)

    
def sat2rank(sat : torch.Tensor, dim = -1)->torch.Tensor:
    r"""convert satisfaction to rank
    
    Shape:
        - sat: (B, N, M)
        - output: (B, N, M)
    Args:
        sat: preference lists formatted in satisfaction.
        dim: dim of agents' matching targets.
        
    Return:
        preference lists formatted in rank.
    """
    return torch.argsort(sat, dim=dim, descending = True)

def cost2rank(cost : torch.Tensor, dim : int = -1)->torch.Tensor:
    r"""convert cost to rank
    
    Shape:
        - cost: (B, N, M)
        - output: (B, N, M)
    Args:
        cost: preference lists formatted in cost.
        dim: dim of agents' matching targets.
        
    Return:
        preference lists formatted in rank.
    """
    return torch.argsort(cost, dim=dim, descending = False)

def sat2cost(sat : torch.Tensor, dim : int = -1)->torch.Tensor:
    r"""convert satisfaction to cost
    
    Shape:
        - sat: (B, N, M)
        - output: (B, N, M)
    Args:
        sat: preference lists formatted in satisfaction.
        dim: dim of agents' matching targets.
        
    Return:
        preference lists formatted in cost.
    """
    n_opposites = sat.size(dim)
    return torch.round((n_opposites-1)*(1-(sat-0.1)/0.9)) + 1 # invert numpy.linspace

def cost2sat(cost : torch.Tensor, dim : int = -1)->torch.Tensor:
    r"""convert cost to satisfaction
    
    Shape:
        - cost: (B, N, M)
        - output: (B, N, M)
    Args:
        cost: preference lists formatted in cost.
        dim: dim of agents' matching targets.
        
    Return:
        preference lists formatted in satisfaction.
    """
    raise NotImplementedError("cost2sat is not implemented.")
    
def to_cost(mat : torch.Tensor, pformat : PreferenceFormat, dim : int = -1)->torch.Tensor:
    r"""convert **mat** to cost, where **mat**'s format is assumed to be **pformat**.

    Shape:
        - mat: (B, N, M)
        - output: (B, N, M)
        
    Args:
        mat: preference lists.
        pformat: the format of **sat**
        dim: dim of agents' matching targets.
        
    Return:
        preference lists formatted in cost.
    
    """
    if pformat == PreferenceFormat.cost:
        return mat
    if pformat == PreferenceFormat.satisfaction:
        return sat2cost(mat, dim)
    if pformat == PreferenceFormat.rank:
        raise RuntimeError("Impossible conversion: rank to cost")
    raise RuntimeError("Unsupported format")
    
def to_sat(mat : torch.Tensor, pformat : PreferenceFormat, dim : int = -1)->torch.Tensor:
    r"""convert **mat** to satisfaction, where **mat**'s format is assumed to be **pformat**.
    
    Shape:
        - mat: (B, N, M)
        - output: (B, N, M)
    Args:
        mat: preference lists.
        pformat: the format of **sat**
        dim: dim of agents' matching targets.
        
    Return:
        preference lists formatted in satisfaction.
    
    """
    if pformat == PreferenceFormat.cost:
        return cost2sat(mat, dim)
    if pformat == PreferenceFormat.satisfaction:
        return mat
    if pformat == PreferenceFormat.rank:
        raise RuntimeError("Impossible conversion: rank to sat")
    raise RuntimeError("Unsupported format")

def to_rank(mat : torch.Tensor, pformat : PreferenceFormat, dim : int = -1)->torch.Tensor:
    r"""convert **mat** to rank, where **mat**'s format is assumed to be **pformat**.
    
    Shape:
        - mat: (B, N, M)
        - output: (B, N, M)
    Args:
        mat: preference lists.
        pformat: the format of **sat**
        dim: dim of agents' matching targets.
        
    Return:
        preference lists formatted in rank.
    
    """
    if pformat == PreferenceFormat.cost:
        return cost2rank(mat, dim=dim)
    if pformat == PreferenceFormat.satisfaction:
        return sat2rank(mat, dim=dim)
    if pformat == PreferenceFormat.rank:
        return mat
    raise RuntimeError("Unsupported format")
    
