# sparse weavenet layers.
import torch
from torch import nn
from typing import Optional, Callable, Tuple

from ..layers import compute_cosine_similarity


# ---------------------------------------------------------------------------
# Native segment ops (replace torch_scatter; see omron-sinicx/weavenet #369/#367).
#
# torch_scatter ships version-locked compiled binaries with no prebuilt wheel for
# recent torch+CUDA combinations, making it an unreproducible hidden dependency.
# `torch.scatter_reduce_` (amax) and `scatter_add_` cover the only two ops the
# sparse path needs, so we drop the external dependency entirely. Both helpers
# operate along dim 0 — the only axis the sparse aggregators use (vertex_id is a
# flat per-edge segment id).
# ---------------------------------------------------------------------------

def _segment_max(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    r"""Per-segment maximum along dim 0 (drop-in for ``scatter_max(...)[0]``).

    Shape:
       - src:   :math:`(E, \ldots)`
       - index: :math:`(E,)` segment id in ``[0, num_segments)``
       - output: :math:`(num\_segments, \ldots)`

    Segments that receive no edge are filled with 0, matching ``scatter_max``'s
    default ``fill_value`` (a non-empty segment can never stay at the ``-inf``
    sentinel, so the fill only ever touches genuinely empty segments).
    """
    num = int(index.max()) + 1 if index.numel() > 0 else 0
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out = src.new_full((num, *src.shape[1:]), float("-inf"))
    out.scatter_reduce_(0, idx, src, reduce="amax", include_self=False)
    return out.masked_fill(out == float("-inf"), 0.0)


def _segment_softmax(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    r"""Per-segment softmax along dim 0 (drop-in for ``scatter_softmax``).

    Numerically stabilised by subtracting the per-segment max before ``exp``,
    then normalising by the per-segment sum — the same scheme ``scatter_softmax``
    uses internally.

    Shape:
       - src:   :math:`(E, \ldots)`
       - index: :math:`(E,)` segment id in ``[0, num_segments)``
       - output: :math:`(E, \ldots)`
    """
    num = int(index.max()) + 1 if index.numel() > 0 else 0
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    seg_max = src.new_full((num, *src.shape[1:]), float("-inf"))
    seg_max.scatter_reduce_(0, idx, src, reduce="amax", include_self=False)
    exp = (src - seg_max[index]).exp()
    seg_sum = exp.new_zeros((num, *exp.shape[1:]))
    seg_sum.scatter_add_(0, idx, exp)
    return exp / seg_sum[index]

@torch.jit.ignore
def _resampling_relaxed_Bernoulli(logits:torch.Tensor, tau:float)->torch.Tensor:
    sampler = torch.distributions.RelaxedBernoulli(tau, logits=logits)  
    return sampler.rsample()



def _gumbel_sigmoid_logits(logits:torch.Tensor,
                          tau:float=1.,
                          hard:bool=False,
                  )->torch.Tensor:
    #sampler = MyRelaxedBernoulli(tau, logits=logits)   
    # y_soft = sampler.rsample()
    y_soft = _resampling_relaxed_Bernoulli(logits, tau)
    if hard:
        # do resampling trick
        y_hard = (y_soft > 0.5).to(logits.dtype)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

def _kthlargest_resampling(x: torch.Tensor, dim:int, tau:float, drop_rate:float)->torch.Tensor:
    # Keep the top-(1-drop_rate) fraction of elements along `dim` after gumbel sigmoid.
    # `kthvalue(K, dim)` returns the K-th smallest, so for "keep N_keep elements"
    # we need the (N - N_keep + 1)-th smallest = (N - N_keep + 1)-th in 1-indexed
    # so that exactly N_keep elements satisfy `>= kth_val`.
    y_soft = _gumbel_sigmoid_logits(x, tau, hard=False)
    N = y_soft.size(dim)
    N_keep = max(int(round(N * (1.0 - drop_rate))), 1)
    # kthvalue is 1-indexed: K_smallest such that exactly K_smallest - 1 are strictly less.
    # For N=100, N_keep=50 → we want kth_val to be the (100-50+1)=51st smallest, so
    # 50 elements >= kth_val (excluding the kth_val itself if all unique).
    # `>= kth_val` is inclusive, so this gives exactly N_keep when ties are sparse.
    K_smallest = max(N - N_keep + 1, 1)
    kth_val = y_soft.kthvalue(K_smallest, dim=dim, keepdim=True)[0]
    y_hard = (y_soft >= kth_val).to(x.dtype)
    return y_hard - y_soft.detach() + y_soft
                 
class MaskSelectorByLinearInferenceOr(nn.Module):
    r"""Selects edges based on linear prediction. The result for each direction is aggregated by OR rule.
    
    Args:
        dim_src: `dim` of source vertex of edges.
        dim_tar: `dim` of target vertex of edges.
        drop_rate: sets drop rate of edges for each vertex. The OR rule selection may results in less drop-rate in actual calculations.
        tau: the temperature of gumbel sigmoid.
    
    """
    def __init__(self,
                 dim_src:int=-3,
                 dim_tar:int=-2,
                 drop_rate:float = 0.5,
                 tau:float = 1.0,
                )->None:
        super().__init__()
        self.tau = tau
        self.dim_src = dim_src
        self.dim_tar = dim_tar
        self.drop_rate = drop_rate
        self.linear: Optional[nn.Linear] = None

    def build(self,
              input_channels:int,
              output_channels:int = 1,)->None:
        r"""Build the linear layer for the prediction. This function is automatically called in :class:`TrainableMatchingModuleSp <weavenet.sparse.weavenet.TrainableMatchingModuleSp>`.

        Args:
            input_channels: the number of input channels.
            output_channels: the number of output channels.

        """
        self.linear = nn.Linear(input_channels, output_channels, bias=True)

        
    def forward(self,
                xab: torch.Tensor,
                xba_t: torch.Tensor,
               )->torch.Tensor:        
        r"""
        Shape:
           - xab: :math:`(B, N, M, C)`
           - xba_t: :math:`(B, N, M, C)`
           - output:  :math:`(B, N, M, C')`, where :math:`C'` is typically 1.
           
        Args:
           xab: batched feature map, typically with the size of (B, N, M, C) where ij-th feature at :math:`(i, j)\in N \times M` represent edges from side `a` to `b`.
           
           xba_t: batched feature map with the same shape with xab, and represent edges from side `b` to `a`.

        Returns:
           - mask, where edges with score 1.0 are selected and 0.0 are dropped.
        """

        assert self.linear is not None, "build() must be called before forward()."
        xab = self.linear.forward(xab)
        xab = _kthlargest_resampling(xab, self.dim_src, self.tau, self.drop_rate)
        xba_t = self.linear.forward(xba_t)
        xba_t = _kthlargest_resampling(xba_t, self.dim_tar, self.tau, self.drop_rate)
        y = xab + xba_t
        y[y==2.0] /= 2
        return y

class MaskSelectorByNorm(nn.Module):
    r"""Selects edges based on linear prediction. The result for each direction is aggregated by OR rule.
    
    Args:
        dim_src: `dim` of source vertex of edges.
        dim_tar: `dim` of target vertex of edges.
        drop_rate: sets drop rate of edges for each vertex. The OR rule selection may results in less drop-rate in actual calculations.
        tau: the temperature of gumbel sigmoid.
    
    """
    def __init__(self,
                 dim_src:int=-3,
                 dim_tar:int=-2,
                 drop_rate:float = 0.5,
                 tau:float = 1.0,
                )->None:
        super().__init__()
        self.tau = tau
        self.dim_src = dim_src
        self.dim_tar = dim_tar
        self.drop_rate = drop_rate
        
        
    def forward(self,
                xab: torch.Tensor,
                xba_t: torch.Tensor,
               )->torch.Tensor:        
        r"""
        Shape:
           - xab: :math:`(B, N, M, C)`
           - xba_t: :math:`(B, N, M, C)`
           - output:  :math:`(B, N, M, C')`, where :math:`C'` is typically 1.
           
        Args:
           xab: batched feature map, typically with the size of (B, N, M, C) where ij-th feature at :math:`(i, j)\in N \times M` represent edges from side `a` to `b`.
           
           xba_t: batched feature map with the same shape with xab, and represent edges from side `b` to `a`.

        Returns:
           - mask, where edges with score 1.0 are selected and 0.0 are dropped.
        """

        xab = xab.abs().norm(p=2,dim=-1, keepdim=True)
        xab = _kthlargest_resampling(xab, self.dim_src, self.tau, self.drop_rate)
        xba_t = xba_t.abs().norm(p=2,dim=-1, keepdim=True)
        xba_t = _kthlargest_resampling(xba_t, self.dim_tar, self.tau, self.drop_rate)
        y = xab + xba_t
        y[y==2.0] /= 2
        return y
    


    
class MaskSelectorBySimilarity(nn.Module):
    r"""Experimental: similarity-based mask selector base class.

    .. warning::
        This class and its subclasses (:class:`MaskSelectorRadiusNeighbor`,
        :class:`MaskSelectorReciprocalNeighbor`) are **experimental** and not
        used by any of the live :class:`TrainableMatchingModuleSp` / :class:`WeaveNetSp`
        paths (which default to :class:`MaskSelectorByLinearInferenceOr`).
        :meth:`get_threshold_by_k` contains undefined-name bugs in the original
        v1.1.0 release (``N``, ``M``, ``max_edge_survive_rate``,
        ``max_suvive_edges_per_sample`` are not in scope) and has never been
        executed. Until the intended semantics are clarified by the original
        author, this class raises :class:`NotImplementedError` on use rather
        than risking incorrect silent behaviour.

    Args:
        max_edge_survive_rate: see original docstring above.
        max_survive_edges_per_sample: see original docstring above.
    """
    def __init__(self,
                 max_edge_survive_rate:float=1.1,
                 max_survive_edges_per_sample:int = -1):
        super().__init__()
        self.max_edge_survive_rate = max_edge_survive_rate
        self.max_survive_edges_per_sample = max_survive_edges_per_sample
        self.set_thresh_by_kthvalue = (0<=max_edge_survive_rate and  max_edge_survive_rate<1.0) or max_survive_edges_per_sample>0

    def get_threshold_by_k(self, sim:torch.Tensor, dim:int=-1, k:int=-1)->torch.Tensor:
        raise NotImplementedError(
            "MaskSelectorBySimilarity.get_threshold_by_k is experimental and "
            "contains undefined-name bugs in v1.1.0. Use MaskSelectorByLinearInferenceOr "
            "or MaskSelectorByNorm instead."
        )

    def wrapup(self, sim:torch.Tensor, sim_selected:torch.Tensor)->torch.Tensor:
        # B7: `self.train` is the method (always truthy); the intended check is
        # `self.training` (the bool flag set by .train()/.eval()).
        if self.training:
            sim = sim - sim.detach() # make the discretized mask differential
        else:
            sim.fill_(0)
        sim[sim_selected] += 1
        return sim


class MaskSelectorRadiusNeighbor(MaskSelectorBySimilarity):
    r"""Experimental — see :class:`MaskSelectorBySimilarity` warning."""
    def __init__(self,
                 radius:float=0.0,
                 max_edge_survive_rate:float=1.1,
                 max_survive_edges_per_sample:int = -1):
        super().__init__(max_edge_survive_rate, max_survive_edges_per_sample)
        self.radius = radius
        assert(radius > 0.0 or self.set_thresh_by_kthvalue)

    # B3: original v1.1.0 forward lacked `self`. Even with `self` restored,
    # the body depends on the broken get_threshold_by_k.
    def forward(self, sim: torch.Tensor)->torch.Tensor:
        raise NotImplementedError(
            "MaskSelectorRadiusNeighbor is experimental — see base class warning."
        )

class MaskSelectorReciprocalNeighbor(MaskSelectorBySimilarity):
    r"""Experimental — see :class:`MaskSelectorBySimilarity` warning."""
    def __init__(self,
                 k:int=0,
                 max_edge_survive_rate:float=0.5,
                 max_survive_edges_per_sample:int = -1):
        # Original v1.1.0 had a typo in the local param name
        # (`max_suvive_edges_per_sample`) that made the super() call
        # raise NameError on construction. Fixed here for parity, but the
        # forward path is still NotImplementedError.
        super().__init__(max_edge_survive_rate, max_survive_edges_per_sample)
        self.k = k
        assert(k > 0 or self.set_thresh_by_kthvalue)

    def forward(self, sim:torch.Tensor)->torch.Tensor:
        raise NotImplementedError(
            "MaskSelectorReciprocalNeighbor is experimental — see base class warning."
        )

class SimilarityBasedMaskInference(nn.Module):
    r"""Inferences mask based on similarity.

        Args:
            compute_similarity: a callable object that calculates a similarity matrix.
            select_mask: a callable object that select a (differentiable) mask based on the similarity. [Default: :class:`MaskSelectorRadiusNeighbor`(0.0,0.5)].
    """    
    def __init__(self,
                 compute_similarity:Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = compute_cosine_similarity,
                 mask_selector:Callable[[torch.Tensor], torch.Tensor] = MaskSelectorRadiusNeighbor(0.0, 0.5),
                ):
        super().__init__()
        self.compute_similarity = compute_similarity
        self.mask_selector = mask_selector
        
    def forward(self,
                xab: torch.Tensor,
                xba_t: torch.Tensor,
               )->torch.Tensor:
        sim = self.compute_similarity(xab, xba_t)
        return self.mask_selector(sim)    
    
class SparseDenseAdaptor():
    r"""Adapt sparse-dense matrix conversion, based on a given **mask**.
    
    Shape:
        - mask: (\ldots, N, M, 1)
    Args:
        mask: a mask that selects edges.        
    
    """
    def __init__(self, mask:torch.Tensor):
        # Mask convention: ``> 0.5`` selects an edge (matches the ST estimator
        # output where ``y_hard - y_soft.detach() + y_soft`` produces forward
        # values of exactly 0 or 1). Using ``> 0.5`` consistently in
        # ``__init__`` and ``to_sparse`` avoids an index/value shape mismatch
        # if the mask ever carries non-binary values (e.g., soft pseudo-mask
        # without ST hardening).
        self.shape = mask.shape[:-1]
        self.N, self.M = self.shape[-2:]
        self.mask_lo = mask.view(-1, self.N, self.M)
        self.indices = torch.nonzero(self.mask_lo > 0.5).t()
        self.src_vertex_id = self.indices[0]*self.N+self.indices[1]
        self.tar_vertex_id = self.indices[0]*self.M+self.indices[2]
        
    def _local_view(self,
                   x:torch.Tensor)->torch.Tensor:
        C = x.size(-1)
        return x.view(-1, self.N, self.M, C)
        
    def to_sparse(self,
                x: torch.Tensor)->torch.Tensor:
        r"""
        Shape:
            - x: (\ldots, N, M, C)
            - output: :math:`(\text{num_of_selected_edges_in_batch}, C)`
        Args:
            x: batched edge features.
            
        Return:
            a flatten edge features, whose elements are selected by *mask*.
        """
        # (\ldots, N, M, C)
        x = self._local_view(x)
        C = x.size(-1)
        values = x[self.mask_lo>0.5].view(-1, C)
        return values
    
    @torch.jit.ignore
    def to_dense(self,
                 x_sparse: torch.Tensor)->torch.Tensor:
        r"""
        Shape:
            - x_sparse: :math:`(\text{num_of_selected_edges_in_batch}, C)`
            - output: :math:`(\dots, N, M, C)`
        Args:
            x_sparse: a flattend edge features.
            
        Return:
            a edge features reformatted in the original shape.
        """
        shape = self.shape + (x_sparse.size(-1),)
        return torch.sparse_coo_tensor(self.indices, x_sparse, shape, device=x_sparse.device, dtype=x_sparse.dtype).to_dense()
        
    

        
class MaxPoolingAggregatorSp(nn.Module):
    def __init__(self):
        r"""A sparse version of :class:`MaxPoolingAggregator <weavenet.sparse.layers.MaxPoolingAggregator>`
        
        Args:
            dim: the axis aggregated in the forward function.
        """
        super().__init__()
        
    def forward(self, x_sp:torch.Tensor, 
                vertex_id:torch.Tensor, 
                dim:int = 0)->torch.Tensor:
        r"""
        Shape:
           - x: :math:`(\ldots, M, D)` if dim = -2, otherwise, the axis directed by dim should have M and aggregated while keeping dims.
           - output:  :math:`(\ldots, 1, D)`　 

        Args:
           x: an input tensor.

        Returns:
           x_aggregated

        """        
        assert dim == 0, "sparse MaxPoolingAggregator only aggregates along dim 0"
        return _segment_max(x_sp, vertex_id)

class SetEncoderBaseSp(nn.Module):
    r"""A sparse version of :class:`SetEncoderBase <weavenet.layers.SetEncoderBase>`
        
    Args:
       first_process: a callable (and typically trainable) object that converts a :math:`(B, N, M, C_{input})` tensor  to :math:`(B, N, M, C_{mid})`.
       aggregator: a callable object that aggregate :math:`M` edge features for each of :math:`N` vertices. The resultant tensor is reformatted into the shape of :math:`(B, N, M, C_{mid})` tensor.
       merger: a callable object that merge  :math:`(B, N, M, C_{input})` edge features and  :math:`(B, N, M, C_{mid})` vertex features into  :math:`(B, N, M, C_{merged})`.
       second_process: a callable (and typically trainable) object that converts a :math:`(B, N, M, C_{merged})` tensor  to :math:`(B, N, M, C_{output})`.    
       
    """
    def __init__(self, 
                 first_process: Callable[[torch.Tensor], torch.Tensor],
                 aggregator: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
                 second_process_edge: Callable[[torch.Tensor], torch.Tensor],
                 second_process_vertex: Callable[[torch.Tensor], torch.Tensor],
                 #return_vertex_feature:bool=False,
                ):
        super().__init__()
        self.first_process = first_process
        self.aggregator = aggregator
        self.second_process_edge = second_process_edge
        self.second_process_vertex = second_process_vertex
        #self.return_vertex_feature = return_vertex_feature
        
    def forward(self, 
                x:torch.Tensor,
                vertex_id: torch.Tensor, # the only difference from dense SetEncoderBase
               )->torch.Tensor:
        r"""Applies set encoding operations.
        
        Shape:
           - x: :math:`(\text{num_of_edges_in_batch}, \text{in_channels})`
           - vertex_id:  :math:`(\text{num_of_edges_in_batch}, )`
           - output:  :math:`(\text{num_of_edges_in_batch}, \text{output_channels})`　

        Args:
           x: an input tensor.
           vertex_id: an index list of vertex id for each edge.

        Returns:
           x_processed

        """
        z_fut = torch.jit.fork(self.second_process_edge, x)
        z = self.first_process(x)
        z_vertex = self.aggregator(z, vertex_id, 0)
        z_vertex = self.second_process_vertex(z_vertex)        
        
        return torch.jit.wait(z_fut) + torch.index_select(z_vertex, 0, vertex_id)

        
class SetEncoderPointNetSp(SetEncoderBaseSp):
    r"""A sparse version of :class:`SetEncoderPointNet <weavenet.layers.SetEncoderPointNet>`

    Args:
        in_channels: the number of input channels.
        mid_channels: the number of output channels at the first convolution.
        out_channels: the number of output channels at the second convolution.

    """ 
    def __init__(self, in_channels:int, mid_channels:int, output_channels:int, **kwargs):
        first_process = nn.Linear(in_channels, mid_channels)
        second_process_edge = nn.Linear(in_channels, output_channels, bias=False)    
        second_process_vertex = nn.Linear(mid_channels, output_channels, bias=False)    
            
        super().__init__(
            first_process, 
            MaxPoolingAggregatorSp(),
            second_process_edge,
            second_process_vertex,
            **kwargs,
        )
        
StreamAggregatorSp = Callable[
    [torch.Tensor,torch.Tensor,torch.Tensor, Optional[torch.Tensor]],
    Tuple[torch.Tensor,torch.Tensor,torch.Tensor]]
class DualSoftmaxSp(nn.Module):
    r"""A sparse version of :class:`DualSoftmax <weavenet.layers.DualSoftmax>`
        
    
    """        
    def apply_softmax(self,
                      xab:torch.Tensor, 
                      src_id:torch.Tensor,
                      tar_id:torch.Tensor,
                      xba:Optional[torch.Tensor]=None,
                     )->Tuple[torch.Tensor, torch.Tensor]:
        if xba is None:
            xba = xab
        zab = _segment_softmax(xab, src_id)
        zba = _segment_softmax(xba, tar_id)
        return zab, zba
    
    

    def forward(self, 
                xab:torch.Tensor, 
                src_id:torch.Tensor,
                tar_id:torch.Tensor,
                xba:Optional[torch.Tensor] = None,
               )->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        r""" Calculate the dual softmax for batched matrices.
                
        Shape:
           - xab: :math:`(\text{num_of_edges_in_batch}, \text{in_channels})`
           - src_id: :math:`(\text{num_of_edges_in_batch}, )`
           - tar_id: :math:`(\text{num_of_edges_in_batch}, )`
           - xba: :math:`(\text{num_of_edges_in_batch}, \text{in_channels})`
           - output:  :math:`(\text{num_of_edges_in_batch}, \text{in_channels})` (all the three outputs has the same shape).
           
        Args:
           xab: 1st batched matrices.
           src_id: an index list of source vertex id for each edge.
           tar_id: an index list of target vertex id for each edge.           
           xba: 2nd batched matrices. If None, **xab** is used as **xba**. 
           
        Returns:
           a triplet of **(mab * mba)**, **mab** (=softmax(xab, dim=-2)), **mba** (=softmax(xba_t, dim=-1)
 
           
        """
        zab, zba = self.apply_softmax(xab, src_id, tar_id, xba=xba)
        return zab * zba, zab, zba

class DualSoftmaxSqrtSp(DualSoftmaxSp):
    r""" A sparse version of :class:`DualSoftmaxSqrt <weavenet.layers.DualSoftmaxSqrt>`        
    
    """
    def forward(self, 
                xab:torch.Tensor, 
                src_id:torch.Tensor,
                tar_id:torch.Tensor,
                xba:Optional[torch.Tensor] = None,
               )->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        r""" 
        
        **Shape and Args**: same as :class:`DualSoftmaxSp`

           
        Args:
           xab: 1st batched matrices.
           src_id: an index list of source vertex id for each edge.
           tar_id: an index list of target vertex id for each edge.           
           xba: 2nd batched matrices. If None, **xab** is used as (transposed) **xba**. This option corresponds to the original implementation of LoFTR's dual softmax.
           
       Returns:
           values (mab * mba_t).sqrt(), mab (=softmax(xab, dim=-2)), mba_t (=softmax(xba_t, dim=-1)
        """
        epsilon:float=10**-7
        zab, zba = self.apply_softmax(xab, src_id, tar_id, xba=xba)
        return torch.clamp(zab*zba, epsilon).sqrt(), zab, zba

class DualSoftmaxFuzzyLogicAndSp(DualSoftmaxSp):
    r"""  A sparse version of :class:`DualSoftmaxFuzzyLogicAnd <weavenet.layers.DualSoftmaxFuzzyLogicAnd>`        
    
    """
    def forward(self,
                xab:torch.Tensor,
                src_id:torch.Tensor,
                tar_id:torch.Tensor,
                xba:Optional[torch.Tensor] = None,
               )->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        r"""

        **Shape and Args**: same as :class:`DualSoftmaxSp`

           
        Args:
           xab: 1st batched matrices.
           src_id: an index list of source vertex id for each edge.
           tar_id: an index list of target vertex id for each edge.           
           xba: 2nd batched matrices. If None, **xab** is used as (transposed) **xba**. This option corresponds to the original implementation of LoFTR's dual softmax.
           
       Returns:
           values torch.min(mab, mba_t), mab (=softmax(xab, dim=-2)), mba_t (=softmax(xba_t, dim=-1)
           
        """
        zab, zba = self.apply_softmax(xab, src_id, tar_id, xba=xba)
        return zab.min(zba), zab, zba
