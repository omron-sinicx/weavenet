# sparse weavenet layers.
import torch
from torch import nn
from typing import Optional, Callable, Tuple
from torch_scatter import scatter_max #, scatter_min, scatter_mean
from torch_scatter.composite import scatter_softmax

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
    # select k-th largest elems along to `dim` after gumbel sigmoid.
    y_soft = _gumbel_sigmoid_logits(x, tau, hard=False)
    K = max(int(y_soft.size(dim) * (1.0-drop_rate)), 1)
    kth_val = y_soft.kthvalue(K, dim=dim, keepdim=True)[0]
    y_hard = (y_soft >= kth_val).to(x.dtype)
    return y_hard -y_soft.detach() + y_soft
                 
class LinearMaskInferenceOr(nn.Module):
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
        self.linear = None
        
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
           mask, in which edges with score 1.0 are selected and 0.0 are dropped.
        """

        xab = self.linear.forward(xab)
        xab = _kthlargest_resampling(xab, self.dim_src, self.tau, self.drop_rate)
        xba_t = self.linear.forward(xba_t)
        xba_t = _kthlargest_resampling(xba_t, self.dim_tar, self.tau, self.drop_rate)
        y = xab + xba_t
        y[y==2.0] /= 2
        return y
    

class SparseDenseAdaptor():
    r"""Adapt sparse-dense matrix conversion, based on a given **mask**.
    
    Shape:
        - mask: (\ldots, N, M, 1)
    Args:
        mask: a mask that selects edges.        
    
    """
    def __init__(self, mask:torch.Tensor):
        
        self.shape = mask.shape[:-1]
        self.N, self.M = self.shape[-2:]
        self.mask_lo = mask.view(-1, self.N, self.M)
        self.indices = torch.nonzero(self.mask_lo).t()
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
        x_max, _ = scatter_max(x_sp, vertex_id, dim)
        return x_max

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
                 aggregator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
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
        z_vertex = self.aggregator(z, vertex_id, dim=0)
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
        zab = scatter_softmax(xab, src_id, dim=0)
        zba = scatter_softmax(xba, tar_id, dim=0)
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
                xba:Optional[torch.Tensor]=None, 
                is_xba_transposed:bool=True)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
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

if __name__ == "__main__":
    #_ = WeaveNetOldImplementation(2, 2,1)
    _ = WeaveNet(
            WeaveNetHead6(1,), 2, #input_channel:int,
                 [4,8,16], #out_channels:List[int],
                 [2,4,8], #mid_channels:List[int],1,2,2)
                 calc_residual=[False, False, True],
                 keep_first_var_after = 0,
                 stream_aggregator = DualSoftMaxSqrt())
    