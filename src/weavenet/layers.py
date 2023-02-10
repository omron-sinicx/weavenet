import torch
import torch.nn as nn

from typing import Optional, Tuple, List, Callable


from torch.nn.modules.batchnorm import _BatchNorm

__all__ = [
    'SetEncoderBase',
    'SetEncoderPointNet',
    'SetEncoderPointNetCrossDirectional',
    'SetEncoderPointNetTotalDirectional',
    'Interactor',
    'CrossConcat',
    'CrossConcatVertexFeatures',
    'CrossDifferenceConcat',
    'MaxPoolingAggregator',
    'StreamAggregator',
    'DualSoftmax',
    'DualSoftmaxSqrt',
    'DualSoftmaxFuzzyLogicAnd',
    'BatchNormXXC',
    'compute_cosine_similarity',
]

@torch.jit.script
def compute_cosine_similarity(xa: torch.Tensor, xb:torch.Tensor)->torch.Tensor:
    r"""
    Shapes:
        - xa: :math:`(B, N, C)`
        - xb: :math: `(B, M, C)`
        - output: :math: `(B, N, M)`
    """
    return torch.einsum("nlc,nsc->nls", xa, xb)

class BatchNormXXC(nn.Module):
    r"""
    
    Applies :class:`BatchNorm1d` to :math:`(\ldots, C)`-shaped tensors. This module is prepered since :class:`nn.BatchNorm2d` assumes the input format of :math:`(B, C, H, W)` but if kernel size is 1, :class:`nn.Conv2d` to :math:`(B, C, H, W)` is slower than :class:`nn.Linear` to :math:`(B, H, W, C)`, which is our case for bipartite-graph edge embedding of :math:`(B, N, M, C)`. In addition, :class:`nn.Linear` works well with sparse cases.
    
    
    **Example of Usage**::
    
        # assume batch_size=8, the problem instance size is 5x5,  and each edge feature is 32 channels.
        B, N, M, C = 8, 5, 5, 32        
        linear = nn.Linear(32, 64)
        bn = BatchNormXXC(64)
        x = torch.rand((B, N, M, C), dtype=torch.float) # prepare a random input.
        x = linear(x)
        x = bn(x)
    
    """
    
    def __init__(self, C)->None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=C)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        r"""
        Shape:
           - x: :math:`(\ldots, C)`
           - output:  :math:`(\ldots, C)`

        Args:
           x: target variable.

        Returns:
           x: batch-normed variable.

        """
        shape = x.shape
        x = self.bn(x.view(-1,shape[-1]))
        return x.view(shape)
        
class Interactor(nn.Module):
    r"""
    
    Abstract :class:`CrossConcat` and any other interactor between feature blocks of two stream architecture. It must have a function :func:`output_channels` to report its resultant feature's output channels (estimated based on the **input_channels**).    
    
    """    
    def output_channels(self, input_channels:int)->torch.Tensor:        
        r"""
        Args:
           input_channels: assumed input channels.
           
        Returns:
           output_channels: **input_channels** (dummy)
        """
        return input_channels
    def forward(self, 
                xab: torch.Tensor, 
                xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("Interactor is an abstract class and should not call its forward function.")
    
class CrossConcat(Interactor):
    r"""
    
    Applies cross-concatenation introduced in `Edge-Selective Feature Weaving for Point Cloud Matching <https://arxiv.org/abs/2202.02149>`_
    
    .. math::
        \text{CrossConcat}([x^{ab}, {x^{ba}}^\top]) = [\text{cat}([x^{ab}, {x^{ba}}^\top], dim=-1), \text{cat}([{x^{ba}}^\top,x^{ab}], dim=-1)]
        
    
    **Example of Usage**::
    
        # prepare an instance of this class.
        interactor = CrossConcat()
        
        # assume batch_size=8, the problem instance size is 6x5,  and each edge feature is 32 channels.
        B, N, M, C = 8, 6, 5, 32        
        
        xab = torch.rand((B, N, M, C), dtype=torch.float) # prepare a random input. # NxM
        xba = torch.rand((B, M, N, C), dtype=torch.float) # prepare a random input. # MxN
        xba_t = xba.transpose(1,2)
        zab, zba_t = interactor(xab, xba_t)
        assert(xab.size(-1)*2 == 2*C)
        assert(xba_t.size(-1)*2 == 2*C)
        assert(interactor.output_channels(C)==2*C)

    """
    def __init__(self, dim_feature:int=-1):
        super().__init__()
        self.dim_feature = dim_feature
    
    def output_channels(self, input_channels:int)->torch.Tensor:
        r"""
        Args:
           input_channels: assumed input channels.
           
        Returns:
           output_channels: :math:`2*` **input_channels**
        """
        return input_channels*2
        
    def forward(self, 
                xab: torch.Tensor, 
                xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Shape:
           - xab: :math:`(\ldots, C)`
           - xba_t: :math:`(\ldots, C)`
           - output(zab, zba_t):  :math:`[(\ldots, 2*C),(\ldots, 2*C)]`

        Args:
           xab: batched feature map, typically with the size of (B, N, M, C) where ij-th feature at :math:`(i, j)\in N \times M` represent edges from side `a` to `b`.
           
           xba_t: batched feature map with the same shape with xab, and represent edges from side `b` to `a`.

        Returns:
           (zab, zba_t) calculated as :math:`\text{CrossConcat}([x^{ab}, {x^{ba}}^\top])` .

        """
        zab_fut = torch.jit.fork(torch.cat, [xab,xba_t],dim=self.dim_feature)
        zba_t = torch.cat([xba_t,xab],dim=self.dim_feature)
        zab = torch.jit.wait(zab_fut)
        return (zab, zba_t)


class CrossDifferenceConcat(Interactor):
    r"""
    
    Applies cross-concatenation of mean and difference (experimental). 
    
    .. math::
        \text{CrossDiffConcat}([x^{ab}, {x^{ba}}^\top]) = [\text{cat}([x^{ab}, {x^{ba}}^\top - x^{ab}], \text{dim}=-1), \text{cat}([{x^{ba}}^\top,x^{ab}-{x^{ba}}^\top], \text{dim}=-1)]
        
    .. note::
        This class was not very effective with stable matching test.
    """
    def __init__(self, dim_feature:int=-1):
        super().__init__()
        self.dim_feature = dim_feature
    def output_channels(self, input_channels:int)->torch.Tensor:
        r"""
        Args:
           input_channels: assumed input channels.
           
        Returns:
           output_channels: :math:`2*` **input_channels**
        """
        return input_channels * 2
        
    def forward(self, 
                xab: torch.Tensor, 
                xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Shape:
           - xab: :math:`(\ldots, C)`
           - xba_t: :math:`(\ldots, C)`
           - output(zab, zba_t):  :math:`[(\ldots, 2*C),(\ldots, 2*C)]`

        Args:
           xab: batched feature map, typically with the size of :math:`(B, N, M, C)` where :math:`ij`-th feature at :math:`(i, j)\in N \times M` represent edges from side `a` to `b`.
           
           xba_t: batched feature map with the same shape with **xab**, and represent edges from side `b` to `a`.

        Returns:
           **Tuple[zab, zba_t]** calculated as :math:`\text{CrossDiffConcat}([x^{ab}, {x^{ba}}^\top])` .

        """
        #merged = xab.min(xba_t)
        zab_fut = torch.jit.fork(torch.cat, [xab, xab - xba_t],dim=self.dim_feature)
        zba_t = torch.cat([xba_t, xba_t - xab],dim=self.dim_feature)
        zab = torch.jit.wait(zab_fut)
        return (zab, zba_t)
    

class MaxPoolingAggregator(nn.Module):
    r""" Aggregates edge features for each vertex, and virtually reshape it to have the same size (other chann :math:`C`) for merger.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x:torch.Tensor, dim_target:int)->torch.Tensor:
        r"""
        Shape:
           - x: :math:`(\ldots, N, M, C)` 
           - output:  :math:`(\ldots, N, M, C)`　 

        Args:
           x: an input tensor.
           dim_target: typically -2 or -3.

        Returns:
           **x_aggregated**

        """        
        return x.max(dim=dim_target, keepdim=True)[0]
    
class SetEncoderBase(nn.Module):
    r"""Applies abstracted set-encoding process.
    
    .. math::
        \text{SetEncoderBase}(x) = \text{second_process}(\text{merger}(x, \text{aggregator}(\text{first_process}(x))))
        
    Args:
       first_process: a callable (and typically trainable) object that converts a :math:`(B, N, M, C_{input})` tensor  to :math:`(B, N, M, C_{mid})`.
       aggregator: a callable object that aggregate :math:`M` edge features for each of :math:`N` vertices. The resultant tensor is reformatted into the shape of :math:`(B, N, M, C_{mid})` tensor.
       merger: a callable object that merge  :math:`(B, N, M, C_{input})` edge features and  :math:`(B, N, M, C_{mid})` vertex features into  :math:`(B, N, M, C_{merged})`.
       second_process: a callable (and typically trainable) object that converts a :math:`(B, N, M, C_{merged})` tensor  to :math:`(B, N, M, C_{output})`.    
       
    """
    
    def __init__(self, 
                 first_process: Callable[[torch.Tensor], torch.Tensor], 
                 aggregator: Callable[[torch.Tensor, int], torch.Tensor], 
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
                dim_target:int)->torch.Tensor:
        r"""
        Shape:
           - x: :math:`(\ldots, N, M, \text{in_channels})`
           - output:  :math:`(\ldots, N, M, \text{output_channels})`　

        Args:
           x: an input tensor.
           dim_target: the dimension of edge's target vertex.           

        Returns:
           z_edge_features

        """        
        z_edge_fut = torch.jit.fork(self.second_process_edge,x)
        z = self.first_process(x)
        z_vertex = self.second_process_vertex(self.aggregator(z, dim_target))
        return torch.jit.wait(z_edge_fut) + z_vertex
    
        """
        z = self.first_process(x)
        z_vertex = self.aggregator(z, dim_target)
        z = self.merger(x, z_vertex)
        return self.second_process(z)
        """
        
        '''
        if not self.return_vertex_feature:
            return z
        return z, z_vertex
        '''

class SetEncoderPointNet(SetEncoderBase):
    r""" Applies a process proposed in `DeepSet <https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html>`_ and `PointNet <https://github.com/charlesq34/pointnet>`_ to a set of out-edge features of each vertex. See :class:`SetEncoderBase` for its forwarding process.
    
    Args:
        in_channels: the number of input channels.
        mid_channels: the number of output channels at the first convolution.
        output_channels: the number of output channels at the second convolution.           
    """ 
    def __init__(self, in_channels:int, mid_channels:int, output_channels:int, **kwargs):
        first_process = nn.Linear(in_channels, mid_channels)
        second_process_edge = nn.Linear(in_channels, output_channels, bias=False)    
        second_process_vertex = nn.Linear(mid_channels, output_channels, bias = False)
        
        super().__init__(
            first_process, 
            MaxPoolingAggregator(),
            second_process_edge,
            second_process_vertex,
            **kwargs,
        )
        
    
class SetEncoderPointNetCrossDirectional(SetEncoderPointNet):
    r"""Applies a variation of :class:`SetEncoderPointNet`. This class max-pools in **dim_src** direction in addition to **dim_target** direction of standard SetEncoder. 
    
    .. note:
        This seems effective since it enhances the interaction between two side more frequently.

    Args:
        in_channels: the number of input channels.
        mid_channels: the number of output channels at the first convolution.
        output_channels: the number of output channels at the second convolution.           
    """ 

    def forward(self, 
                x:torch.Tensor,
                dim_target:int)->torch.Tensor:
        r"""
        **Shape and Args**: same as :class:`SetEncoderPointNet`

        Args:
           x: an input tensor of edge features.
           dim_target: the dimension of edge's target vertex and **must be -3 or -2**.

        Returns:
           **z_edge_features**
        """
        
        z_edge_fut = torch.jit.fork(self.second_process_edge,x)
        z = self.first_process(x)
        if dim_target==-2:
            dim_src = -3
        elif dim_target==-3:
            dim_src = -2
        else:
            raise RuntimeError("Unexpected dim_tar: {}. It must be -3 or -2.".format(dim_target))
        
        z_src_vertex_fut = torch.jit.fork(self.second_process_vertex, self.aggregator(z, dim_src))
        z_tar_vertex = self.second_process_vertex(self.aggregator(z, dim_target))
        return torch.jit.wait(z_edge_fut) + torch.jit.wait(z_src_vertex_fut) + z_tar_vertex

        
class SetEncoderPointNetTotalDirectional(SetEncoderBase):
    r"""Applies a variation of :class:`SetEncoderPointNet`. This class max-pools all the edge features in addition to :class:`SetEncoderPointNetCrossDirectional`, and concatenate the summerized features to original edge feature. 
    
    .. note:
        Experimentally, this seems not very effective.
    
    Args:
        in_channels: the number of input channels.
        mid_channels: the number of output channels at the first convolution.
        output_channels: the number of output channels at the second convolution.

    """ 

    def forward(self, 
                x:torch.Tensor,
                dim_target:int)->torch.Tensor:
        r"""
        Shape:
           - x: :math:`(\ldots)` (not defined with this abstractive class)
           - output:  :math:`(\ldots)`　 (not defined with this abstractive class)

        Args:
           x: an input tensor.

        Returns:
           z_edge_features, z_vertex_features

        """        
        z_edge_fut = torch.jit.fork(self.second_process_edge(x))
        z = self.first_process(x)
        if dim_target==-2:
            dim_src = -3
        elif dim_target==-3:
            dim_src = -2
        else:
            raise RuntimeError("Unexpected dim_tar: {}. It must be -3 or -2.".format(dim_target))
        z_vertex_src = self.aggregator(z, dim_src)
        z_src_vertex_fut = torch.jit.fork(self.second_process_vertex, z_vertex_src)
        z_vertex_tar = self.aggregator(z, dim_tar)
        z_tar_vertex_fut = torch.jit.fork(self.second_process_vertex, z_vertex_tar)
        z_vertex_all = self.aggregator(z_vertex_tar, dim_src)
        z_all_vertex =self.second_process_vertex(z_vertex_all)
                
        return torch.jit.wait(z_edge_fut) + torch.jit.wait(z_src_vertex_fut) + torch.jit.wait(z_tar_vertex_fut) + z_all_vertex
        
StreamAggregator = Callable[[torch.Tensor,Optional[torch.Tensor], bool],Tuple[torch.Tensor,torch.Tensor,torch.Tensor]]
class StreamAggregatorTHRU(nn.Module):
    r"""
    
    Applies nothing, but just multiply. 
    
    .. math::
        \text{THRU}(x^{ab}_{ij}, x^{ba}_{ij}) = x^{ab}_{ij} * x^{ba}_{ji}

    
    """
    def set_xba_t(self, 
                  xab:torch.Tensor, 
                  xba_t:Optional[torch.Tensor],
                     )->Tuple[torch.Tensor]:
        if xba_t is None:
            xba_t = xab
        return xba_t
        
        
    def forward(self, xab:torch.Tensor, 
                      xba_t:Optional[torch.Tensor],
                     )->Tuple[torch.Tensor, torch.Tensor]:
        r""" Calculate the dual softmax for batched matrices.
                
        Shape:
           - xab: :math:`(B, \ldots, N, M, C)`
           - xba_t: :math:`(B, \ldots, N, M, C)`
           - output:  :math:`(B, \ldots, N, M, C')`
           
        Args:
           xab: 1st batched matrices.
           
           xba_t: 2nd batched matrices.           
           
        Returns:
           xab*xba_t, xab, xba_t
 
           
        """
        xba_t = self.set_xba_t(xab, xba_t)
        return xab*xba_t, xab, xba_t
    
class DualSoftmax(StreamAggregatorTHRU):
    r"""
    
    Applies the dual-softmax calculation to a batched matrices. DualSoftMax is originally proposed in `LoFTR (CVPR2021) <https://zju3dv.github.io/loftr/>`_. 
    
    .. math::
        \text{DualSoftmax}(x^{ab}_{ij}, x^{ba}_{ij}) = \frac{\exp(x^{ab}_{ij})}{\sum_j \exp(x^{ab}_{ij})} * \frac{\exp(x^{ba}_{ij})}{\sum_i \exp(x^{ba}_{ij})} 

    In original definition, always :math:`x^{ba}=x^{ab}`. This is an extensional implementation that accepts :math:`x^{ba}\neq x^{ab}` to input the two stream outputs of `WeaveNet`. 
    
    """
    def __init__(self, dim_src:int=-3, dim_tar:int=-2)->None:
        super().__init__()
        self.sm_col = nn.Softmax(dim=dim_tar)
        self.sm_row = nn.Softmax(dim=dim_src)
         
    
    def _apply_softmax(self,
                      xab:torch.Tensor, 
                      xba_t:Optional[torch.Tensor],
                     )->Tuple[torch.Tensor, torch.Tensor]:
        zab_fut = torch.jit.fork(self.sm_col, xab)
        
        xba_t = self.set_xba_t(xab, xba_t)
        zba_t = self.sm_row(xba_t)
        zab = torch.jit.wait(zab_fut)
        return zab, zba_t
    
    

    def forward(self, 
                xab:torch.Tensor, 
                xba_t:Optional[torch.Tensor]=None)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        r""" Calculate the dual softmax for batched matrices.
                
        Shape:
           - xab: :math:`(B, \ldots, N, M, C)`
           - xba_t: :math:`(B, \ldots, N, M, C)`
           - output:  :math:`(B, \ldots, N, M, 1)`
           
        Args:
           xab: 1st batched matrices.
           
           xba_t: 2nd batched matrices. 
           
           
        Returns:
           a triplet of **(mab * mba_t)**, **mab** (=softmax(xab, dim=-2)), **mba_t** (=softmax(xba_t, dim=-1)
 
           
        """
        zab, zba_t = self._apply_softmax(xab, xba_t)
        return zab * zba_t, zab, zba_t

class DualSoftmaxSqrt(DualSoftmax):
    r"""
    
    A variation of :class:`DualSoftmax`. This variation is effective when **xab** and **xba** derive from different computational flow since the forward sqrt operation amplify the backward gradient to each flow.
    
    .. math::
        \text{DualSoftmaxSqrt}(x^{ab}_{ij}, x^{ba}_{ij}) = \sqrt{\text{DualSoftmax}(x^{ab}_{ij}, x^{ba}_{ij})}
    
    """    
    def forward(self, 
                xab:torch.Tensor, 
                xba:Optional[torch.Tensor]=None, 
                is_xba_transposed:bool=True)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        r"""
        **Shape and Args**: same as :class:`DualSoftmax`
           
        Args:
           xab: 1st batched matrices.
           
           xba: 2nd batched matrices. If None, **xab** is used as (transposed) **xba**. 
           
           is_xba_transposed: set **False** if :math:`(N_1, M_1)==(N_2, M_2)` and set **True** if :math:`(N_1, M_1)==(M_2, N_2)`. Default: **False**.
           
        Returns:
           a triplet of **(mab * mba_t).sqrt()**, **mab** (=softmax(xab, dim=-2)), **mba_t** (=softmax(xba_t, dim=-1)
        """
        epsilon:float=10**-7
        zab, zba_t = self._apply_softmax(xab, xba, is_xba_transposed)
        return torch.clamp(zab*zba_t, epsilon).sqrt(), zab, zba_t

class DualSoftmaxFuzzyLogicAnd(DualSoftmax):
    r"""
    
    Applies the calculation proposed in `Shira Li, "Deep Learning for Two-Sided Matching Markets" <https://www.math.harvard.edu/media/Li-deep-learning-thesis.pdf>`_.
    
    .. math::
        \text{DualSoftmaxFuzzyLogicAnd}(x^{ab}_{ij}, x^{ba}_{ij}) = \min(\frac{\exp(x^{ab}_{ij})}{\sum_j \exp(x^{ab}_{ij})},  \frac{\exp(x^{ba}_{ij})}{\sum_i \exp(x^{ba}_{ij})})

    
    """
    def forward(self,
                xab:torch.Tensor, 
                xba:Optional[torch.Tensor]=None, 
                is_xba_transposed:bool=True)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        r""" 
                
        **Shape and Args**: same as :class:`DualSoftmax`
           
        Args:
           xab: 1st batched matrices.
           
           xba: 2nd batched matrices. If None, **xab** is used as (transposed) **xba**. 
           
           is_ba_transposed: set **False** if :math:`(N_1, M_1)==(N_2, M_2)` and set **True** if :math:`(N_1, M_1)==(M_2, N_2)`. Default: **False**.
           
        Returns:
           a triplet of **torch.min(mab, mba_t)**, **mab** (=softmax(xab, dim=-2)), **mba_t** (=softmax(xba_t, dim=-1)
           
        """
        zab, zba_t = self._apply_softmax(xab, xba, is_xba_transposed)
        return zab.min(zba_t), zab, zba_t
        
class CrossConcatVertexFeatures(Interactor):
    r""" CrossConcat vertex features for side `a` and `b`, which are typically provided from a feature extractor.

        Args:
           dim_a: the dimension of side `a` in the output tensor.
           dim_b: the dimension of side `b` in the output tensor.
           compute_similarity: an operator to calculate similarity between two vertex features. If set, the output has an additional channels where the similarity is set.
           directional_normalization: an normalizer applied to the axe **dim_a** and **dim_b**. If set, the output has two additional channels where normalized similarities on the two axe are set. If compute_similarity is None, this function will be ignored.
    """
    def __init__(self,
             dim_a = -3,
             dim_b = -2,
             dim_feature = -1,
             compute_similarity:Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
             directional_normalization:Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
            ):
        self.dim_a, self.dim_b, self.dim_feature = dim_a, dim_b, dim_feature
        self.compute_similarity = compute_similarity
        self.directional_normalization = directional_normalization
        
    def forward(self, xa:torch.Tensor, xb:torch.Tensor)->torch.Tensor:
        r""" Concat vertex features on the two sides `a` and `b`, while appending similarities based on the callback functions.
                
        Shape:
           - x_a: :math:`(\ldots, N, C)`
           - x_b: :math:`(\ldots, M, C)`
           - output:  :math:`(\ldots, N, M, C')` if **dim_a** = -3 and **dim_b** = -2. :math:`C' = 2*C` if **compute_similarity** is None,  :math:`C' = 2*C+1` if only **compute_similarity** is set, and :math:`C' = 2*C+2`if both **compute_similarity** and  **directional_normalization** are set.
           
        Args:
           xa: vertex features on the side `a`.
           
           xb: vertex features on the side `b`.           
           
        Returns:
           A pair of edge-wise feature block tensors. Its feature consists of :math:`2*C`-dimensional features (whose :math:`ij`-th feature is a concatenation of :math:` and 1 or 2 dimensional similarities (depending on the callback function setting at :func:`__init__`.  
        """
        xa = xa.unsqueeze(dim = self.dim_b)
        xb = xb.unsqueeze(dim = self.dim_a)        
        shape = xa.shape
        shape[self.dim_b] = xb.size(self.dim_b)        
        
        if self.compute_similarity is None:
            return torch.cat([xa.expand(shape), xb.expand(shape)], dim=self.dim_feature)
        
        similarity_matrix = self.compute_similarity(xa, xb, dim=self.dim_feature)
        shape_sim = shape
        shape_sim[self.dim_feature] = 1
        
        xa = xa.expand(shape)
        xb = xb.expand(shape)
        if self.directional_normalization is None:
            sim = similarity_matrix.view(shape_sim)
            xab = torch.cat([xa, sim, xb], dim=self.dim_feature)
            xba_t = torch.cat([xb, sim, xa], dim=self.dim_feature)
            return xab, xba_t
        
        sim_a = self.directional_normalization(similarity_matrix, dim=self.dim_a).view(shape_sim)
        sim_b = self.directional_normalization(similarity_matrix, dim=self.dim_b).view(shape_sim)
        xab = torch.cat([xa, sim_a, xb, sim_b], dim=self.dim_feature)
        xba_t =  torch.cat([xb, sim_b, xa, sim_a], dim=self.dim_feature)
        
    def output_channels(self, input_channels:int)->int:
        r"""
        Args:
           input_channels: assumed input channels.
           
        Returns:
           output_channels: :math:`2*` **input_channels** if **self.compute_similarity** is None,  :math:`2*` **input_channels** :math:`+1` if only **self.compute_similarity** is set, and :math:`2*` **input_channels** :math:`+2`if both **self.compute_similarity** and  **self.directional_normalization** are set.
        """
        output_channels = 2 * input_channels
        if self.compute_similarity is None:
            return output_channels
        
        if self.directional_normalization is None:
            # output_channels = xa's channels + 1 + xb's channels 
            return output_channels + 1
        
        # output_channels = xa's channels + col-wisely-normalized similarity + xb's channels + row-wisely-normalized similarity 
        return output_channels + 2
    
    

        