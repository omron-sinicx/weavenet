import torch
from torch import nn

from ..model import TrainableMatchingModule, MatchingModule, Unit, UnitListGenerator, ExclusiveElementsOfUnit, UnitProcOrder
from .layers import *
from ..layers import CrossConcat
from typing import List, Tuple, Optional

class TrainableMatchingModuleSp(TrainableMatchingModule):
    r""" A variant of :class:`TrainableMatchingModule <weavenet.weavenet.TrainableMatchingModule>` that treats sparse bipartite graph.

    Args:
        net: a sparse GNN net that estimate matching.
        mask_estimator: an algorithm to estimate mask from input of :func:`forward` or from the output of **pre_net**.
        output_channels: see :class:`TrainableMatchingModule <weavenet.weavenet.TrainableMatchingModule>`.
        pre_interactor: :class:`TrainableMatchingModule <weavenet.weavenet.TrainableMatchingModule>`.
        pre_net: a GNN pre_net applied before **mask_estimator**.
        stream_aggregator: the aggregator that merges estimation from all the streams (Default: :class:`DualSoftMaxSp <weavenet.sparse.layers.DualSoftmaxSqrtSp>`)            
        
    **Example of Usage (1) Use WeaveNet to solve stable matching or any other combinatorial optimization**::
    
        from weavenet import TrainableMatchingModule, WeaveNet
        
        weave_net_sp = TrainableMatchingModule(
            net = WeaveNetSp(2*32, [64]*3, [32]*3),
            pre_net = WeaveNet(2, [64]*3,  [32]*3),
        )

        for xab, xba_t in batches:
            y_pred = weave_net_sp(xab, xba_t)
            loss = calc_loss(y_pred, y_true)
            ...
        

    **Example of Usage (2) Use WeaveNet for matching extracted features **::
    
        from sparse.weavenet import TrainableMatchingModuleSp, WeaveNetSp
        from weavenet import WeaveNet
        from layers import CrossConcatVertexFeatures
        
        weave_net_sp = TrainableMatchingModule(
            net = WeaveNetSp(2*32, [64]*3, [32]*3),
            pre_net = WeaveNet(2*vfeature_channels+2, [64]*3,  [32]*3),
            pre_interactor = CrossConcatVertexFeatures(compute_similarity_cosine, softmax)
        )

        for xa, xb, y_true in batches:
            xa = feature_extractor(xa)
            xb = feature_extractor(xb)
            y_pred = weave_net_sp(xa, xb)
            loss = calc_loss(y_pred, y_true)
            ...

    """
    def __init__(self,
                 net_sparse: nn.Module, # typically MatchingModuleSparse
                 pre_interactor_sparse: Optional[CrossConcat] = CrossConcat(),
                 mask_estimator:nn.Module=LinearMaskInferenceOr(tau=10.0, drop_rate=0.3),
                 output_channels:int=1,
                 net_dense: Optional[nn.Module]=None, # typically MatchingModule
                 pre_interactor_dense:Optional[CrossConcat] = None, # typically CrossConcat
                 stream_aggregator:Optional[StreamAggregatorSp] = DualSoftmaxSqrtSp()):
        super().__init__(net_dense, output_channels, pre_interactor_dense, stream_aggregator)
        self.net_sparse = self.net_sparse
        self.pre_interactor_sparse = self.pre_interactor_sparse
        
        self.mask_estimator = mask_estimator            
        if hasattr(self.mask_estimator, "build"):
            self.mask_estimator.build(pre_net.output_channels, output_channels)
        
    def _forward_sp(self, xab:torch.Tensor, xba_t:torch.Tensor, 
                    src_vertex_id:torch.Tensor, tar_vertex_id:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        
        # first interaction
        if self.pre_interactor is not None:
            xab, xba_t = self.pre_interactor_sparse(xab, xba_t)
            
        # net
        xab, xba_t = self.net_sparse.forward(xab, xba_t,src_vertex_id, tar_verted_id)
        return xab, xba_t
        
    def _forward_wrapup(self, xab:torch.Tensor, xba_t:torch.Tensor, 
               src_vertex_id:torch.Tensor, tar_vertex_id:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # wrap up into logits
        xab_fut = torch.jit.fork(self.last_layer(xab))
        xba_t = self.last_layer(xba_t)
        
        # aggregate two streams while applying logistic regression.
        return self.stream_aggregator(torch.jit.wait(xab_fut), src_vertex_id, tar_vertex_id, xba=xba_t)

    def forward(self, xab:torch.Tensor, xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r""" Try to match a bipartite agents on side `a` and `b`.
                
        Shape:
           - xab: :math:`(\ldots, N, M, C)`
           - xba_t: :math:`(\ldots, N, M, C)`
           - output:  :math:`(\ldots, N, M, \text{output_channels}')` if **dim_a** = -3 and **dim_b** = -2. :math:`C' = 2*C` if **compute_similarity** is None,  :math:`C' = 2*C+1` if only **compute_similarity** is set, and :math:`C' = 2*C+2`if both **compute_similarity** and  **directional_normalization** are set.
           
        Args:
           xab: vertex features on the side `a` or edge features directed from the side `a` to `b`.
           
           xba_t: vertex features on the side `b` or edge features directed from the side `b` to `a`.           
           
        Returns:
           A resultant tensor aggregated by stream_aggregator after processed through the network.
        """      
        if self.net is not None:
            super()._forward(xab, xba_t)
        
        
        # edge pruning
        mask = self.mask_estimator(xab, xba_t)

        sd_adaptor = SparseDenseAdaptor(mask)
        xab = sd_adaptor.to_sparse(xab*mask)
        xba_t = sd_adaptor.to_sparse(xba_t*mask)
        
        
        xab, xba_t = self._forward_sp(
            xab, 
            xba_t, 
            sd_adaptor.src_vertex_id,
            sd_adaptor.tar_vertex_id,
        )
        
        m, mab, mba_t = self._forward_wrapup(xab, xba_t, sd_adaptor.src_vertex_id, sd_adaptor.tar_vertex_id)
        m_fut = torch.jit.fork(sd_adaptor.to_dense, m)
        mab_fut = torch.jit.fork(sd_adaptor.to_dense, mab)
        mba_t_fut =  torch.jit.fork(sd_adaptor.to_dense, mba_t)
        
        if self.train:            
            m = torch.jit.wait(m_fut)
            m[m==0] = mask[m==0] # backward path for negative edges. Not average for negative samples to deny clearly negative edges individually.
            mab = torch.jit.wait(mab_fut)
            mab[mab==0] = mask[mab==0]
            mba_t = torch.jit.wait(mba_t_fut)
            mba_t[mba_t==0] = mask[mba_t==0]
        else:
            m = torch.jit.wait(m_fut)
            mab = torch.jit.wait(mab_fut)
            mba_t = torch.jit.wait(mba_t_fut)

        return m, mab, mba_t


class MatchingModuleSp(MatchingModule):
    r""" A net of matching module. This controlls the way of interaction at each end of unit-process and residual paths. See :class:`MatchingModule <weavenet.weavenet.MatchingModule>` for initialization.
    
     """                
    def forward(self,
                xab:torch.Tensor, 
                xba_t:torch.Tensor,
                src_vertex_id:torch.Tensor,
                tar_vertex_id:torch.Tensor,
               )->Tuple[torch.Tensor, torch.Tensor]:
        r""" Applies a series of unit process with a two-stream manner.
                
        Shape:
           - xab: :math:`(\text{num_edges_in_batch}, C)`
           - xba_t: :math:`(\text{num_edges_in_batch}, C)`
           - src_vertex_id: `(\text{num_edges_in_batch})`
           - tar_vertex_id: `(\text{num_edges_in_batch})`
           - output: a pair of tensors with the shape :math:`(num_edges_in_batch, C')`
           
        Args:
           xab: edge features directed from the side `a` to `b`.           
           xba_t: edge features directed from the side `b` to `a`.           
           
        Returns:
           A pair of processed features.
        """        
        xab_keep, xba_t_keep = xab, xba_t # None is OK, but xab/xba_t for jit script
        for l, (unit0, unit1) in enumerate(zip(self.stream0, self.stream1)):
            calc_res = self.calc_residual[l]
            xab_fut = torch.jit.fork(unit0,xab, src_vertex_id)
            xba_t = unit1(xba_t, tar_vertex_id)
            xab = torch.jit.wait(xab_fut)
            
            if self.use_residual:              
                if l==self.keep_first_var_after:
                    # keep values after the directed unit's process.
                    xab_keep = xab
                    xba_t_keep = xba_t
                if calc_res:
                    xab_keep, xab = xab, xab + xab_keep
                    xba_t_keep, xba = xba_t, xba_t + xba_t_keep
            
            xab, xba_t = self.interactor(xab, xba_t)            
        
        return xab, xba_t
    
    def _forward_single_stream(self, 
                              xab:torch.Tensor, 
                              xba_t:torch.Tensor,
                              src_vertex_id:torch.Tensor,
                              tar_vertex_id:torch.Tensor,
                             )->Tuple[torch.Tensor, torch.Tensor]:
        r""" Applies a series of unit process with a single-stream manner. This function is set as self.forward if interactor is `None` at :func:`__init__`.
        """

        x_keep = xab
        for l, unit in enumerate(self.stream):
            calc_res = self.calc_residual[l]
            if l%2==0:
                vid = src_vertex_id
            else:
                vid = tar_vertex_id
            xab = unit(xab, vid)
            
            if self.use_residual:              
                if i==self.keep_first_var_after:
                    # keep values after the directed unit's process.
                    xab_keep = xab
                if calc_res:
                    xab_keep, xab = xab, xab + xab_keep            
        return xab, xab
    
class UnitSp(Unit):
    r""" a sparse version of :class:`Unit <weavenet.weavenet.Unit>`.
    
        Args:
           encoder: a trainable unit
           order: a direction of process order. ['ena'|'nae'|'ean'|'ane'], e: encoder, a: activator, n: normalizer. e.g.) 'ena' applies encoder->normalizer->activator.
           normalizer: a normalizer, such as :class:`nn.BatchNorm1d`.
           activator: an activation function, such as :obj:`nn.PReLU`.
           
        """            
    
    def __init__(self, 
                 encoder:nn.Module, 
                 order: UnitProcOrder,
                 normalizer:Optional[nn.Module]=None, 
                 activator:Optional[nn.Module]=None):
        super().__init__(encoder, order, normalizer, activator)
        self.forward = eval("self._forward_{}".format(order))
        
    def _forward_ena(self, x:torch.Tensor, vertex_id:torch.Tensor)->torch.Tensor:
        x = self.encoder(x, vertex_id)
        if self.normalizer is not None:
            x = self.normalizer(x)
        if self.activator is not None:
            x = self.activator(x)
        return x
    def _forward_nae(self, x:torch.Tensor, vertex_id:torch.Tensor)->torch.Tensor:
        if self.normalizer is not None:
            x = self.normalizer(x)
        if self.activator is not None:
            x = self.activator(x)
        x = self.encoder(x, vertex_id)
        return x
    
    def _forward_ean(self, x:torch.Tensor, vertex_id:torch.Tensor)->torch.Tensor:
        x = self.encoder(x, vertex_idr)
        if self.activator is not None:
            x = self.activator(x)
        if self.normalizer is not None:
            x = self.normalizer(x)
        return x
    
    def _forward_ane(self, x:torch.Tensor, vertex_id:torch.Tensor)->torch.Tensor:
        if self.activator is not None:
            x = self.activator(x)
        if self.normalizer is not None:
            x = self.normalizer(x)
        x = self.encoder(x, vertex_id)
        return x
    
    def forward(self, x:torch.Tensor, vertex_id:torch.Tensor)->torch.Tensor:
        r""" Applies unit process. This function is replaced to any of Unit._forward_* functions in :func:`__init__` based on the argument **order**.
                
        Shape:
           - x: :math:`(\text{num_edges_in_batch}, C)`
           - vertex_id: :math:`(\text{num_edges_in_batch})`
           
        Args:
           x: a edge features, which are flatten through a batch.
           vertex_id: ID list of each edge (ID identifies src or target vetex of each edge).
           
        Returns:
           A processed features.
        """        
        raise RuntimeError("This function should never called since replaced to other function at initialization.")        

    
    
class WeaveNetUnitListGeneratorSp(UnitListGenerator):
    r""" Sparse version of :class:`WeaveNetUnitListGenerator <weavenet.weavenet.WeaveNetUnitListGenerator>`
    
        Args:
           input_channels: input_channels for the first unit.
           mid_channels_list: mid_channels for each point-net-based set encoders.
           output_channels_list: output_channels for the units. 
        """            
    def __init__(self,
                 input_channels:int,
                 mid_channels_list:List[int],
                 output_channels_list:List[int],
            ):
        self.mid_channels_list = mid_channels_list
        super().__init__(input_channels, output_channels_list)
        assert(len(output_channels_list) == len(mid_channels_list))               

        
    def _build(self, in_channels_list:List[int]):
        return [
            UnitSp(
                SetEncoderPointNetSp(in_ch, mid_ch, out_ch),
                'ena',
                nn.BatchNorm1d(out_ch),
                nn.PReLU(),)
            for in_ch, mid_ch, out_ch in zip(in_channels_list, self.mid_channels_list, self.output_channels_list)
        ]
        
class ExperimentalUnitListGeneratorSp(UnitListGenerator):
    r""" Sparse version of :class:`ExperimentalUnitListGenerator <weavenet.weavenet.ExperimentalUnitListGenerator>`
    
        Args:
           input_channels: input_channels for the first unit.
           mid_channels_list: mid_channels for each point-net-based set encoders.
           output_channels_list: output_channels for the units. 
    """            
    
    class Encoder(SetEncoderBaseSp):
        def __init__(self, in_channels:int, mid_channels:int, output_channels:int, **kwargs):
            r"""        
            Args:
                in_channels: the number of input channels.
                mid_channels: the number of output channels at the first convolution.
                output_channels: the number of output channels at the second convolution.

            """ 
            first_process = nn.Linear(in_channels, mid_channels)
            second_process = nn.Linear(in_channels + mid_channels, output_channels, bias=False)    

            super().__init__(
                first_process, 
                MaxPoolingAggregatorSp(),
                DifferenceConcatMerger(dim_feature=-1),
                second_process,
                **kwargs,
            )
    def _build(self, in_channels_list:List[int]):
        return [
            UnitSp(
                self.Encoder(in_ch, mid_ch, out_ch),
                'ena',
                nn.BatchNorm1d(out_ch),
                nn.PReLU(),)
            for in_ch, mid_ch, out_ch in zip(in_channels_list, self.mid_channels_list, self.out_channels_list)
        ]    
        
class WeaveNetSp(MatchingModuleSp):
    r""" Sparse version of :class:`WeaveNet <weavenet.weavenet.WeaveNet>`

        Args:
            input_channels: input_channels for the first unit (see :class:`WeaveNetUnitListGenerator <weavenet.weavenet.WeaveNetUnitListGenerator>`).
            output_channels_list: output_channels for the units (see :class:`WeaveNetUnitListGenerator <weavenet.weavenet.WeaveNetUnitListGenerator>`). 
            mid_channels_list: mid_channels for each point-net-based set encoders (see :class:`WeaveNetUnitListGenerator <weavenet.weavenet.WeaveNetUnitListGenerator>`).
            calc_residual: see :class:`MatchingModule <weavenet.weavenet.MatchingModule>`
            keep_first_var_after: see :class:`MatchingModule <weavenet.weavenet.MatchingModule>`
            exclusive_elements_of_unit: see :class:`MatchingModule <weavenet.weavenet.MatchingModule>`
            is_single_stream: see :class:`MatchingModule <weavenet.weavenet.MatchingModule>`

    """
    def __init__(self,
                 input_channels:int,
                 output_channels_list:List[int],
                 mid_channels_list:List[int],
                 calc_residual:Optional[List[bool]]=None,
                 keep_first_var_after:int=0,
                 exclusive_elements_of_unit:ExclusiveElementsOfUnit='none',
                 is_single_stream:bool=False,
                ):
        if is_single_stream:
            interactor = None
        else:
            interactor = CrossConcat()
            
        super().__init__(
            WeaveNetUnitListGeneratorSp(input_channels, mid_channels_list, output_channels_list),
            interactor = interactor,            
            calc_residual = calc_residual,
            keep_first_var_after = keep_first_var_after,
            exclusive_elements_of_unit = exclusive_elements_of_unit,
        )        
        
class ExperimentalSp(MatchingModuleSp):
    r""" Sparse version of :obj:`Experimental <weavenet.weavenet.Experimental>`

        Args:
            input_channels: input_channels for the first unit (see :class:`ExperimentalUnitListGenerator <weavenet.weavenet.ExperimentalUnitListGenerator>`).
            output_channels_list: output_channels for the units (see :class:`ExperimentalUnitListGenerator <weavenet.weavenet.ExperimentalUnitListGenerator>`). 
            mid_channels_list: mid_channels for each point-net-based set encoders (see :class:`ExperimentalUnitListGenerator <weavenet.weavenet.ExperimentalUnitListGenerator>`).
            calc_residual: see :class:`MatchingModule <weavenet.weavenet.MatchingModule>`
            keep_first_var_after: see :class:`MatchingModule <weavenet.weavenet.MatchingModule>`
            exclusive_elements_of_unit: see :class:`MatchingModule <weavenet.weavenet.MatchingModule>`
            is_single_stream: see :class:`MatchingModule <weavenet.weavenet.MatchingModule>`

    """
    def __init__(self,
                 input_channels:int,
                 output_channels:List[int],
                 mid_channels:List[int],
                 calc_residual:Optional[List[bool]]=None,
                 keep_first_var_after:int=0,
                 exclusive_elements_of_unit:ExclusiveElementsOfUnit='none',
                ):
        super().__init__(
            ExperimentalUnitListGeneratorSp(input_channels,  mid_channels_list, output_channels_list),
            interactor = CrossConcat(),            
            calc_residual = calc_residual,
            keep_first_var_after = keep_first_var_after,
            exclusive_elements_of_unit = exclusive_elements_of_unit,
        )        
             