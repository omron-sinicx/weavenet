import torch
from torch import nn

from .preference import to_rank, PreferenceFormat
from .layers import *
from copy import deepcopy

from typing import List, Optional, Tuple
from collections import UserList

try:
    # Literal is available with python >=3.8.0
    from typing import Literal
except:
    # pip install typing_extensions with python < 3.8.0
    from typing_extensions import Literal

class TrainableMatchingModule(nn.Module):
    r""" wrap a GNN head to solve various matching problems.

    Args:
        head: a GNN head that estimate matching.
        output_channels: a number of matching results (Default: 1).
        pre_interactor: the interactor that first merge two input at the forward function (Default: :class:`CrossConcat`).
        stream_aggregator: the aggregator that merges estimation from all the streams (Default: :class:`DualSoftmaxSqrt`)            
        
    **Example of Usage (1) Use WeaveNet to solve stable matching or any other combinatorial optimization**::
    
        from weavenet import TrainableMatchingModule, WeaveNetHead
        
        weave_net = TrainableMatchingModule(
            head = WeaveNetHead(2, [64]*6, [32]*6),
        )

        for xab, xba_t in batches:
            y_pred = weave_net(xab, xba_t)
            loss = calc_loss(y_pred, y_true)
            ...
        

    **Example of Usage (2) Use WeaveNet for matching extracted features **::
    
        from weavenet import TrainableMatchingModule, WeaveNetHead
        from layers import CrossConcatVertexFeatures
        
        weave_net = TrainableMatchingModule(
            head = WeaveNetHead(2*vfeature_channels+2, [64]*6, [32]*6),
            pre_interactor = CrossConcatVertexFeatures(compute_similarity_cosine, softmax)
        )

        for xa, xb, y_true in batches:
            xa = feature_extractor(xa)
            xb = feature_extractor(xb)
            y_pred = weave_net(xa, xb)
            loss = calc_loss(y_pred, y_true)
            ...

    """
    def __init__(self,
                 head:nn.Module,
                 output_channels:int=1,
                 pre_interactor:Optional[CrossConcat] = CrossConcat(),
                 stream_aggregator:Optional[StreamAggregator] = DualSoftmaxSqrt(dim_src=-3, dim_tar=-2)):
        super().__init__()
        self.pre_interactor = pre_interactor
        self.head = head
        self.last_layer = nn.Sequential(
            nn.Linear(head.output_channels, output_channels, bias=False),
            BatchNormXXC(output_channels),
        )
        self.stream_aggregator = stream_aggregator
        
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
        # first interaction
        if self.pre_interactor is not None:
            xab, xba_t = self.pre_interactor(xab, xba_t)
            
        # head
        xab, xba_t = self.head.forward(xab, xba_t)

        # wrap up into logits
        xab = self.last_layer(xab)
        xba_t = self.last_layer(xba_t)
        
        # aggregate two streams while applying logistic regression.
        m, mab, mba_t = self.stream_aggregator(xab, xba_t)
        return m, mab, mba_t

UnitProcOrder = Literal['ena','nae','ean','ane']
class Unit(nn.Module):
    r""" Applies a series of process with encoder, normalizer, and activator in the directed order.
                           
        Args:
           encoder: a trainable unit
           order: a direction of process order. ['ena'|'nae'|'ean'|'ane'], e: encoder, a: activator, n: normalizer. e.g.) 'ena' applies encoder->normalizer->activator.
           normalizer: a normalizer, such as :class:`BatchNormXXC`.
           activator: an activation function, such as :obj:`nn.PReLU`.
        """            
    def __init__(self, 
                 encoder:nn.Module, 
                 order: UnitProcOrder,
                 normalizer:Optional[nn.Module]=None, 
                 activator:Optional[nn.Module]=None):
        super().__init__()
        self.encoder = encoder
        self.normalizer = normalizer
        self.activator = activator
        self.order = order
        self.forward = eval("self._forward_{}".format(order))
        
    def _forward_ena(self, x:torch.Tensor, dim_target:int)->torch.Tensor:
        x = self.encoder(x, dim_target)
        if self.normalizer is not None:
            x = self.normalizer(x)
        if self.activator is not None:
            x = self.activator(x)
        return x
    def _forward_nae(self, x:torch.Tensor, dim_target:int)->torch.Tensor:
        if self.normalizer is not None:
            x = self.normalizer(x)
        if self.activator is not None:
            x = self.activator(x)
        x = self.encoder(x, dim_target)
        return x
    
    def _forward_ean(self, x:torch.Tensor, dim_target:int)->torch.Tensor:
        x = self.encoder(x, dim_tar)
        if self.activator is not None:
            x = self.activator(x)
        if self.normalizer is not None:
            x = self.normalizer(x)
        return x
    
    def _forward_ane(self, x:torch.Tensor, dim_target:int)->torch.Tensor:
        if self.activator is not None:
            x = self.activator(x)
        if self.normalizer is not None:
            x = self.normalizer(x)
        x = self.encoder(x, dim_target)
        return x
    
    def forward(self, x:torch.Tensor, dim_target:int)->torch.Tensor:
        r""" Applies unit process. This function is replaced to any of Unit._forward_* functions in :func:`__init__` based on the argument **order**.
                
        Shape:
           - x: :math:`(\ldots, C)`
           
        Args:
           x: a source features.
           dim_target: dimention of target vertex.
           
        Returns:
           A processed features.
        """        
        raise RuntimeError("This function should never called since replaced to other function at initialization.")        
    
class UnitListGenerator():
    r""" A factory of units.
    
        Args:
           input_channels: input_channels for the first unit.
           output_channels_list: output_channels for the units. 
        """            
    def __init__(self,
                 input_channels:int,
                 output_channels_list:List[int],
                ):
        self.input_channels = input_channels
        self.output_channels_list = output_channels_list    
        
    def generate(self, interactor:Optional[Interactor]=None)->List[Unit]:            
        r""" Generates a list of units, assuming the interactor at each end of unit-process.
                           
        Args:
           interactor: a concrete class of :class:`Interactor`. Typically, :class:`CrossConcat`. If None, assumes no interaction at each end of unit-process.
           
        Returns:
           a list of units.

        """        
            
        if interactor:
            in_chs = [self.input_channels]+[interactor.output_channels(out_ch) for out_ch in self.output_channels_list[:-1] ]
        else:
            in_chs = [self.input_channels]+[out_ch for out_ch in self.output_channels_list[:-1] ]
            
        L = len(in_chs)
        return self._build(in_chs)
                
    def _build(self, in_channels_list:List[int]):
        r""" Generates the list of units based on the directed in/out channels.
                           
        Args:
           in_channels_list: the list of in_channels calculated in :func:`generate`.
           
        Returns:
           a list of units.
        """        
        raise RuntimeError("This function must be implemented in each child class.")
        
ExclusiveElementsOfUnit = Literal['none', 'normalizer', 'all'] # standard, biased, dual
class MatchingModuleHead(nn.Module):
    r""" A head of matching module. This controlls the way of interaction at each end of unit-process and residual paths.
    
        Args:
           units_generator: an instance of :class:`UnitListGenerator` that generate unit-process.
           interactor: an instance of :class:`Interactor` applied at each end of unit-process. If `None`, the class assumes the single-stream process (transpose src/tar at each unit-process instead of interaction).
           calc_residual: set the layer where residual pass is connected.
           keep_first_var_after: set the first layer where the feature is saved for the first residual pass.
           exclusive_elements_of_unit: directs shared elements of each unit at two-stream mode (thus ignored if interactor is `None`). 'none': all the elements in each unit is shared among streams. 'normalizer': all the elements other than normalizer are shared. 'all': all the elements are cloned for the 2nd stream (thus, not shared with the 1st stream).

     """                
    def __init__(self,
                 units_generator: UnitListGenerator,
                 interactor:Optional[Interactor]=None,
                 calc_residual:Optional[List[bool]]=None,
                 keep_first_var_after:int=0,
                 exclusive_elements_of_unit:ExclusiveElementsOfUnit='none',
                ):
        super().__init__()
        if interactor:
            self.interactor = interactor
        units = units_generator.generate(interactor)
        # prepare for residual paths.
        L = len(units)
        self.keep_first_var_after = keep_first_var_after
        if calc_residual is None:
            self.calc_residual = [False] * L
            self.use_residual = False
        else:            
            assert(L == len(calc_residual))
            self.calc_residual = calc_residual
            self.use_residual = sum(self.calc_residual)>0
            assert(0 == sum(self.calc_residual[:self.keep_first_var_after]))
            
        self._build_two_stream_structure(units, exclusive_elements_of_unit, interactor is None)
        
        self.input_channels = units_generator.input_channels
        
        if interactor:
            self.output_channels = self.interactor.output_channels(units_generator.output_channels_list[-1])
        else:
            self.output_channels = units_generator.output_channels_list[-1]

        
    def _build_two_stream_structure(self, 
                                    units:List[Unit],
                                    exclusive_elements_of_unit:ExclusiveElementsOfUnit='none',
                                    is_single_stream:bool = False,
                                   )->None:
        # register module_units as a child nn.Module.
        
        if is_single_stream:
            # !!!override forward by forward_single_stream!!!
            assert(exclusive_elements_of_unit=='none') 
            # assert non-default exclusive_... value for the fail-safe (the value is ignored when is_single_stream==True).
            self.forward = self._forward_single_stream
            self.stream = nn.ModuleList(units)
        # make 2nd stream
        elif exclusive_elements_of_unit == 'none':
            self.stream0 = nn.ModuleList(units)
            self.stream1 = nn.ModuleList(units) # shallow copy
            return
        elif exclusive_elements_of_unit == 'all':
            units2 = deepcopy(units) # deep copy
            self.stream0 = nn.ModuleList(units)
            self.stream1 = nn.ModuleList(units2) # shallow copy
            return
        elif exclusive_elements_of_unit == 'normalizer':
            module_normalizers2 = [deepcopy(m.normalizer) for m in units]  # deep copy normalizers   
            units2 = [                
                Unit(
                    unit.encoder, # shallow copy
                    unit.order, # shallow copy
                    normalizer, # deep-copied normalizer 
                    unit.activator, # shallow copy
                )
                for unit, normalizer in zip(units, module_normalizers2)
            ]
            self.stream0 = nn.ModuleList(units)
            self.stream1 = nn.ModuleList(units) # shallow copy
            return
        else:
            raise RuntimeError("Unknown ExclusiveElementsOfUnit: {}".format(exclusive_elements_of_unit))
            
            
    def forward(self, xab:torch.Tensor, xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        r""" Applies a series of unit process with a two-stream manner.
                
        Shape:
           - xab: :math:`(\ldots, N, M, C)`
           - xba_t: :math:`(\ldots, N, M, C)`
           - output: a pair of tensors with the shape :math:`(\ldots, N, M, C')`
           
        Args:
           xab: edge features directed from the side `a` to `b`.           
           xba_t: edge features directed from the side `b` to `a`.           
           
        Returns:
           A pair of processed features.
        """
        xab_keep, xba_t_keep = xab, xba_t
        for l, (unit0, unit1) in enumerate(zip(self.stream0, self.stream1)):
            calc_res = self.calc_residual[l]
            xab_fut = torch.jit.fork(unit0, xab, dim_target=-2)
            xba_t = unit1(xba_t, dim_target=-3)
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
    
    def _forward_single_stream(self, xab:torch.Tensor, xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        r""" Applies a series of unit process with a single-stream manner. This function is set as self.forward if interactor is `None` at :func:`__init__`.
                
        Shape:
           - xab: :math:`(\ldots, N, M, C)`
           - xba_t: an ignored argument (an argument only for the interface compatibility).
           - output: a pair of tensors with the shape :math:`(\ldots, N, M, C')`
           
        Args:
           xab: a source features.
           xba_t: 
           
        Returns:
           A pair of processed features, but in the single stream mode, they are the same tensor.
        """
        xab_keep = xab
        for l, unit in enumerate(self.stream):
            calc_res = self.calc_residual[l]
            if l%2==0:
                dim = -3
            else:
                dim = -2
            xab = unit(xab, dim_target=dim)
            
            if self.use_residual:              
                if l==self.keep_first_var_after:
                    # keep values after the directed unit's process.
                    xab_keep = xab
                if calc_res:
                    xab_keep, xab = xab, xab + xab_keep            
        return xab, xab # return xab as two-stream output.
    


        
class WeaveNetUnitListGenerator(UnitListGenerator):
    r""" A factory of weavenet units.
    
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
        r""" Generates the list of units for weavenet.
                           
        Args:
           in_channels_list: the list of in_channels calculated in :func:`generate`.
           
        Returns:
           a list of weavenet units.
        """
        return [
            Unit(
                SetEncoderPointNet(in_ch, mid_ch, out_ch),
                'ena',
                BatchNormXXC(out_ch),
                nn.PReLU(),)
            for in_ch, mid_ch, out_ch in zip(in_channels_list, self.mid_channels_list, self.output_channels_list)
        ]
        
class ExperimentalUnitListGenerator(WeaveNetUnitListGenerator):
    r""" A factory of experimental units. This is a sample class for user custom units.
    
        Args:
           input_channels: input_channels for the first unit.
           mid_channels_list: mid_channels for each point-net-based set encoders.
           output_channels_list: output_channels for the units. 
        """            
    class Encoder(SetEncoderBase):
        r""" A sample of experimental unit encoder.
    
        Args:
           in_channels: input_channels for the first unit.
           mid_channels_list: mid_channels for each point-net-based set encoders.
           output_channels_list: output_channels for the units. 
        """            
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
                MaxPoolingAggregator(),
                DifferenceConcatMerger(dim_feature=-1),
                second_process,
                **kwargs,
            )
            
    def _build(self, in_channels_list:List[int]):
        r""" Generates the list of experimental units. Customizing this class makes it easy to test a new primitive weavenet structure.
                           
        Args:
           in_channels_list: the list of in_channels calculated in :func:`generate`.
           
        Returns:
           a list of experimental units. 
        """
        return [
            Unit(
                #self.Encoder(in_ch, mid_ch, out_ch),
                SetEncoderPointNetCrossDirectional(in_ch, mid_ch, out_ch),
                'ena',
                BatchNormXXC(out_ch),
                nn.PReLU(),)
            for in_ch, mid_ch, out_ch in zip(in_channels_list, self.mid_channels_list, self.output_channels_list)
        ]
        
class WeaveNetHead(MatchingModuleHead):
    r""" A head for WeaveNet.
    
        Args:
            input_channels: input_channels for the first unit (see :class:`WeaveNetUnitListGenerator`).
            output_channels_list: output_channels for the units (see :class:`WeaveNetUnitListGenerator`). 
            mid_channels_list: mid_channels for each point-net-based set encoders (see :class:`WeaveNetUnitListGenerator`).
            calc_residual: see :class:`MatchingModuleHead`
            keep_first_var_after: see :class:`MatchingModuleHead`
            exclusive_elements_of_unit: see :class:`MatchingModuleHead`
            is_single_stream: see :class:`MatchingModuleHead`

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
            WeaveNetUnitListGenerator(input_channels, mid_channels_list, output_channels_list),
            interactor = interactor,            
            calc_residual = calc_residual,
            keep_first_var_after = keep_first_var_after,
            exclusive_elements_of_unit = exclusive_elements_of_unit,
        )        
        
class ExperimentalHead(MatchingModuleHead):
    r""" A head for the experimental Model.
    
        Args:
            input_channels: input_channels for the first unit (see :class:`ExperimentalUnitListGenerator`).
            output_channels_list: output_channels for the units (see :class:`ExperimentalUnitListGenerator`). 
            mid_channels_list: mid_channels for each point-net-based set encoders (see :class:`ExperimentalUnitListGenerator`).
            calc_residual: see :class:`MatchingModuleHead`
            keep_first_var_after: see :class:`MatchingModuleHead`
            exclusive_elements_of_unit: see :class:`MatchingModuleHead`
            is_single_stream: see :class:`MatchingModuleHead`

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
            ExperimentalUnitListGenerator(input_channels, mid_channels_list, output_channels_list),
            interactor = interactor,            
            calc_residual = calc_residual,
            keep_first_var_after = keep_first_var_after,
            exclusive_elements_of_unit = exclusive_elements_of_unit,
        )        
        

if __name__ == "__main__":
    #_ = WeaveNetOldImplementation(2, 2,1)
    _ = WeaveNet(
            WeaveNetHead6(1,), 2, #input_channel:int,
                 [4,8,16], #out_channels:List[int],
                 [2,4,8], #mid_channels:List[int],1,2,2)
                 calc_residual=[False, False, True],
                 keep_first_var_after = 0,
                 stream_aggregator = DualSoftMaxSqrt())
                 
