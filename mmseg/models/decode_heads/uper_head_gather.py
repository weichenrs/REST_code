# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM
import numpy as np
import torch.distributed as dist

def BN_convert_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module

@MODELS.register_module()
class UPerHead_gather(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        BN_convert_float(self)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # inputs = gather_inputs.apply(inputs)

        # input_list = [torch.zeros_like(inputs[0])] * dist.get_world_size()
        # input_list = all_gather([torch.zeros_like(inputs[0])] * dist.get_world_size(), inputs[0])

        # my_inputs = []
        # for input in inputs:
        #     my_input = torch.cat(all_gather([torch.zeros_like(input)] * dist.get_world_size(), input), -2)
        #     my_inputs.append(my_input)

        # my_inputs = [ torch.cat(all_gather([torch.zeros_like(input)] * dist.get_world_size(), input), -2) for input in inputs]
        my_inputs = tuple([my_gather(input) for input in inputs])
        # inputs[0] = torch.cat(all_gather([torch.zeros_like(inputs[0])] * dist.get_world_size(), inputs[0]), -2)
        # inputs = tuple([ torch.cat([input[ind].cuda(dist.get_rank()) for input in input_list], -2) for ind in range(len(input_list[0])) ])
        # if dist.get_rank() == 0:
        #     import pdb
        #     pdb.set_trace()
        # dist.barrier()

        inputs = self._transform_inputs(my_inputs)

        # if dist.get_rank() == 0:
        #     import pdb;pdb.set_trace()
        # dist.barrier()

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)

        # if torch.isnan(feats).any():
        # import torch.distributed as dist
        # if dist.get_rank() == 0:
        #     import pdb;pdb.set_trace()
        # dist.barrier()

        # feats = split_feats.apply(feats)
        # feats = torch.chunk(feats, dist.get_world_size(), dim=-2)[dist.get_rank()] 
        feats = my_split(feats)

        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    

# class gather_inputs(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, inputs):
#         input_list = [None] * dist.get_world_size()
#         dist.all_gather_object(input_list, inputs)
#         inputs = tuple([ torch.cat([input[ind].cuda(dist.get_rank()) for input in input_list], -2) for ind in range(len(input_list[0])) ])

#         # ctx.save_for_backward(inputs)
#         return inputs

#     @staticmethod
#     def backward(ctx, grad_output):

#         dist.scatter_object_list()
#         dist.reduce_scatter()

#         return grad_output
#         # result, = ctx.saved_tensors
#         # return grad_output * result

# def gather_inputs(inputs):

#     input_list = [None] * dist.get_world_size()
#     # dist.gather_object(inputs, input_list if dist.get_rank() == 0 else None, dst=0)
#     dist.all_gather_object(input_list, inputs)
#     # inputs = []
#     # for ind in range(len(input_list[0])):
#     #     inputs.append( np.concatenate( [input[ind] for input in input_list], -2 ) )
#     inputs = tuple([ torch.cat([input[ind].cuda(dist.get_rank()) for input in input_list], -2) for ind in range(len(input_list[0])) ])

#     if dist.get_rank() == 0:
#         import pdb;pdb.set_trace()
#     dist.barrier()

#     return inputs

# def split_feats(feats):
#         import torch.distributed as dist
#         input_list = [None] * dist.get_world_size()
#         # dist.gather_object(inputs, input_list if dist.get_rank() == 0 else None, dst=0)
#         dist.all_gather_object(input_list, inputs)
#         inputs = []
#         # for ind in range(len(input_list[0])):
#         #     inputs.append( np.concatenate( [input[ind] for input in input_list], -2 ) )
#         inputs = tuple([ torch.cat([input[ind].cuda(dist.get_rank()) for input in input_list], -2) for ind in range(len(input_list[0])) ])

#         if dist.get_rank() == 0:
#             import pdb;pdb.set_trace()
#         dist.barrier()

#         return feats

# class split_feats(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, i):
#         result = torch.exp(i)
#         ctx.save_for_backward(result)
#         return result

#     @staticmethod
#     def backward(ctx, grad_output):
#         result, = ctx.saved_tensors
#         return grad_output * result

# class AllGather(torch.autograd.Function):
#     """ 
#     all_gather with gradient back-propagation
#     """
#     @staticmethod
#     def forward(ctx, tensor_list, tensor):
#         dist.all_gather(tensor_list, tensor)
#         # return tuple(tensor_list)
#         return tensor_list

#     @staticmethod
#     # def backward(ctx, *grad_list):
#     #     grad_list = list(grad_list)
#     def backward(ctx, grad_list):
#         rank = dist.get_rank()

#         dist_ops = [
#             dist.reduce(grad_list[i], i, async_op=True) for i in range(dist.get_world_size())
#         ]

#         for op in dist_ops:
#             op.wait()

#         return None, grad_list[rank]

# all_gather = AllGather.apply

import torch
import torch.distributed as dist

def all_gather(tensor, async_op: bool = False):
    r"""Gathers all tensors from the parallel group and concatenates them in a
    specific dimension.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be gathered.
        dim (int): The dimension concatenating in.
        parallel_mode (:class:`colossalai.context.ParallelMode`): Parallel group mode used in this communication.
        async_op (bool, optional): Whether operations are asynchronous.

    Returns:
        Union[tuple(:class:`torch.Tensor`, work handle), :class:`torch.Tensor`]: The result of all-together only,
        if async_op is set to False. A tuple of output of all-gather and Async work handle, if async_op is set to True.
    """
    tensor = tensor.contiguous()
    out = [torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)] * dist.get_world_size()
    group = None
    work = dist.all_gather(out, tensor, group=group, async_op=async_op)
    out = torch.cat(out, dim=-2)
    # if dist.get_rank() == 0:
    #     import pdb
    #     pdb.set_trace()
    # dist.barrier()
    if async_op:
        return out, work
    else:
        return out
    
from torch.distributed import ReduceOp
def all_reduce(tensor, op: ReduceOp = ReduceOp.SUM, async_op: bool = False):
    r"""Reduces all tensors then scatters it in a specific dimension to all
    
    members in the parallel group.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be reduce_scattered.
        dim (int): The dimension concatenating in.
        parallel_mode (:class:`colossalai.context.ParallelMode`): Parallel group mode used in this communication.
        op (torch.distributed.ReduceOp, optional): The type of reduce operation,
            should be included in [SUM, AVG, PRODUCT, MIN, MAX, BAND, BOR, BXOR].
            More details about ReduceOp please refer to
            `ReduceOp <https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp>`_.
        async_op (bool, optional): Whether operations are asynchronous.

    Returns:
        Union[tuple(:class:`torch.Tensor`, work handle), :class:`torch.Tensor`]: The result of reduce_scatter only,
        if async_op is set to False. A tuple of output of all-gather and Async work handle, if async_op is set to True.
    """
    temp = torch.chunk(tensor, dist.get_world_size(), dim=-2)
    out = temp[dist.get_rank()].contiguous()
    group = None

    # if dist.get_rank() == 0:
    #     import pdb
    #     pdb.set_trace()
    # dist.barrier()

    # work = dist.all_reduce(tensor_out, op=op, group=group, async_op=async_op)
    if async_op:
        return out, _
    else:
        return out

# from torch.cuda.amp import custom_bwd, custom_fwd
class _AllGather2(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return all_gather(input_)
    
    @staticmethod
    def forward(ctx, inputs):
        return all_gather(inputs)

    @staticmethod
    def backward(ctx, output_grad):
        return all_reduce(output_grad)

def my_gather(tensor):
    r"""All gather the tensor of 2D parallelism.

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        dim (int): Dimension to gather.
        parallel_mode (:class:`colossalai.context.ParallelMode`): The parallel mode tensor used.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    # return _AllGather.apply(tensor)
    return _AllGather2.apply(tensor)

def my_split(tensor):
    r"""All gather the tensor of 2D parallelism.

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        dim (int): Dimension to gather.
        parallel_mode (:class:`colossalai.context.ParallelMode`): The parallel mode tensor used.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    # return _AllGather.apply(tensor)
    return _AllSplit.apply(tensor)

class _AllGather(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)
    
    @staticmethod
    def forward(ctx, inputs):
        return _gather_along_last_dim(inputs)

    @staticmethod
    def backward(ctx, output_grad):
        return _split_along_last_dim(output_grad)

class _AllSplit(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _split_along_last_dim(input_)
    
    @staticmethod
    def forward(ctx, inputs):
        return _split_along_last_dim(inputs)

    @staticmethod
    def backward(ctx, output_grad):
        return _gather_along_last_dim(output_grad)


from typing import List
def _split_along_last_dim(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = dist.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank()
    output = input_list[rank].contiguous()

    return output

def _gather_along_last_dim(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = dist.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 2
    rank = dist.get_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=None)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output

def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 2
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list