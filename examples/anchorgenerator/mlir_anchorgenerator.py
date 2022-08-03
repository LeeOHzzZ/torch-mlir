# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import math
from typing import List, Tuple

import torch
from torch import nn

<<<<<<< HEAD
import numpy as np
=======
>>>>>>> 2ca47271a4149015bd74a0ee23531596a00e98df
# from detectron2.config import configurable
# from detectron2.layers import ShapeSpec, move_device_like
# from detectron2.structures import Boxes, RotatedBoxes
# from detectron2.utils.registry import Registry

# ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")
# ANCHOR_GENERATOR_REGISTRY.__doc__ = """
# Registry for modules that creates object detection anchors for feature maps.
#
# The registered object will be called with `obj(cfg, input_shape)`.
# """

# detectron2/structures/boxes.py
class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(
                tensor, dtype=torch.float32, device=torch.device("cpu")
            )
        else:
            tensor = tensor.to(torch.float32)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert (
            b.dim() == 2
        ), "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(
        self, box_size: Tuple[int, int], boundary_threshold: int = 0
    ) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    #    @property
    #    def device(self) -> device:
    #        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor


# detectron2/layers/wrappers.yp
@torch.jit.script_if_tracing
def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Tracing friendly way to cast tensor to another tensor's device. Device will be treated
    as constant during tracing, scripting the casting process as whole can workaround this issue.
    """
    return src.to(dst.device)


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # Use non-persistent buffer so the values are not saved in checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def _create_grid_offsets(
    size: List[int], stride: int, offset: float, target_device_tensor: torch.Tensor
):

<<<<<<< HEAD
    # print("GRID_OFFSETS")
    # print(size)
=======
    print("GRID_OFFSETS")
    print(size)
>>>>>>> 2ca47271a4149015bd74a0ee23531596a00e98df
    grid_height, grid_width = size
    # shifts_xo = move_device_like(
    #    torch.arange(offset * stride, grid_width * stride, step=stride, dtype=torch.float32),
    #    target_device_tensor,
    # )
    shifts_x = torch.arange(
        offset * stride, grid_width * stride, step=stride, dtype=torch.float32
    )

    # shifts_y = move_device_like(
    #    torch.arange(offset * stride, grid_height * stride, step=stride, dtype=torch.float32),
    #    target_device_tensor,
    # )

    shifts_y = torch.arange(
        offset * stride, grid_height * stride, step=stride, dtype=torch.float32
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


def _broadcast_params(params, num_features, name):
    """
    If one size (or aspect ratio) is specified and there are multiple feature
    maps, we "broadcast" anchors of that single size (or aspect ratio)
    over all feature maps.

    If params is list[float], or list[list[float]] with len(params) == 1, repeat
    it num_features time.

    Returns:
        list[list[float]]: param for each feature
    """
    assert isinstance(
        params, collections.abc.Sequence
    ), f"{name} in anchor generator has to be a list! Got {params}."
    assert len(params), f"{name} in anchor generator cannot be empty!"
    if not isinstance(params[0], collections.abc.Sequence):  # params is list[float]
        return [params] * num_features
    if len(params) == 1:
        return list(params) * num_features
    assert len(params) == num_features, (
        f"Got {name} of length {len(params)} in anchor generator, "
        f"but the number of input features is {num_features}!"
    )
    return params


# @ANCHOR_GENERATOR_REGISTRY.register()
class DefaultAnchorGenerator(nn.Module):
    """
    Compute anchors in the standard ways described in
    "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".
    """

    box_dim: torch.jit.Final[int] = 4
    """
    the dimension of each anchor box.
    """

    #    @configurable
    def __init__(self, *, sizes, aspect_ratios, strides, offset=0.5):
        """
        This interface is experimental.

        Args:
            sizes (list[list[float]] or list[float]):
                If ``sizes`` is list[list[float]], ``sizes[i]`` is the list of anchor sizes
                (i.e. sqrt of anchor area) to use for the i-th feature map.
                If ``sizes`` is list[float], ``sizes`` is used for all feature maps.
                Anchor sizes are given in absolute lengths in units of
                the input image; they do not dynamically scale if the input image size changes.
            aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
                (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
            strides (list[int]): stride of each input feature.
            offset (float): Relative offset between the center of the first anchor and the top-left
                corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
        """
        super().__init__()

        self.strides = strides
        self.num_features = len(self.strides)
        # sizes = _broadcast_params(sizes, self.num_features, "sizes")
        # aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

        # print("CELLANCHORS")
        # print(self.cell_anchors)
        #        assert False

        self.offset = offset
        assert 0.0 <= self.offset < 1.0, self.offset

    #    @classmethod
    #    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
    #        return {
    #            "sizes": cfg.MODEL.ANCHOR_GENERATOR.SIZES,
    #            "aspect_ratios": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
    #            "strides": [x.stride for x in input_shape],
    #            "offset": cfg.MODEL.ANCHOR_GENERATOR.OFFSET,
    #        }

    def _calculate_anchors(self, sizes, aspect_ratios):
        # cell_anchors = [
        #    self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
        # ]
        cell_anchors = self.generate_cell_anchors(sizes, aspect_ratios).float()
        # print(cell_anchors)
        return cell_anchors
        # res = torch.tensor(cell_anchors)
        # res = torch.cat(cell_anchors)
        # print(res)
        # return res
        # return BufferList(cell_anchors)

    @property
    @torch.jit.unused
    def num_cell_anchors(self):
        """
        Alias of `num_anchors`.
        """
        return self.num_anchors

    @property
    @torch.jit.unused
    def num_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS in config)

                In standard RPN models, `num_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self, grid_sizes: List[int]):
        """
        Returns:
            list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4
        """
        # anchors = []
        # buffers() not supported by torchscript. use named_buffers() instead
        # buffers: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_buffers()]
        # for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
        #    shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors)
        #    shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        #    anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        # return anchors

        shift_x, shift_y = _create_grid_offsets(
            grid_sizes, self.strides[0], self.offset, self.cell_anchors
        )
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        result = (shifts.view(-1, 1, 4) + self.cell_anchors.view(1, -1, 4)).reshape(
            -1, 4
        )
        return result

    def generate_cell_anchors(
        self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
    ):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).

        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        for size in sizes:
            area = size**2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    #    def forward(self, features: List[torch.Tensor]):
    def forward_orig(self, feature_map: torch.Tensor):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[Boxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
        """
        #        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        grid_sizes = feature_map.shape[-2:]
        #        print("GRIDSIZE")
        #        print(grid_sizes)
        # grid_sizes = torch.tensor([feature_map.shape[-2:]],dtype=torch.float)
        # grid_sizes = torch.tensor([[-4]],dtype=torch.float)
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        # n        print("ANCHORSOVERALL")
        #        print(anchors_over_all_feature_maps)
        #        return anchors_over_all_feature_maps
        #        return [Boxes(x) for x in anchors_over_all_feature_maps]
        # res = Boxes(anchors_over_all_feature_maps)
        # print(res)
        # return res
        return anchors_over_all_feature_maps
        # return grid_sizes
        # return torch.add(feature_map,feature_map)

    def forward_o2(self, feature_map: torch.Tensor):
        grid_sizes = feature_map.shape[-2:]
        # shift_x, shift_y = _create_grid_offsets(grid_sizes, self.strides[0], self.offset, self.cell_anchors)

        grid_height, grid_width = grid_sizes
        shifts_x = torch.arange(
            self.offset * self.strides[0],
            grid_width * self.strides[0],
            step=self.strides[0],
            dtype=torch.float32,
        )
        shifts_y = torch.arange(
            self.offset * self.strides[0],
            grid_height * self.strides[0],
            step=self.strides[0],
            dtype=torch.float32,
        )

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        # return shift_x, shift_y

        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        anchors_over_all_feature_maps = (
            shifts.view(-1, 1, 4) + self.cell_anchors.view(1, -1, 4)
        ).reshape(-1, 4)
        return anchors_over_all_feature_maps

    def forward_o3(self, feature_map: torch.Tensor):
        grid_height, grid_width = 13, 19
        shifts_x = torch.arange(
            self.offset * self.strides[0],
            grid_width * self.strides[0],
            step=self.strides[0],
            dtype=torch.float32,
        )
        shifts_y = torch.arange(
            self.offset * self.strides[0],
            grid_height * self.strides[0],
            step=self.strides[0],
            dtype=torch.float32,
        )

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        cell_anchors = torch.tensor(
            [
                [-22.6274, -11.3137, 22.6274, 11.3137],
                [-16.0000, -16.0000, 16.0000, 16.0000],
                [-11.3137, -22.6274, 11.3137, 22.6274],
                [-45.2548, -22.6274, 45.2548, 22.6274],
                [-32.0000, -32.0000, 32.0000, 32.0000],
                [-22.6274, -45.2548, 22.6274, 45.2548],
                [-67.8822, -33.9411, 67.8822, 33.9411],
                [-48.0000, -48.0000, 48.0000, 48.0000],
                [-33.9411, -67.8822, 33.9411, 67.8822],
                [-90.5097, -45.2548, 90.5097, 45.2548],
                [-64.0000, -64.0000, 64.0000, 64.0000],
                [-45.2548, -90.5097, 45.2548, 90.5097],
                [-113.1371, -56.5685, 113.1371, 56.5685],
                [-80.0000, -80.0000, 80.0000, 80.0000],
                [-56.5685, -113.1371, 56.5685, 113.1371],
            ]
        )

        anchors_over_all_feature_maps = (
            shifts.view(-1, 1, 4) + cell_anchors.view(1, -1, 4)
        ).reshape(-1, 4)
        return anchors_over_all_feature_maps

    def forward_works(self, feature_map: torch.Tensor):
        grid_height, grid_width = 13, 19
        offset = 0.5
        stride = 2.0
        shifts_x = torch.arange(
            offset * stride, grid_width * stride, step=stride, dtype=torch.float32
        )
        shifts_y = torch.arange(
            offset * stride, grid_height * stride, step=stride, dtype=torch.float32
        )

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

        shift_x = torch.flatten(shift_x)
        shift_y = torch.flatten(shift_y)

        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        """
        cell_anchors = torch.tensor(
            [
                [-22.6274, -11.3137, 22.6274, 11.3137],
                [-16.0000, -16.0000, 16.0000, 16.0000],
                [-11.3137, -22.6274, 11.3137, 22.6274],
                [-45.2548, -22.6274, 45.2548, 22.6274],
                [-32.0000, -32.0000, 32.0000, 32.0000],
                [-22.6274, -45.2548, 22.6274, 45.2548],
                [-67.8822, -33.9411, 67.8822, 33.9411],
                [-48.0000, -48.0000, 48.0000, 48.0000],
                [-33.9411, -67.8822, 33.9411, 67.8822],
                [-90.5097, -45.2548, 90.5097, 45.2548],
                [-64.0000, -64.0000, 64.0000, 64.0000],
                [-45.2548, -90.5097, 45.2548, 90.5097],
                [-113.1371, -56.5685, 113.1371, 56.5685],
                [-80.0000, -80.0000, 80.0000, 80.0000],
                [-56.5685, -113.1371, 56.5685, 113.1371],
            ]
        )
        """

        anchors_over_all_feature_maps = shifts.view(-1, 1, 4) + self.cell_anchors.view(
            1, -1, 4
        )
        res = anchors_over_all_feature_maps.reshape(-1, 4)
        return res

    def forward(self, feature_map: torch.Tensor):
        # grid_height, grid_width = 13, 19
        
        # set the numbers to multiple of 4 to avoid padding
        grid_height, grid_width = 16, 20

        offset = 0.5
        stride = 2.0
        # offset = torch.tensor([0.5], dtype=torch.float32)
        # stride = torch.tensor([2.0], dtype=torch.float32)

        # this basically accomplishes the same thing as the meshgrid
        shifts_x2 = torch.arange(
            0, grid_width * grid_height * stride, step=stride, dtype=torch.float32
        )
        # print("a.shape is ", shifts_x2.shape)
        a, b = shifts_x2, stride * grid_width
        # same as doing a modulus
        shifts_x3 = a - torch.floor(a.div(b)) * b
        shifts_x3 = shifts_x3 + offset * stride

        shifts_y2 = torch.arange(
            0, grid_width * grid_height * stride, step=stride, dtype=torch.float32
        )
        shifts_y2 = torch.floor(shifts_y2.div(stride * grid_width))
        shifts_y2 = shifts_y2 * stride + offset * stride

        shift_x = shifts_x3
        shift_y = shifts_y2

        # same as: torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        # print(shift_x.shape, type(list(shift_x.shape)))
        sx = torch.unsqueeze(shift_x, 1)
        sy = torch.unsqueeze(shift_y, 1)

        # sx = shift_x.reshape(list(shift_x.shape) + [1])
        # sy = shift_y.reshape(list(shift_y.shape) + [1])

        shifts = torch.cat((sx, sy, sx, sy), 1)

        # print(sx.shape, sx_2.shape)

        """
        cell_anchors = torch.tensor([[ -22.6274,  -11.3137,   22.6274,   11.3137],
                                     [ -16.0000,  -16.0000,   16.0000,   16.0000],
                                     [ -11.3137,  -22.6274,   11.3137,   22.6274],
                                     [ -45.2548,  -22.6274,   45.2548,   22.6274],
                                     [ -32.0000,  -32.0000,   32.0000,   32.0000],
                                     [ -22.6274,  -45.2548,   22.6274,   45.2548],
                                     [ -67.8822,  -33.9411,   67.8822,   33.9411],
                                     [ -48.0000,  -48.0000,   48.0000,   48.0000],
                                     [ -33.9411,  -67.8822,   33.9411,   67.8822],
                                     [ -90.5097,  -45.2548,   90.5097,   45.2548],
                                     [ -64.0000,  -64.0000,   64.0000,   64.0000],
                                     [ -45.2548,  -90.5097,   45.2548,   90.5097],
                                     [-113.1371,  -56.5685,  113.1371,   56.5685],
                                     [ -80.0000,  -80.0000,   80.0000,   80.0000],
                                     [ -56.5685, -113.1371,   56.5685,  113.1371]])
        """
        # same as: shifts.view(-1, 1, 4)
        s_v = torch.unsqueeze(shifts, 1)

        # same as: self.cell_anchors.view(1, -1, 4)
        ca_v = torch.unsqueeze(self.cell_anchors, 0)
        anchors_over_all_feature_maps = s_v + ca_v

        # same as: reshape(-1, 4)
        res = torch.flatten(anchors_over_all_feature_maps, start_dim=0, end_dim=1)
        return res