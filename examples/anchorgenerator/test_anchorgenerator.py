import torch
import torch_mlir
import detectron2

# from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from mlir_anchorgenerator import DefaultAnchorGenerator

anchor_sizes = [32, 64, 96, 128, 160]
anchor_aspect_ratios = [0.5, 1.0, 2.0]
strides = [2]
offset = 0.5
feature_height = 16
feature_width = 20

anchor_gen = DefaultAnchorGenerator(
    sizes = anchor_sizes,
    aspect_ratios = anchor_aspect_ratios,
    strides = strides,
    offset = offset
)

example_tensor = torch.randn(
    [1, 256, feature_height, feature_width], dtype=torch.float32
)

anchors = anchor_gen(example_tensor)[0]
# print(anchors)
# print(len(anchors))

# compiled_torch = torch_mlir.compile(
#     anchor_gen, example_tensor, torch_mlir.OutputType.LINALG_ON_TENSORS, 
#     # set the tracing to true can avoid the dynamic shape of 
#     use_tracing=True
# )

# with open("linalg_anchorgenerator.mlir", "w") as fout:
#     fout.write(compiled_torch.operation.get_asm())
# print(compiled_torch)