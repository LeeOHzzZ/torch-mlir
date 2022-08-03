import numpy as np
import torch
import torch_mlir
import detectron2

# from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from mlir_anchorgenerator import DefaultAnchorGenerator

anchor_sizes = [32, 64, 96, 128, 160]
anchor_aspect_ratios = np.array([0.5, 1.0, 2.0], dtype=np.float32)
strides = [2]
# offset = np.float32(0.5)
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

anchor_gen(example_tensor)

# anchors = anchor_gen(example_tensor)[0]
# print(anchors)
# print(len(anchors))

compiled_linalg = torch_mlir.compile(
    anchor_gen, example_tensor, torch_mlir.OutputType.LINALG_ON_TENSORS, 
    # set the tracing to true can avoid the dynamic shape of 
    use_tracing=True
)

# compiled_tosa = torch_mlir.compile(
#     anchor_gen, example_tensor, torch_mlir.OutputType.TOSA, use_tracing=True
# )

compiled_torch = torch_mlir.compile(
    anchor_gen, example_tensor,  
    # set the tracing to true can avoid the dynamic shape of 
    use_tracing=True
)

with open("torch_anchorgenerator.mlir", "w") as fout:
    fout.write(compiled_torch.operation.get_asm())

# with open("tosa_anchorgenerator.mlir", "w") as fout:
#     fout.write(compiled_tosa.operation.get_asm())

with open("linalg_anchorgenerator.mlir", "w") as fout:
    fout.write(compiled_linalg.operation.get_asm())
# print(compiled_torch)