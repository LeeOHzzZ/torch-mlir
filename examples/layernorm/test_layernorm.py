""" This file try lowering pytorch layernorm operator down to 
    mlir
"""

import torch
import torchvision

import torch_mlir
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

import numpy as np


N, C, H, W = 2, 3, 36, 80 
# N, C, H, W = 2, 3, 4, 4 
layernorm = torch.nn.LayerNorm([C, H, W])

input = torch.randn((N, C, H, W))

# print(f"jit script version of layernorm\n", torch.jit.script(layernorm))
# print(f"\njit traced version of layernorm\n", torch.jit.trace(layernorm, input))

compiled_torch = torch_mlir.compile(
    layernorm, input, output_type=torch_mlir.OutputType.TORCH
)
compiled_linalg = torch_mlir.compile(
    layernorm, input, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)
with open("linalg_layernorm.mlir", "w") as fout:
    fout.write(compiled_linalg.operation.get_asm())

print("torchversion:", compiled_torch)
# print("linalg_version: ", compiled_linalg)

# flatten version
N, CHW = 2, C * H * W

layernorm_flatten = torch.nn.LayerNorm([C*H*W])
input = torch.randn((N, CHW))

compiled_linalg = torch_mlir.compile(
    layernorm_flatten, input, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)
with open("linalg_layernorm_flatten.mlir", "w") as fout:
    fout.write(compiled_linalg.operation.get_asm())


class LayerNorm_Linear(torch.nn.Module):
    def __init__(self, layersize) -> None:
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(layersize)

    def forward(self, x, weights, bias):
        normalizedFeature = self.layernorm(x)
        return torch.add(torch.mul(normalizedFeature, weights), bias)


N, C, H, W = 4, 8, 16, 16 

layernorm_flatten = torch.nn.LayerNorm([C*H*W])
input = torch.randn((N, C*H*W))
weights = torch.randn((C*H*W))
bias = torch.randn((C*H*W))
compiled_linalg = torch_mlir.compile(
    LayerNorm_Linear((C*H*W)), [input, weights, bias],
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)
compiled_tosa = torch_mlir.compile(
    LayerNorm_Linear((C*H*W)), [input, weights, bias],
    output_type=torch_mlir.OutputType.TOSA
)
with open("linalg_layernorm_linear_4x8x16x16.mlir", "w") as fout:
    fout.write(compiled_linalg.operation.get_asm())
with open("tosa_layernorm_linear_4x8x16x16.mlir", "w") as fout:
    fout.write(compiled_tosa.operation.get_asm())


test_layer = LayerNorm_Linear((N, C*H*W))
out = test_layer(input, weights, bias)
print(out.shape)
