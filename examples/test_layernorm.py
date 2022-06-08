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


N, C, H, W = 1, 3, 3, 3
layernorm = torch.nn.LayerNorm([C, H, W])

input = torch.randn((N, C, H, W))

print(f"jit script version of layernorm\n", torch.jit.script(layernorm))
print(f"\njit traced version of layernorm\n", torch.jit.trace(layernorm, input))

compiled = torch_mlir.compile(
    layernorm, input, output_type=torch_mlir.OutputType.TORCH
)
# print(compiled)

