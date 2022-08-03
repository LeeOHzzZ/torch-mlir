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


N, C, H, W = 16, 3, 8, 8

softmax = torch.nn.Softmax(1)
input = torch.randn((N, C*H*W), dtype=torch.float32)



# print(f"jit script version of layernorm\n", torch.jit.script(layernorm))
# print(f"\njit traced version of layernorm\n", torch.jit.trace(layernorm, input))

compiled_linalg = torch_mlir.compile(
    softmax, input, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)
compiled_tosa = torch_mlir.compile(
    softmax, input, output_type=torch_mlir.OutputType.TOSA
)
with open("linalg_softmax_large.mlir", "w") as fout:
    fout.write(compiled_linalg.operation.get_asm())
with open("tosa_softmax_large.mlir", "w") as fout:
    fout.write(compiled_tosa.operation.get_asm())


N, C, H, W = 2, 3, 4, 4

softmax = torch.nn.Softmax(1)
input = torch.randn((N, C*H*W), dtype=torch.float32)


# print(f"jit script version of layernorm\n", torch.jit.script(layernorm))
# print(f"\njit traced version of layernorm\n", torch.jit.trace(layernorm, input))

compiled_linalg = torch_mlir.compile(
    softmax, input, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)
compiled_tosa = torch_mlir.compile(
    softmax, input, output_type=torch_mlir.OutputType.TOSA
)
with open("linalg_softmax_2x3x4x4.mlir", "w") as fout:
    fout.write(compiled_linalg.operation.get_asm())
with open("tosa_softmax_2x3x4x4.mlir", "w") as fout:
    fout.write(compiled_tosa.operation.get_asm())