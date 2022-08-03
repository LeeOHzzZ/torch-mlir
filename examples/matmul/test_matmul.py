""" This file intends to test offloading a simple vector add operator
    through the compiler pipeline
"""

import torch
import torch.nn as nn

import torch_mlir
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

import numpy as np

class matmul(torch.nn.Module):
    
    def forward(self, x, y):
        z = torch.matmul(x, y)
        return z


shape = (128, 128)
x = torch.randn(shape, dtype=torch.float32)
y = torch.randn(shape, dtype=torch.float32)
# z = torch.randn(shape, dtype=torch.float32)

matmul_compiled = torch_mlir.compile(matmul(),
                                   [x, y], torch_mlir.OutputType.LINALG_ON_TENSORS)

with open("linalg_matmul_128x128.mlir", "w") as fout:
    fout.write(matmul_compiled.operation.get_asm())

shape = (256, 256)
x = torch.randn(shape, dtype=torch.float32)
y = torch.randn(shape, dtype=torch.float32)
# z = torch.randn(shape, dtype=torch.float32)
matmul_256x256_compiled = torch_mlir.compile(matmul(),
                                   [x, y], torch_mlir.OutputType.LINALG_ON_TENSORS)

with open("linalg_256x256_matmul.mlir", "w") as fout:
    fout.write(matmul_256x256_compiled.operation.get_asm())

shape = (4, 4)
x = torch.randn(shape, dtype=torch.float32)
y = torch.randn(shape, dtype=torch.float32)

matmul_compiled = torch_mlir.compile(matmul(),
                                   [x, y], torch_mlir.OutputType.LINALG_ON_TENSORS)

with open("linalg_matmul_4x4.mlir", "w") as fout:
    fout.write(matmul_compiled.operation.get_asm())

shape = (64, 64)
x = torch.randn(shape, dtype=torch.float32)
y = torch.randn(shape, dtype=torch.float32)

matmul_compiled = torch_mlir.compile(matmul(),
                                   [x, y], torch_mlir.OutputType.LINALG_ON_TENSORS)

with open("linalg_matmul_64x64.mlir", "w") as fout:
    fout.write(matmul_compiled.operation.get_asm())


# core vectorization analysis
x = torch.randn((1,1), dtype=torch.float32)
y = torch.randn((1,4), dtype=torch.float32)

matmul_compiled = torch_mlir.compile(
    matmul(), [x, y], torch_mlir.OutputType.LINALG_ON_TENSORS)
with open("linalg_matmul_vec_core.mlir", "w") as fout:
    fout.write(matmul_compiled.operation.get_asm())
