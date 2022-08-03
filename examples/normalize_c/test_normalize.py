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


class turing_normalize(nn.Module):
    def forward(self, x):
        slice_size = x.shape[0] * x.shape[1] * x.shape[2]
        mean = torch.sum(x, (0,1,2))
        mean = mean / slice_size
        y = x - mean
        y = y * y
        std = torch.sum(y, (0,1,2))
        std = std / slice_size
        inv_std = torch.rsqrt(std)
        out = (x - mean) * inv_std

        return out


# NHWC format
N, H, W, C = 1, 8, 8, 4
shape = (N, H, W, C)

x = torch.randn(shape, dtype=torch.float32)

compiled = torch_mlir.compile(
    turing_normalize(), [x], torch_mlir.OutputType.LINALG_ON_TENSORS
)
with open("linalg_normalize.mlir", "w") as fout:
    fout.write(compiled.operation.get_asm())


N, H, W, C = 1, 64, 64, 16
shape = (N, H, W, C)

x = torch.randn(shape, dtype=torch.float32)

compiled = torch_mlir.compile(
    turing_normalize(), [x], torch_mlir.OutputType.LINALG_ON_TENSORS
)
with open("linalg_normalize_large.mlir", "w") as fout:
    fout.write(compiled.operation.get_asm())


N, H, W, C = 3, 8, 8, 4
shape = (N, H, W, C)

x = torch.randn(shape, dtype=torch.float32)

compiled = torch_mlir.compile(
    turing_normalize(), [x], torch_mlir.OutputType.LINALG_ON_TENSORS
)
with open("linalg_normalize_3b.mlir", "w") as fout:
    fout.write(compiled.operation.get_asm())