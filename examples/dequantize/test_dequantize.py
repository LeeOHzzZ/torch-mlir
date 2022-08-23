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


class dequantize(nn.Module):
    def forward(self, x, scale, zeropoint):
        # x = x.to(torch.int32)
        out = (x - zeropoint) * scale
        return out

# NHWC format
N, C, H, W = 1, 3, 32, 32
shape = (N, C, H, W)

scale = torch.tensor([0.51], dtype=torch.float32)
zeroPoint = torch.tensor([-130], dtype=torch.float32)

x = torch.randint(0, 255, shape, dtype=torch.int32)

with torch.no_grad():
    compiled = torch_mlir.compile(
        dequantize(), [x, scale, zeroPoint], torch_mlir.OutputType.LINALG_ON_TENSORS
    )
    with open("linalg_dequantize_1x3x32x32.mlir", "w") as fout:
        fout.write(compiled.operation.get_asm())
