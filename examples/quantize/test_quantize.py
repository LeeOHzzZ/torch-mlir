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


class quantization(nn.Module):
    def forward(self, x, params):
        invScale, zeroPoint, min, max = params[0], params[1], params[2], params[3], 
        # xScaled = torch.round(x * 1 + 0.5)
        out = torch.clamp(x, 0, 255)

        return out

class round(nn.Module):
    def forward(self, x):
        return torch.round(x)


# NHWC format
N, C, H, W = 1, 3, 3, 3
shape = (N, C, H, W)

invScale = 1
zeroPoint = 2
min = -235
max = 285
params = torch.tensor([invScale, zeroPoint, min, max], dtype=torch.float32)

x = 500 * torch.randn(shape, dtype=torch.float32)
# print(x)

# print(quantization()(x, params))

quantize = quantization()
quantize.eval()

with torch.no_grad():
    # compiled = torch_mlir.compile(
    #     round(), x
    # )
    compiled = torch_mlir.compile(
        quantize, [x, params], torch_mlir.OutputType.LINALG_ON_TENSORS
    )
# with open("linalg_quantize.mlir", "w") as fout:
#     fout.write(compiled.operation.get_asm())
