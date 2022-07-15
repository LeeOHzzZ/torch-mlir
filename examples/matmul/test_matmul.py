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


shape = (1024, 1024)

class matmul(torch.nn.Module):
    
    def forward(self, x, y):
        return torch.matmul(x, y)

x = torch.randn(shape, dtype=torch.float32)
y = torch.randn(shape, dtype=torch.float32)

matmul_compiled = torch_mlir.compile(matmul(),
                                   [x, y], torch_mlir.OutputType.LINALG_ON_TENSORS)

with open("linalg_matmul.mlir", "w") as fout:
    fout.write(matmul_compiled.operation.get_asm())

