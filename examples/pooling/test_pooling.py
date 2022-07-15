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


shape = (1, 8, 32, 32)
kernel_size = (2,2)
stride = (2,2)


x = torch.randn(shape, dtype=torch.float32)

maxp_compiled = torch_mlir.compile(nn.MaxPool2d(kernel_size, stride),
                                   [x], torch_mlir.OutputType.LINALG_ON_TENSORS)
meanp_compiled = torch_mlir.compile(nn.AvgPool2d(kernel_size, stride),
                                    [x], torch_mlir.OutputType.LINALG_ON_TENSORS)

with open("linalg_maxp2d.mlir", "w") as fout:
    fout.write(maxp_compiled.operation.get_asm())
with open("linalg_avgp2d.mlir", "w") as fout:
    fout.write(meanp_compiled.operation.get_asm())

