""" This file intends to test offloading a simple vector add operator
    through the compiler pipeline
"""

import torch

import torch_mlir
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

import numpy as np


N = 4


# a very simple add module
class VAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return torch.add(x, y)

x = torch.randn(4)
y = torch.randn(4)

compiled = torch_mlir.compile(VAdd(), [x, y], torch_mlir.OutputType.LINALG_ON_TENSORS)
print(compiled)

asm_for_error_report = compiled.operation.get_asm(
    large_elements_limit=10, enable_debug_info=True
)
print(asm_for_error_report)
with open("test_vadd_asm.mlir", "w") as f:
    f.write(asm_for_error_report)