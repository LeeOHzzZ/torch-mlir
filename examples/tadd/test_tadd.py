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


# N = 4
shape = (80, 80)


# a very simple add module
class TAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y, z):
        return torch.add(torch.add(x, y), z)

x = torch.randn(shape, dtype=torch.float32)
y = torch.randn(shape, dtype=torch.float32)
z = torch.randn(shape, dtype=torch.float32)

compiled_torch = torch_mlir.compile(TAdd(), [x, y, z], torch_mlir.OutputType.TORCH)
compiled_tosa = torch_mlir.compile(TAdd(), [x, y, z], torch_mlir.OutputType.TOSA)
# compiled_torch = torch_mlir.compile(torch.add(), [x, y], torch_mlir.OutputType.TORCH)

print("torch mlir:\n", compiled_torch)
with open("torch_tadd.mlir", "w") as f:
    f.write(compiled_torch.operation.get_asm())

with open("tosa_tadd.mlir", "w") as f:
    f.write(compiled_tosa.operation.get_asm())

compiled_linalg = torch_mlir.compile(TAdd(), [x, y, z], torch_mlir.OutputType.LINALG_ON_TENSORS)
print("linalg mlir:\n", compiled_linalg)
with open("linalg_tadd.mlir", "w") as f:
    f.write(compiled_linalg.operation.get_asm())

backend = RefBackendLinalgOnTensorsBackend()
compiled_backend = backend.compile(compiled_linalg)

print()

# asm_for_error_report = compiled.operation.get_asm(
#     large_elements_limit=10, enable_debug_info=True
# )
# print(asm_for_error_report)
# with open("test_vadd_asm.mlir", "w") as f:
#     f.write(asm_for_error_report)
