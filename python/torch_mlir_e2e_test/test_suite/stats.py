# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================

class MeanModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x)


@register_test_case(module_factory=lambda: MeanModule())
def MeanModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))

# ==============================================================================

class MeanDynamicSizesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x)


@register_test_case(module_factory=lambda: MeanDynamicSizesModule())
def MeanDynamicSizesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))

# ==============================================================================

class MeanDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, dtype=torch.float32)


@register_test_case(module_factory=lambda: MeanDtypeModule())
def MeanDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))

# ==============================================================================

class MeanDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, (0, 2))


@register_test_case(module_factory=lambda: MeanDimModule())
def MeanDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))

# ==============================================================================

class MeanDimDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, 0, dtype=torch.float32)


@register_test_case(module_factory=lambda: MeanDimDtypeModule())
def MeanDimDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))

# ==============================================================================

class MeanDimKeepdimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, (1, 2), keepdim=True)


@register_test_case(module_factory=lambda: MeanDimKeepdimModule())
def MeanDimKeepdimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class MeanDimAllReduceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, (0, 1, 2))


@register_test_case(module_factory=lambda: MeanDimAllReduceModule())
def MeanDimAllReduceModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class MeanDimAllReduceKeepdimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, (0, 1, 2), keepdim=True)


@register_test_case(module_factory=lambda: MeanDimAllReduceKeepdimModule())
def MeanDimAllReduceKeepdimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class MeanDimNegativeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, (-1, 1))


@register_test_case(module_factory=lambda: MeanDimNegativeModule())
def MeanDimNegativeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class VarUnbiasedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, unbiased=True)

@register_test_case(module_factory=lambda: VarUnbiasedModule())
def VarUnbiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))

# ==============================================================================

class VarBiasedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, unbiased=False)

@register_test_case(module_factory=lambda: VarBiasedModule())
def VarBiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))

# ==============================================================================

class StdUnbiasedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.std(x, unbiased=True)

@register_test_case(module_factory=lambda: StdUnbiasedModule())
def StdUnbiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))

# ==============================================================================

class StdBiasedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.std(x, unbiased=False)

@register_test_case(module_factory=lambda: StdBiasedModule())
def StdBiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


# ==============================================================================


class VarDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=(0, 2), keepdim=True)


@register_test_case(module_factory=lambda: VarDimModule())
def VarDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))


# ==============================================================================


class VarDimUnbiasedModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=(0, 2), unbiased=True, keepdim=True)


@register_test_case(module_factory=lambda: VarDimUnbiasedModule())
def VarDimUnbiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))


# ==============================================================================


class VarDimBiasedModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=0, unbiased=False, keepdim=True)


@register_test_case(module_factory=lambda: VarDimBiasedModule())
def VarDimBiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))


# ==============================================================================


class VarDimSingleDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=0, keepdim=True)


@register_test_case(module_factory=lambda: VarDimSingleDimModule())
def VarDimSingleDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))


# ==============================================================================


class VarDimMultiDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=[0, 2], keepdim=False)


@register_test_case(module_factory=lambda: VarDimMultiDimModule())
def VarDimMultiDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))


# ==============================================================================


class VarDimAllDimReduceModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=(0, 1, 2), keepdim=True)


@register_test_case(module_factory=lambda: VarDimAllDimReduceModule())
def VarDimAllDimReduceModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class VarDimNegativeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=(-1, 1), keepdim=True)


@register_test_case(module_factory=lambda: VarDimNegativeModule())
def VarDimNegativeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class VarDimKeepDimFalseModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=(0, 1, 2), keepdim=False)


@register_test_case(module_factory=lambda: VarDimKeepDimFalseModule())
def VarDimKeepDimFalseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))
