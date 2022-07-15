import torch
import torch_mlir

from torch_mlir.compiler_utils import run_pipeline_with_repro_report
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import LOWERING_PIPELINE


class Conv2D(torch.nn.Conv2d):
    def __init__(self, in_chans, out_chans, kernel_size):
        super().__init__(in_chans, out_chans, kernel_size)

    def forward(self, weights, bias, x):
        self.weight = torch.nn.Parameter(weights, requires_grad=False)
        self.bias = torch.nn.Parameter(bias, requires_grad=False)
        return super().forward(x)
        

# covn2d = torch.nn.Conv2d(3, 3, 3, 1, 1)
conv2d = Conv2D(3, 3, 3)
print(conv2d(*[
            torch.randn((3, 3, 3, 3)),
            torch.randn((3,)),
            torch.randn((1, 3, 32, 32))
        ]))
with torch.no_grad():

    # compiled_linalg = torch_mlir.compile(
    #     conv2d,
    #     [
    #         torch.randn((3, 3, 3, 3)),
    #         torch.randn((3,)),
    #         torch.randn((1, 3, 32, 32))
    #     ],
    #     torch_mlir.OutputType.LINALG_ON_TENSORS
    # )

    # compiled_tosa = torch_mlir.compile(
    #     conv2d,
    #     [
    #         torch.randn((3, 3, 3, 3)),
    #         torch.randn((3,)),
    #         torch.randn((1, 3, 32, 32))
    #     ],
    #     torch_mlir.OutputType.TOSA
    # )

    compiled_linalg = torch_mlir.compile(
        torch.nn.Conv2d(3, 3, 3),
        torch.randn(3, 3, 32, 32),
        torch_mlir.OutputType.LINALG_ON_TENSORS
    )
    compiled_tosa = torch_mlir.compile(
        torch.nn.Conv2d(3, 3, 3),
        torch.randn(3, 3, 32, 32),
        torch_mlir.OutputType.TOSA
    )

    # LOWERING_PIPELINE = ",".join([
    #     "one"
    # ])
    print(compiled_tosa)
    print(compiled_linalg)
    with open("linalg_conv2d.mlir", "w") as f:
        f.write(compiled_linalg.operation.get_asm())
    with open("tosa_conv2d.mlir", "w") as f:
        f.write(compiled_tosa.operation.get_asm())
