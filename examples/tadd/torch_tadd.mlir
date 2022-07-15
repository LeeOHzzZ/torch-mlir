module attributes {torch.debug_module_name = "TAdd"} {
  func.func @forward(%arg0: !torch.vtensor<[80,80],f32>, %arg1: !torch.vtensor<[80,80],f32>, %arg2: !torch.vtensor<[80,80],f32>) -> !torch.vtensor<[80,80],f32> {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[80,80],f32>, !torch.vtensor<[80,80],f32>, !torch.int -> !torch.vtensor<[80,80],f32>
    %1 = torch.aten.add.Tensor %0, %arg2, %int1 : !torch.vtensor<[80,80],f32>, !torch.vtensor<[80,80],f32>, !torch.int -> !torch.vtensor<[80,80],f32>
    return %1 : !torch.vtensor<[80,80],f32>
  }
}
