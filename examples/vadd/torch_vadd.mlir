module attributes {torch.debug_module_name = "VAdd"} {
  func.func @forward(%arg0: !torch.vtensor<[4],f32>, %arg1: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4],f32>
    return %0 : !torch.vtensor<[4],f32>
  }
}
