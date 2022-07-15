module attributes {torch.debug_module_name = "MaxPool2d"} {
  func.func @forward(%arg0: tensor<1x8x32x32xf32>) -> tensor<1x8x16x16xf32> {
    %cst = arith.constant -3.40282347E+38 : f32
    %0 = linalg.init_tensor [1, 8, 16, 16] : tensor<1x8x16x16xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %2 = linalg.init_tensor [2, 2] : tensor<2x2xf32>
    %3 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %2 : tensor<1x8x32x32xf32>, tensor<2x2xf32>) outs(%1 : tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    return %3 : tensor<1x8x16x16xf32>
  }
}
