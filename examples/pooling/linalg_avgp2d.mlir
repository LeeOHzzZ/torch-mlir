#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "AvgPool2d"} {
  func.func @forward(%arg0: tensor<1x8x32x32xf32>) -> tensor<1x8x16x16xf32> {
    %cst = arith.constant 4.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [1, 8, 16, 16] : tensor<1x8x16x16xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %2 = linalg.init_tensor [2, 2] : tensor<2x2xf32>
    %3 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %2 : tensor<1x8x32x32xf32>, tensor<2x2xf32>) outs(%1 : tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x8x16x16xf32>) outs(%0 : tensor<1x8x16x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %5 = arith.divf %arg1, %cst : f32
      linalg.yield %5 : f32
    } -> tensor<1x8x16x16xf32>
    return %4 : tensor<1x8x16x16xf32>
  }
}
