#map0 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "dequantize"} {
  func.func @forward(%arg0: tensor<1x3x32x32xi32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1x3x32x32xf32> {
    %0 = linalg.init_tensor [1, 3, 32, 32] : tensor<1x3x32x32xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg2 : tensor<1x3x32x32xi32>, tensor<1xf32>) outs(%0 : tensor<1x3x32x32xf32>) {
    ^bb0(%arg3: i32, %arg4: f32, %arg5: f32):
      %3 = arith.sitofp %arg3 : i32 to f32
      %4 = arith.subf %3, %arg4 : f32
      linalg.yield %4 : f32
    } -> tensor<1x3x32x32xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %arg1 : tensor<1x3x32x32xf32>, tensor<1xf32>) outs(%0 : tensor<1x3x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %3 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %3 : f32
    } -> tensor<1x3x32x32xf32>
    return %2 : tensor<1x3x32x32xf32>
  }
}
