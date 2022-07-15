#map = affine_map<(d0) -> (d0)>
module attributes {torch.debug_module_name = "VAdd"} {
  func.func @forward(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %c1_i64 = arith.constant 1 : i64
    %0 = linalg.init_tensor [4] : tensor<4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%0 : tensor<4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %2 = arith.sitofp %c1_i64 : i64 to f32
      %3 = arith.mulf %arg3, %2 : f32
      %4 = arith.addf %arg2, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
