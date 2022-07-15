#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "TAdd"} {
  func.func @forward(%arg0: tensor<80x80xf32>, %arg1: tensor<80x80xf32>, %arg2: tensor<80x80xf32>) -> tensor<80x80xf32> {
    %c1_i64 = arith.constant 1 : i64
    %0 = linalg.init_tensor [80, 80] : tensor<80x80xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<80x80xf32>, tensor<80x80xf32>) outs(%0 : tensor<80x80xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %3 = arith.sitofp %c1_i64 : i64 to f32
      %4 = arith.mulf %arg4, %3 : f32
      %5 = arith.addf %arg3, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<80x80xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %arg2 : tensor<80x80xf32>, tensor<80x80xf32>) outs(%0 : tensor<80x80xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %3 = arith.sitofp %c1_i64 : i64 to f32
      %4 = arith.mulf %arg4, %3 : f32
      %5 = arith.addf %arg3, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<80x80xf32>
    return %2 : tensor<80x80xf32>
  }
}
