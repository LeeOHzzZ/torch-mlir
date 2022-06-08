#loc0 = loc(unknown)
#loc1 = loc("test_vadd.py":26:15)
#map = affine_map<(d0) -> (d0)>
module attributes {torch.debug_module_name = "VAdd"} {
  func.func @forward(%arg0: tensor<4xf32> loc(unknown), %arg1: tensor<4xf32> loc(unknown)) -> tensor<4xf32> {
    %c1_i64 = arith.constant 1 : i64 loc(#loc0)
    %0 = linalg.init_tensor [4] : tensor<4xf32> loc(#loc1)
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%0 : tensor<4xf32>) {
    ^bb0(%arg2: f32 loc("test_vadd.py":26:15), %arg3: f32 loc("test_vadd.py":26:15), %arg4: f32 loc("test_vadd.py":26:15)):
      %2 = arith.sitofp %c1_i64 : i64 to f32 loc(#loc1)
      %3 = arith.mulf %arg3, %2 : f32 loc(#loc1)
      %4 = arith.addf %arg2, %3 : f32 loc(#loc1)
      linalg.yield %4 : f32 loc(#loc1)
    } -> tensor<4xf32> loc(#loc1)
    return %1 : tensor<4xf32> loc(#loc0)
  } loc(#loc0)
} loc(#loc0)
