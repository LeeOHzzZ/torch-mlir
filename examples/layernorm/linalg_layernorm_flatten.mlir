#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "LayerNorm"} {
  func.func @forward(%arg0: tensor<2x8640xf32>) -> tensor<2x8640xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c8640_i64 = arith.constant 8640 : i64
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<8640xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<8640xf32>
    %0 = arith.cmpi eq, %c8640_i64, %c8640_i64 : i64
    cf.assert %0, "mismatching contracting dimension"
    cf.assert %0, "mismatching contracting dimension"
    cf.assert %0, "mismatching contracting dimension"
    %1 = arith.sitofp %c8640_i64 : i64 to f32
    %2 = linalg.init_tensor [2] : tensor<2xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2xf32>) -> tensor<2xf32>
    %4 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<2x8640xf32>) outs(%3 : tensor<2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %12 = arith.addf %arg2, %arg1 : f32
      linalg.yield %12 : f32
    } -> tensor<2xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%4 : tensor<2xf32>) outs(%2 : tensor<2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %12 = arith.divf %arg1, %1 : f32
      linalg.yield %12 : f32
    } -> tensor<2xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2xf32>) -> tensor<2xf32>
    %7 = linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0, %5 : tensor<2x8640xf32>, tensor<2xf32>) outs(%6 : tensor<2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %12 = arith.subf %arg1, %arg2 : f32
      %13 = arith.mulf %12, %12 : f32
      %14 = arith.addf %arg3, %13 : f32
      linalg.yield %14 : f32
    } -> tensor<2xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%7 : tensor<2xf32>) outs(%2 : tensor<2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %12 = arith.divf %arg1, %1 : f32
      linalg.yield %12 : f32
    } -> tensor<2xf32>
    %9 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%8 : tensor<2xf32>) outs(%2 : tensor<2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %12 = arith.truncf %cst_0 : f64 to f32
      %13 = arith.addf %arg1, %12 : f32
      %14 = math.rsqrt %13 : f32
      linalg.yield %14 : f32
    } -> tensor<2xf32>
    %10 = linalg.init_tensor [2, 8640] : tensor<2x8640xf32>
    %11 = linalg.generic {indexing_maps = [#map0, #map1, #map1, #map3, #map3, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %5, %9, %cst_1, %cst_2 : tensor<2x8640xf32>, tensor<2xf32>, tensor<2xf32>, tensor<8640xf32>, tensor<8640xf32>) outs(%10 : tensor<2x8640xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
      %12 = arith.subf %arg1, %arg2 : f32
      %13 = arith.mulf %12, %arg3 : f32
      %14 = arith.mulf %13, %arg4 : f32
      %15 = arith.addf %14, %arg5 : f32
      linalg.yield %15 : f32
    } -> tensor<2x8640xf32>
    return %11 : tensor<2x8640xf32>
  }
}
