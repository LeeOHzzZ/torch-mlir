#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "LayerNorm_Linear"} {
  func.func @forward(%arg0: tensor<2x8640xf32>, %arg1: tensor<8640xf32>, %arg2: tensor<8640xf32>) -> tensor<2x8640xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c8640_i64 = arith.constant 8640 : i64
    %c1_i64 = arith.constant 1 : i64
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
    ^bb0(%arg3: f32, %arg4: f32):
      %14 = arith.addf %arg4, %arg3 : f32
      linalg.yield %14 : f32
    } -> tensor<2xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%4 : tensor<2xf32>) outs(%2 : tensor<2xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %14 = arith.divf %arg3, %1 : f32
      linalg.yield %14 : f32
    } -> tensor<2xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2xf32>) -> tensor<2xf32>
    %7 = linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0, %5 : tensor<2x8640xf32>, tensor<2xf32>) outs(%6 : tensor<2xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %14 = arith.subf %arg3, %arg4 : f32
      %15 = arith.mulf %14, %14 : f32
      %16 = arith.addf %arg5, %15 : f32
      linalg.yield %16 : f32
    } -> tensor<2xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%7 : tensor<2xf32>) outs(%2 : tensor<2xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %14 = arith.divf %arg3, %1 : f32
      linalg.yield %14 : f32
    } -> tensor<2xf32>
    %9 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%8 : tensor<2xf32>) outs(%2 : tensor<2xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %14 = arith.truncf %cst_0 : f64 to f32
      %15 = arith.addf %arg3, %14 : f32
      %16 = math.rsqrt %15 : f32
      linalg.yield %16 : f32
    } -> tensor<2xf32>
    %10 = linalg.init_tensor [2, 8640] : tensor<2x8640xf32>
    %11 = linalg.generic {indexing_maps = [#map0, #map1, #map1, #map3, #map3, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %5, %9, %cst_1, %cst_2 : tensor<2x8640xf32>, tensor<2xf32>, tensor<2xf32>, tensor<8640xf32>, tensor<8640xf32>) outs(%10 : tensor<2x8640xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):
      %14 = arith.subf %arg3, %arg4 : f32
      %15 = arith.mulf %14, %arg5 : f32
      %16 = arith.mulf %15, %arg6 : f32
      %17 = arith.addf %16, %arg7 : f32
      linalg.yield %17 : f32
    } -> tensor<2x8640xf32>
    %12 = linalg.generic {indexing_maps = [#map0, #map3, #map0], iterator_types = ["parallel", "parallel"]} ins(%11, %arg1 : tensor<2x8640xf32>, tensor<8640xf32>) outs(%10 : tensor<2x8640xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %14 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %14 : f32
    } -> tensor<2x8640xf32>
    %13 = linalg.generic {indexing_maps = [#map0, #map3, #map0], iterator_types = ["parallel", "parallel"]} ins(%12, %arg2 : tensor<2x8640xf32>, tensor<8640xf32>) outs(%10 : tensor<2x8640xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %14 = arith.sitofp %c1_i64 : i64 to f32
      %15 = arith.mulf %arg4, %14 : f32
      %16 = arith.addf %arg3, %15 : f32
      linalg.yield %16 : f32
    } -> tensor<2x8640xf32>
    return %13 : tensor<2x8640xf32>
  }
}
