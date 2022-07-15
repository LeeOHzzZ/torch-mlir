#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
module attributes {torch.debug_module_name = "LayerNorm"} {
  func.func @forward(%arg0: tensor<2x3x36x80xf32>) -> tensor<2x3x36x80xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c80_i64 = arith.constant 80 : i64
    %c36_i64 = arith.constant 36 : i64
    %c3_i64 = arith.constant 3 : i64
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<3x36x80xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<3x36x80xf32>
    %0 = arith.cmpi eq, %c3_i64, %c3_i64 : i64
    cf.assert %0, "mismatching contracting dimension"
    cf.assert %0, "mismatching contracting dimension"
    cf.assert %0, "mismatching contracting dimension"
    %1 = arith.cmpi eq, %c36_i64, %c36_i64 : i64
    cf.assert %1, "mismatching contracting dimension"
    cf.assert %1, "mismatching contracting dimension"
    cf.assert %1, "mismatching contracting dimension"
    %2 = arith.cmpi eq, %c80_i64, %c80_i64 : i64
    cf.assert %2, "mismatching contracting dimension"
    cf.assert %2, "mismatching contracting dimension"
    cf.assert %2, "mismatching contracting dimension"
    %3 = arith.muli %c3_i64, %c36_i64 : i64
    %4 = arith.muli %3, %c80_i64 : i64
    %5 = arith.sitofp %4 : i64 to f32
    %6 = linalg.init_tensor [2] : tensor<2xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<2xf32>) -> tensor<2xf32>
    %8 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction", "reduction", "reduction"]} ins(%arg0 : tensor<2x3x36x80xf32>) outs(%7 : tensor<2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %16 = arith.addf %arg2, %arg1 : f32
      linalg.yield %16 : f32
    } -> tensor<2xf32>
    %9 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%8 : tensor<2xf32>) outs(%6 : tensor<2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %16 = arith.divf %arg1, %5 : f32
      linalg.yield %16 : f32
    } -> tensor<2xf32>
    %10 = linalg.fill ins(%cst : f32) outs(%6 : tensor<2xf32>) -> tensor<2xf32>
    %11 = linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %9 : tensor<2x3x36x80xf32>, tensor<2xf32>) outs(%10 : tensor<2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %16 = arith.subf %arg1, %arg2 : f32
      %17 = arith.mulf %16, %16 : f32
      %18 = arith.addf %arg3, %17 : f32
      linalg.yield %18 : f32
    } -> tensor<2xf32>
    %12 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%11 : tensor<2xf32>) outs(%6 : tensor<2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %16 = arith.divf %arg1, %5 : f32
      linalg.yield %16 : f32
    } -> tensor<2xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%12 : tensor<2xf32>) outs(%6 : tensor<2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %16 = arith.truncf %cst_0 : f64 to f32
      %17 = arith.addf %arg1, %16 : f32
      %18 = math.rsqrt %17 : f32
      linalg.yield %18 : f32
    } -> tensor<2xf32>
    %14 = linalg.init_tensor [2, 3, 36, 80] : tensor<2x3x36x80xf32>
    %15 = linalg.generic {indexing_maps = [#map0, #map1, #map1, #map3, #map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %9, %13, %cst_1, %cst_2 : tensor<2x3x36x80xf32>, tensor<2xf32>, tensor<2xf32>, tensor<3x36x80xf32>, tensor<3x36x80xf32>) outs(%14 : tensor<2x3x36x80xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
      %16 = arith.subf %arg1, %arg2 : f32
      %17 = arith.mulf %16, %arg3 : f32
      %18 = arith.mulf %17, %arg4 : f32
      %19 = arith.addf %18, %arg5 : f32
      linalg.yield %19 : f32
    } -> tensor<2x3x36x80xf32>
    return %15 : tensor<2x3x36x80xf32>
  }
}
