#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module attributes {torch.debug_module_name = "Softmax"} {
  func.func @forward(%arg0: tensor<2x3888xf32>) -> tensor<2x3888xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant -3.40282347E+38 : f32
    %c0_i64 = arith.constant 0 : i64
    %0 = linalg.init_tensor [2, 1] : tensor<2x1xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<2x1xi64>) -> tensor<2x1xi64>
    %2 = linalg.init_tensor [2, 1] : tensor<2x1xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<2x1xf32>) -> tensor<2x1xf32>
    %4:2 = linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<2x3888xf32>) outs(%3, %1 : tensor<2x1xf32>, tensor<2x1xi64>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64):
      %11 = linalg.index 1 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = arith.cmpf ogt, %arg1, %arg2 : f32
      %14 = arith.select %13, %arg1, %arg2 : f32
      %15 = arith.select %13, %12, %arg3 : i64
      linalg.yield %14, %15 : f32, i64
    } -> (tensor<2x1xf32>, tensor<2x1xi64>)
    %5 = linalg.init_tensor [2, 3888] : tensor<2x3888xf32>
    %6 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %4#0 : tensor<2x3888xf32>, tensor<2x1xf32>) outs(%5 : tensor<2x3888xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %11 = arith.truncf %cst_0 : f64 to f32
      %12 = arith.mulf %arg2, %11 : f32
      %13 = arith.subf %arg1, %12 : f32
      linalg.yield %13 : f32
    } -> tensor<2x3888xf32>
    %7 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<2x3888xf32>) outs(%5 : tensor<2x3888xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %11 = math.exp %arg1 : f32
      linalg.yield %11 : f32
    } -> tensor<2x3888xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x1xf32>) -> tensor<2x1xf32>
    %9 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<2x3888xf32>) outs(%8 : tensor<2x1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %11 = arith.addf %arg1, %arg2 : f32
      linalg.yield %11 : f32
    } -> tensor<2x1xf32>
    %10 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%7, %9 : tensor<2x3888xf32>, tensor<2x1xf32>) outs(%5 : tensor<2x3888xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %11 = arith.divf %arg1, %arg2 : f32
      linalg.yield %11 : f32
    } -> tensor<2x3888xf32>
    return %10 : tensor<2x3888xf32>
  }
}
