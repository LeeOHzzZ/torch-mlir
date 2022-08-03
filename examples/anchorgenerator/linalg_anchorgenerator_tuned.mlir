#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {torch.debug_module_name = "DefaultAnchorGenerator"} {
  func.func @forward(%arg0: tensor<1x256x16x20xf32>) -> tensor<4800x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f64
    %cst_1 = arith.constant dense<4.000000e+01> : tensor<f32>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_3 = arith.constant dense<2.000000e+00> : tensor<f32>
    %cst_4 = arith.constant dense<[[-22.6274166, -11.3137083, 22.6274166, 11.3137083], [-1.600000e+01, -1.600000e+01, 1.600000e+01, 1.600000e+01], [-11.3137083, -22.6274166, 11.3137083, 22.6274166], [-45.2548332, -22.6274166, 45.2548332, 22.6274166], [-3.200000e+01, -3.200000e+01, 3.200000e+01, 3.200000e+01], [-22.6274166, -45.2548332, 22.6274166, 45.2548332], [-67.8822479, -33.941124, 67.8822479, 33.941124], [-4.800000e+01, -4.800000e+01, 4.800000e+01, 4.800000e+01], [-33.941124, -67.8822479, 33.941124, 67.8822479], [-90.5096664, -45.2548332, 90.5096664, 45.2548332], [-6.400000e+01, -6.400000e+01, 6.400000e+01, 6.400000e+01], [-45.2548332, -90.5096664, 45.2548332, 90.5096664], [-113.137085, -56.5685425, 113.137085, 56.5685425], [-8.000000e+01, -8.000000e+01, 8.000000e+01, 8.000000e+01], [-56.5685425, -113.137085, 56.5685425, 113.137085]]> : tensor<15x4xf32>
    %0 = arith.truncf %cst_0 : f64 to f32
    %1 = linalg.init_tensor [320] : tensor<320xf32>
    %2 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%1 : tensor<320xf32>) {
    ^bb0(%arg1: f32):
      %25 = linalg.index 0 : index
      %26 = arith.index_cast %25 : index to i64
      %27 = arith.sitofp %26 : i64 to f32
      %28 = arith.mulf %0, %27 : f32
      %29 = arith.addf %28, %cst : f32
      linalg.yield %29 : f32
    } -> tensor<320xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel"]} ins(%2, %cst_1 : tensor<320xf32>, tensor<f32>) outs(%1 : tensor<320xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %25 = arith.divf %arg1, %arg2 : f32
      linalg.yield %25 : f32
    } -> tensor<320xf32>
    %4 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%3 : tensor<320xf32>) outs(%1 : tensor<320xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %25 = math.floor %arg1 : f32
      linalg.yield %25 : f32
    } -> tensor<320xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel"]} ins(%4, %cst_1 : tensor<320xf32>, tensor<f32>) outs(%1 : tensor<320xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %25 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %25 : f32
    } -> tensor<320xf32>
    %6 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%2, %5 : tensor<320xf32>, tensor<320xf32>) outs(%1 : tensor<320xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %25 = arith.subf %arg1, %arg2 : f32
      linalg.yield %25 : f32
    } -> tensor<320xf32>
    %7 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel"]} ins(%6, %cst_2 : tensor<320xf32>, tensor<f32>) outs(%1 : tensor<320xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %25 = arith.addf %arg1, %arg2 : f32
      linalg.yield %25 : f32
    } -> tensor<320xf32>
    %8 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%1 : tensor<320xf32>) {
    ^bb0(%arg1: f32):
      %25 = linalg.index 0 : index
      %26 = arith.index_cast %25 : index to i64
      %27 = arith.sitofp %26 : i64 to f32
      %28 = arith.mulf %0, %27 : f32
      %29 = arith.addf %28, %cst : f32
      linalg.yield %29 : f32
    } -> tensor<320xf32>
    %9 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel"]} ins(%8, %cst_1 : tensor<320xf32>, tensor<f32>) outs(%1 : tensor<320xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %25 = arith.divf %arg1, %arg2 : f32
      linalg.yield %25 : f32
    } -> tensor<320xf32>
    %10 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%9 : tensor<320xf32>) outs(%1 : tensor<320xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %25 = math.floor %arg1 : f32
      linalg.yield %25 : f32
    } -> tensor<320xf32>
    %11 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel"]} ins(%10, %cst_3 : tensor<320xf32>, tensor<f32>) outs(%1 : tensor<320xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %25 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %25 : f32
    } -> tensor<320xf32>
    %12 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel"]} ins(%11, %cst_2 : tensor<320xf32>, tensor<f32>) outs(%1 : tensor<320xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %25 = arith.addf %arg1, %arg2 : f32
      linalg.yield %25 : f32
    } -> tensor<320xf32>
    %13 = tensor.expand_shape %7 [[0, 1]] : tensor<320xf32> into tensor<320x1xf32>
    %14 = tensor.expand_shape %12 [[0, 1]] : tensor<320xf32> into tensor<320x1xf32>
    %15 = linalg.init_tensor [320, 4] : tensor<320x4xf32>
    %16 = tensor.insert_slice %13 into %15[0, 0] [320, 1] [1, 1] : tensor<320x1xf32> into tensor<320x4xf32>
    %17 = tensor.insert_slice %14 into %16[0, 1] [320, 1] [1, 1] : tensor<320x1xf32> into tensor<320x4xf32>
    %18 = tensor.insert_slice %13 into %17[0, 2] [320, 1] [1, 1] : tensor<320x1xf32> into tensor<320x4xf32>
    %19 = tensor.insert_slice %14 into %18[0, 3] [320, 1] [1, 1] : tensor<320x1xf32> into tensor<320x4xf32>
    %20 = tensor.expand_shape %19 [[0], [1, 2]] : tensor<320x4xf32> into tensor<320x1x4xf32>
    %21 = tensor.expand_shape %cst_4 [[0, 1], [2]] : tensor<15x4xf32> into tensor<1x15x4xf32>
    %22 = linalg.init_tensor [320, 15, 4] : tensor<320x15x4xf32>
    %23 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20, %21 : tensor<320x1x4xf32>, tensor<1x15x4xf32>) outs(%22 : tensor<320x15x4xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %25 = arith.addf %arg1, %arg2 : f32
      linalg.yield %25 : f32
    } -> tensor<320x15x4xf32>
    %24 = tensor.collapse_shape %23 [[0, 1], [2]] : tensor<320x15x4xf32> into tensor<4800x4xf32>
    return %24 : tensor<4800x4xf32>
  }
}

