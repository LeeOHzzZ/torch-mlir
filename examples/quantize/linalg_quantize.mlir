module attributes {torch.debug_module_name = "quantization"} {
  func.func @forward(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
    %0 = tensor.extract_slice %arg1[2] [1] [1] : tensor<4xf32> to tensor<1xf32>
    %1 = tensor.collapse_shape %0 [] : tensor<1xf32> into tensor<f32>
    return %1 : tensor<f32>
  }
}
