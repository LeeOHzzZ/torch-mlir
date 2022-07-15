// -----// IR Dump After GeneralizeTensorPad //----- //
func.func @forward(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.init_tensor [4] : tensor<4xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%0 : tensor<4xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %2 = arith.addf %arg2, %arg3 : f32
    linalg.yield %2 : f32
  } -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----// IR Dump After SCFBufferize //----- //
func.func @forward(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.init_tensor [4] : tensor<4xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%0 : tensor<4xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %2 = arith.addf %arg2, %arg3 : f32
    linalg.yield %2 : f32
  } -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----// IR Dump After TMTensorBufferize //----- //
func.func @forward(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.init_tensor [4] : tensor<4xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%0 : tensor<4xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %2 = arith.addf %arg2, %arg3 : f32
    linalg.yield %2 : f32
  } -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----// IR Dump After LinalgBufferize //----- //
func.func @forward(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = bufferization.to_memref %arg1 : memref<4xf32>
  %1 = bufferization.to_memref %arg0 : memref<4xf32>
  %c4 = arith.constant 4 : index
  %2 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  %3 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  %4 = bufferization.to_tensor %2 : memref<4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %0 : memref<4xf32>, memref<4xf32>) outs(%3 : memref<4xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %6 = arith.addf %arg2, %arg3 : f32
    linalg.yield %6 : f32
  }
  %5 = bufferization.to_tensor %3 : memref<4xf32>
  return %5 : tensor<4xf32>
}

// -----// IR Dump After FuncBufferize //----- //
#map = affine_map<(d0) -> (d0)>
module attributes {torch.debug_module_name = "VAdd"} {
  func.func @forward(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32> {
    %0 = bufferization.to_tensor %arg1 : memref<4xf32>
    %1 = bufferization.to_tensor %arg0 : memref<4xf32>
    %2 = bufferization.to_memref %0 : memref<4xf32>
    %3 = bufferization.to_memref %1 : memref<4xf32>
    %c4 = arith.constant 4 : index
    %4 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
    %5 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
    %6 = bufferization.to_tensor %4 : memref<4xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %2 : memref<4xf32>, memref<4xf32>) outs(%5 : memref<4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %9 = arith.addf %arg2, %arg3 : f32
      linalg.yield %9 : f32
    }
    %7 = bufferization.to_tensor %5 : memref<4xf32>
    %8 = bufferization.to_memref %7 : memref<4xf32>
    return %8 : memref<4xf32>
  }
}


// -----// IR Dump After ArithmeticBufferize //----- //
#map = affine_map<(d0) -> (d0)>
module attributes {torch.debug_module_name = "VAdd"} {
  func.func @forward(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32> {
    %0 = bufferization.to_tensor %arg1 : memref<4xf32>
    %1 = bufferization.to_tensor %arg0 : memref<4xf32>
    %c4 = arith.constant 4 : index
    %2 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
    %3 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
    %4 = bufferization.to_tensor %2 : memref<4xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<4xf32>, memref<4xf32>) outs(%3 : memref<4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %6 = arith.addf %arg2, %arg3 : f32
      linalg.yield %6 : f32
    }
    %5 = bufferization.to_tensor %3 : memref<4xf32>
    return %3 : memref<4xf32>
  }
}


// -----// IR Dump After TensorBufferize //----- //
func.func @forward(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32> {
  %0 = bufferization.to_tensor %arg1 : memref<4xf32>
  %1 = bufferization.to_tensor %arg0 : memref<4xf32>
  %c4 = arith.constant 4 : index
  %2 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  %3 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  %4 = bufferization.to_tensor %2 : memref<4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<4xf32>, memref<4xf32>) outs(%3 : memref<4xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %6 = arith.addf %arg2, %arg3 : f32
    linalg.yield %6 : f32
  }
  %5 = bufferization.to_tensor %3 : memref<4xf32>
  return %3 : memref<4xf32>
}

// -----// IR Dump After FinalizingBufferize //----- //
func.func @forward(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32> {
  %c4 = arith.constant 4 : index
  %0 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  %1 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<4xf32>, memref<4xf32>) outs(%1 : memref<4xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %2 = arith.addf %arg2, %arg3 : f32
    linalg.yield %2 : f32
  }
  return %1 : memref<4xf32>
}

// -----// IR Dump After MungeCallingConventions //----- //
#map = affine_map<(d0) -> (d0)>
module attributes {torch.debug_module_name = "VAdd"} {
  func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func @forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>) attributes {llvm.emit_c_interface} {
    %0 = memref.cast %arg0 : memref<*xf32> to memref<4xf32>
    %1 = memref.cast %arg1 : memref<*xf32> to memref<4xf32>
    %c4 = arith.constant 4 : index
    %2 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
    %3 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%0, %1 : memref<4xf32>, memref<4xf32>) outs(%3 : memref<4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %5 = arith.addf %arg2, %arg3 : f32
      linalg.yield %5 : f32
    }
    %4 = memref.cast %3 : memref<4xf32> to memref<*xf32>
    call @refbackend_consume_func_return_mrf32(%4) : (memref<*xf32>) -> ()
    return
  }
}


// -----// IR Dump After InsertRngGlobals //----- //
#map = affine_map<(d0) -> (d0)>
module attributes {torch.debug_module_name = "VAdd"} {
  memref.global "private" @global_seed : memref<i64> = dense<0>
  func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func @forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>) attributes {llvm.emit_c_interface} {
    %0 = memref.cast %arg0 : memref<*xf32> to memref<4xf32>
    %1 = memref.cast %arg1 : memref<*xf32> to memref<4xf32>
    %c4 = arith.constant 4 : index
    %2 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
    %3 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%0, %1 : memref<4xf32>, memref<4xf32>) outs(%3 : memref<4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %5 = arith.addf %arg2, %arg3 : f32
      linalg.yield %5 : f32
    }
    %4 = memref.cast %3 : memref<4xf32> to memref<*xf32>
    call @refbackend_consume_func_return_mrf32(%4) : (memref<*xf32>) -> ()
    return
  }
}


// -----// IR Dump After TMTensorToLoops //----- //
func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}

// -----// IR Dump After MungeMemrefCopy //----- //
func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}

// -----// IR Dump After LinalgLowerToLoops //----- //
func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertAffineToStandard //----- //
func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}

// -----// IR Dump After TMTensorToLoops //----- //
func.func @forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>) attributes {llvm.emit_c_interface} {
  %0 = memref.cast %arg0 : memref<*xf32> to memref<4xf32>
  %1 = memref.cast %arg1 : memref<*xf32> to memref<4xf32>
  %2 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<4xf32>, memref<4xf32>) outs(%2 : memref<4xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %4 = arith.addf %arg2, %arg3 : f32
    linalg.yield %4 : f32
  }
  %3 = memref.cast %2 : memref<4xf32> to memref<*xf32>
  call @refbackend_consume_func_return_mrf32(%3) : (memref<*xf32>) -> ()
  return
}

// -----// IR Dump After MungeMemrefCopy //----- //
func.func @forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>) attributes {llvm.emit_c_interface} {
  %0 = memref.cast %arg0 : memref<*xf32> to memref<4xf32>
  %1 = memref.cast %arg1 : memref<*xf32> to memref<4xf32>
  %2 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<4xf32>, memref<4xf32>) outs(%2 : memref<4xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %4 = arith.addf %arg2, %arg3 : f32
    linalg.yield %4 : f32
  }
  %3 = memref.cast %2 : memref<4xf32> to memref<*xf32>
  call @refbackend_consume_func_return_mrf32(%3) : (memref<*xf32>) -> ()
  return
}

// -----// IR Dump After LinalgLowerToLoops //----- //
func.func @forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>) attributes {llvm.emit_c_interface} {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.cast %arg0 : memref<*xf32> to memref<4xf32>
  %1 = memref.cast %arg1 : memref<*xf32> to memref<4xf32>
  %2 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  scf.for %arg2 = %c0 to %c4 step %c1 {
    %4 = memref.load %0[%arg2] : memref<4xf32>
    %5 = memref.load %1[%arg2] : memref<4xf32>
    %6 = arith.addf %4, %5 : f32
    memref.store %6, %2[%arg2] : memref<4xf32>
  }
  %3 = memref.cast %2 : memref<4xf32> to memref<*xf32>
  call @refbackend_consume_func_return_mrf32(%3) : (memref<*xf32>) -> ()
  return
}

// -----// IR Dump After ConvertAffineToStandard //----- //
func.func @forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>) attributes {llvm.emit_c_interface} {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.cast %arg0 : memref<*xf32> to memref<4xf32>
  %1 = memref.cast %arg1 : memref<*xf32> to memref<4xf32>
  %2 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  scf.for %arg2 = %c0 to %c4 step %c1 {
    %4 = memref.load %0[%arg2] : memref<4xf32>
    %5 = memref.load %1[%arg2] : memref<4xf32>
    %6 = arith.addf %4, %5 : f32
    memref.store %6, %2[%arg2] : memref<4xf32>
  }
  %3 = memref.cast %2 : memref<4xf32> to memref<*xf32>
  call @refbackend_consume_func_return_mrf32(%3) : (memref<*xf32>) -> ()
  return
}

// -----// IR Dump After SCFToControlFlow //----- //
module attributes {torch.debug_module_name = "VAdd"} {
  memref.global "private" @global_seed : memref<i64> = dense<0>
  func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func @forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>) attributes {llvm.emit_c_interface} {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.cast %arg0 : memref<*xf32> to memref<4xf32>
    %1 = memref.cast %arg1 : memref<*xf32> to memref<4xf32>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%3: index):  // 2 preds: ^bb0, ^bb2
    %4 = arith.cmpi slt, %3, %c4 : index
    cf.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = memref.load %0[%3] : memref<4xf32>
    %6 = memref.load %1[%3] : memref<4xf32>
    %7 = arith.addf %5, %6 : f32
    memref.store %7, %2[%3] : memref<4xf32>
    %8 = arith.addi %3, %c1 : index
    cf.br ^bb1(%8 : index)
  ^bb3:  // pred: ^bb1
    %9 = memref.cast %2 : memref<4xf32> to memref<*xf32>
    call @refbackend_consume_func_return_mrf32(%9) : (memref<*xf32>) -> ()
    return
  }
}


// -----// IR Dump After ExpandOpsForLLVM //----- //
func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}

// -----// IR Dump After ArithmeticExpandOps //----- //
func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertMathToLLVM //----- //
func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}

// -----// IR Dump After ExpandOpsForLLVM //----- //
func.func @forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>) attributes {llvm.emit_c_interface} {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.cast %arg0 : memref<*xf32> to memref<4xf32>
  %1 = memref.cast %arg1 : memref<*xf32> to memref<4xf32>
  %2 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  cf.br ^bb1(%c0 : index)
^bb1(%3: index):  // 2 preds: ^bb0, ^bb2
  %4 = arith.cmpi slt, %3, %c4 : index
  cf.cond_br %4, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %5 = memref.load %0[%3] : memref<4xf32>
  %6 = memref.load %1[%3] : memref<4xf32>
  %7 = arith.addf %5, %6 : f32
  memref.store %7, %2[%3] : memref<4xf32>
  %8 = arith.addi %3, %c1 : index
  cf.br ^bb1(%8 : index)
^bb3:  // pred: ^bb1
  %9 = memref.cast %2 : memref<4xf32> to memref<*xf32>
  call @refbackend_consume_func_return_mrf32(%9) : (memref<*xf32>) -> ()
  return
}

// -----// IR Dump After ArithmeticExpandOps //----- //
func.func @forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>) attributes {llvm.emit_c_interface} {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.cast %arg0 : memref<*xf32> to memref<4xf32>
  %1 = memref.cast %arg1 : memref<*xf32> to memref<4xf32>
  %2 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  cf.br ^bb1(%c0 : index)
^bb1(%3: index):  // 2 preds: ^bb0, ^bb2
  %4 = arith.cmpi slt, %3, %c4 : index
  cf.cond_br %4, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %5 = memref.load %0[%3] : memref<4xf32>
  %6 = memref.load %1[%3] : memref<4xf32>
  %7 = arith.addf %5, %6 : f32
  memref.store %7, %2[%3] : memref<4xf32>
  %8 = arith.addi %3, %c1 : index
  cf.br ^bb1(%8 : index)
^bb3:  // pred: ^bb1
  %9 = memref.cast %2 : memref<4xf32> to memref<*xf32>
  call @refbackend_consume_func_return_mrf32(%9) : (memref<*xf32>) -> ()
  return
}

// -----// IR Dump After ConvertMathToLLVM //----- //
func.func @forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>) attributes {llvm.emit_c_interface} {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.cast %arg0 : memref<*xf32> to memref<4xf32>
  %1 = memref.cast %arg1 : memref<*xf32> to memref<4xf32>
  %2 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  cf.br ^bb1(%c0 : index)
^bb1(%3: index):  // 2 preds: ^bb0, ^bb2
  %4 = arith.cmpi slt, %3, %c4 : index
  cf.cond_br %4, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %5 = memref.load %0[%3] : memref<4xf32>
  %6 = memref.load %1[%3] : memref<4xf32>
  %7 = arith.addf %5, %6 : f32
  memref.store %7, %2[%3] : memref<4xf32>
  %8 = arith.addi %3, %c1 : index
  cf.br ^bb1(%8 : index)
^bb3:  // pred: ^bb1
  %9 = memref.cast %2 : memref<4xf32> to memref<*xf32>
  call @refbackend_consume_func_return_mrf32(%9) : (memref<*xf32>) -> ()
  return
}

// -----// IR Dump After ConvertLinalgToLLVM //----- //
module attributes {torch.debug_module_name = "VAdd"} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.mlir.global private @global_seed(0 : i64) : i64
  func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func @forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>) attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<*xf32> to !llvm.struct<(i64, ptr<i8>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<*xf32> to !llvm.struct<(i64, ptr<i8>)>
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr<i8>)>
    %3 = llvm.bitcast %2 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %4 = llvm.load %3 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %5 = llvm.extractvalue %1[1] : !llvm.struct<(i64, ptr<i8>)>
    %6 = llvm.bitcast %5 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %7 = llvm.load %6 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %8 = llvm.mlir.constant(4 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.null : !llvm.ptr<f32>
    %11 = llvm.getelementptr %10[%8] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %12 = llvm.ptrtoint %11 : !llvm.ptr<f32> to i64
    %13 = llvm.mlir.constant(128 : index) : i64
    %14 = llvm.add %12, %13  : i64
    %15 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr<i8>
    %16 = llvm.bitcast %15 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %17 = llvm.ptrtoint %16 : !llvm.ptr<f32> to i64
    %18 = llvm.mlir.constant(1 : index) : i64
    %19 = llvm.sub %13, %18  : i64
    %20 = llvm.add %17, %19  : i64
    %21 = llvm.urem %20, %13  : i64
    %22 = llvm.sub %20, %21  : i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr<f32>
    %24 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.insertvalue %16, %24[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.insertvalue %23, %25[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.mlir.constant(0 : index) : i64
    %28 = llvm.insertvalue %27, %26[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.insertvalue %8, %28[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.insertvalue %9, %29[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    cf.br ^bb1(%c0 : index)
  ^bb1(%31: index):  // 2 preds: ^bb0, ^bb2
    %32 = builtin.unrealized_conversion_cast %31 : index to i64
    %33 = arith.cmpi slt, %31, %c4 : index
    cf.cond_br %33, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %34 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.getelementptr %34[%32] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %36 = llvm.load %35 : !llvm.ptr<f32>
    %37 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.getelementptr %37[%32] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %39 = llvm.load %38 : !llvm.ptr<f32>
    %40 = arith.addf %36, %39 : f32
    %41 = llvm.extractvalue %30[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.getelementptr %41[%32] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %40, %42 : !llvm.ptr<f32>
    %43 = arith.addi %31, %c1 : index
    cf.br ^bb1(%43 : index)
  ^bb3:  // pred: ^bb1
    %44 = llvm.mlir.constant(1 : index) : i64
    %45 = llvm.alloca %44 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %30, %45 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %46 = llvm.bitcast %45 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %47 = llvm.mlir.constant(1 : index) : i64
    %48 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %49 = llvm.insertvalue %47, %48[0] : !llvm.struct<(i64, ptr<i8>)>
    %50 = llvm.insertvalue %46, %49[1] : !llvm.struct<(i64, ptr<i8>)>
    %51 = builtin.unrealized_conversion_cast %50 : !llvm.struct<(i64, ptr<i8>)> to memref<*xf32>
    call @refbackend_consume_func_return_mrf32(%51) : (memref<*xf32>) -> ()
    return
  }
}


// -----// IR Dump After ConvertMemRefToLLVM //----- //
module attributes {torch.debug_module_name = "VAdd"} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.mlir.global private @global_seed(0 : i64) : i64
  func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func @forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>) attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<*xf32> to !llvm.struct<(i64, ptr<i8>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<*xf32> to !llvm.struct<(i64, ptr<i8>)>
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr<i8>)>
    %3 = llvm.bitcast %2 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %4 = llvm.load %3 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %5 = llvm.extractvalue %1[1] : !llvm.struct<(i64, ptr<i8>)>
    %6 = llvm.bitcast %5 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %7 = llvm.load %6 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %8 = llvm.mlir.constant(4 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.null : !llvm.ptr<f32>
    %11 = llvm.getelementptr %10[%8] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %12 = llvm.ptrtoint %11 : !llvm.ptr<f32> to i64
    %13 = llvm.mlir.constant(128 : index) : i64
    %14 = llvm.add %12, %13  : i64
    %15 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr<i8>
    %16 = llvm.bitcast %15 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %17 = llvm.ptrtoint %16 : !llvm.ptr<f32> to i64
    %18 = llvm.mlir.constant(1 : index) : i64
    %19 = llvm.sub %13, %18  : i64
    %20 = llvm.add %17, %19  : i64
    %21 = llvm.urem %20, %13  : i64
    %22 = llvm.sub %20, %21  : i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr<f32>
    %24 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.insertvalue %16, %24[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.insertvalue %23, %25[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.mlir.constant(0 : index) : i64
    %28 = llvm.insertvalue %27, %26[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.insertvalue %8, %28[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.insertvalue %9, %29[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    cf.br ^bb1(%c0 : index)
  ^bb1(%31: index):  // 2 preds: ^bb0, ^bb2
    %32 = builtin.unrealized_conversion_cast %31 : index to i64
    %33 = arith.cmpi slt, %31, %c4 : index
    cf.cond_br %33, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %34 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.getelementptr %34[%32] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %36 = llvm.load %35 : !llvm.ptr<f32>
    %37 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.getelementptr %37[%32] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %39 = llvm.load %38 : !llvm.ptr<f32>
    %40 = arith.addf %36, %39 : f32
    %41 = llvm.extractvalue %30[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.getelementptr %41[%32] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %40, %42 : !llvm.ptr<f32>
    %43 = arith.addi %31, %c1 : index
    cf.br ^bb1(%43 : index)
  ^bb3:  // pred: ^bb1
    %44 = llvm.mlir.constant(1 : index) : i64
    %45 = llvm.alloca %44 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %30, %45 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %46 = llvm.bitcast %45 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %47 = llvm.mlir.constant(1 : index) : i64
    %48 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %49 = llvm.insertvalue %47, %48[0] : !llvm.struct<(i64, ptr<i8>)>
    %50 = llvm.insertvalue %46, %49[1] : !llvm.struct<(i64, ptr<i8>)>
    %51 = builtin.unrealized_conversion_cast %50 : !llvm.struct<(i64, ptr<i8>)> to memref<*xf32>
    call @refbackend_consume_func_return_mrf32(%51) : (memref<*xf32>) -> ()
    return
  }
}


// -----// IR Dump After ConvertArithmeticToLLVM //----- //
func.func private @refbackend_consume_func_return_mrf32(memref<*xf32>) attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertArithmeticToLLVM //----- //
func.func @forward(%arg0: memref<*xf32>, %arg1: memref<*xf32>) attributes {llvm.emit_c_interface} {
  %0 = builtin.unrealized_conversion_cast %arg0 : memref<*xf32> to !llvm.struct<(i64, ptr<i8>)>
  %1 = builtin.unrealized_conversion_cast %arg1 : memref<*xf32> to !llvm.struct<(i64, ptr<i8>)>
  %2 = llvm.mlir.constant(4 : index) : i64
  %3 = llvm.mlir.constant(0 : index) : i64
  %4 = builtin.unrealized_conversion_cast %3 : i64 to index
  %5 = llvm.mlir.constant(1 : index) : i64
  %6 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr<i8>)>
  %7 = llvm.bitcast %6 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
  %8 = llvm.load %7 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
  %9 = llvm.extractvalue %1[1] : !llvm.struct<(i64, ptr<i8>)>
  %10 = llvm.bitcast %9 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
  %11 = llvm.load %10 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
  %12 = llvm.mlir.constant(4 : index) : i64
  %13 = llvm.mlir.constant(1 : index) : i64
  %14 = llvm.mlir.null : !llvm.ptr<f32>
  %15 = llvm.getelementptr %14[%12] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  %16 = llvm.ptrtoint %15 : !llvm.ptr<f32> to i64
  %17 = llvm.mlir.constant(128 : index) : i64
  %18 = llvm.add %16, %17  : i64
  %19 = llvm.call @malloc(%18) : (i64) -> !llvm.ptr<i8>
  %20 = llvm.bitcast %19 : !llvm.ptr<i8> to !llvm.ptr<f32>
  %21 = llvm.ptrtoint %20 : !llvm.ptr<f32> to i64
  %22 = llvm.mlir.constant(1 : index) : i64
  %23 = llvm.sub %17, %22  : i64
  %24 = llvm.add %21, %23  : i64
  %25 = llvm.urem %24, %17  : i64
  %26 = llvm.sub %24, %25  : i64
  %27 = llvm.inttoptr %26 : i64 to !llvm.ptr<f32>
  %28 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %29 = llvm.insertvalue %20, %28[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %30 = llvm.insertvalue %27, %29[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %31 = llvm.mlir.constant(0 : index) : i64
  %32 = llvm.insertvalue %31, %30[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %33 = llvm.insertvalue %12, %32[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %34 = llvm.insertvalue %13, %33[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  cf.br ^bb1(%4 : index)
^bb1(%35: index):  // 2 preds: ^bb0, ^bb2
  %36 = builtin.unrealized_conversion_cast %35 : index to i64
  %37 = builtin.unrealized_conversion_cast %35 : index to i64
  %38 = llvm.icmp "slt" %36, %2 : i64
  cf.cond_br %38, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %39 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %40 = llvm.getelementptr %39[%37] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  %41 = llvm.load %40 : !llvm.ptr<f32>
  %42 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %43 = llvm.getelementptr %42[%37] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  %44 = llvm.load %43 : !llvm.ptr<f32>
  %45 = llvm.fadd %41, %44  : f32
  %46 = llvm.extractvalue %34[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %47 = llvm.getelementptr %46[%37] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  llvm.store %45, %47 : !llvm.ptr<f32>
  %48 = llvm.add %36, %5  : i64
  %49 = builtin.unrealized_conversion_cast %48 : i64 to index
  cf.br ^bb1(%49 : index)
^bb3:  // pred: ^bb1
  %50 = llvm.mlir.constant(1 : index) : i64
  %51 = llvm.alloca %50 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
  llvm.store %34, %51 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
  %52 = llvm.bitcast %51 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
  %53 = llvm.mlir.constant(1 : index) : i64
  %54 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
  %55 = llvm.insertvalue %53, %54[0] : !llvm.struct<(i64, ptr<i8>)>
  %56 = llvm.insertvalue %52, %55[1] : !llvm.struct<(i64, ptr<i8>)>
  %57 = builtin.unrealized_conversion_cast %56 : !llvm.struct<(i64, ptr<i8>)> to memref<*xf32>
  call @refbackend_consume_func_return_mrf32(%57) : (memref<*xf32>) -> ()
  return
}

// -----// IR Dump After ConvertFuncToLLVM //----- //
module attributes {llvm.data_layout = "", torch.debug_module_name = "VAdd"} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.mlir.global private @global_seed(0 : i64) : i64
  llvm.func @refbackend_consume_func_return_mrf32(%arg0: i64, %arg1: !llvm.ptr<i8>) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr<i8>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr<i8>)>
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i64, ptr<i8>)> : (i64) -> !llvm.ptr<struct<(i64, ptr<i8>)>>
    llvm.store %2, %4 : !llvm.ptr<struct<(i64, ptr<i8>)>>
    llvm.call @_mlir_ciface_refbackend_consume_func_return_mrf32(%4) : (!llvm.ptr<struct<(i64, ptr<i8>)>>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_refbackend_consume_func_return_mrf32(!llvm.ptr<struct<(i64, ptr<i8>)>>) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @forward(%arg0: i64, %arg1: !llvm.ptr<i8>, %arg2: i64, %arg3: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr<i8>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr<i8>)>
    %3 = builtin.unrealized_conversion_cast %2 : !llvm.struct<(i64, ptr<i8>)> to memref<*xf32>
    %4 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %5 = llvm.insertvalue %arg2, %4[0] : !llvm.struct<(i64, ptr<i8>)>
    %6 = llvm.insertvalue %arg3, %5[1] : !llvm.struct<(i64, ptr<i8>)>
    %7 = builtin.unrealized_conversion_cast %6 : !llvm.struct<(i64, ptr<i8>)> to memref<*xf32>
    %8 = builtin.unrealized_conversion_cast %3 : memref<*xf32> to !llvm.struct<(i64, ptr<i8>)>
    %9 = builtin.unrealized_conversion_cast %7 : memref<*xf32> to !llvm.struct<(i64, ptr<i8>)>
    %10 = llvm.mlir.constant(4 : index) : i64
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = builtin.unrealized_conversion_cast %11 : i64 to index
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.extractvalue %8[1] : !llvm.struct<(i64, ptr<i8>)>
    %15 = llvm.bitcast %14 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %16 = llvm.load %15 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %17 = llvm.extractvalue %9[1] : !llvm.struct<(i64, ptr<i8>)>
    %18 = llvm.bitcast %17 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %19 = llvm.load %18 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %20 = llvm.mlir.constant(4 : index) : i64
    %21 = llvm.mlir.constant(1 : index) : i64
    %22 = llvm.mlir.null : !llvm.ptr<f32>
    %23 = llvm.getelementptr %22[%20] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %24 = llvm.ptrtoint %23 : !llvm.ptr<f32> to i64
    %25 = llvm.mlir.constant(128 : index) : i64
    %26 = llvm.add %24, %25  : i64
    %27 = llvm.call @malloc(%26) : (i64) -> !llvm.ptr<i8>
    %28 = llvm.bitcast %27 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %29 = llvm.ptrtoint %28 : !llvm.ptr<f32> to i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.sub %25, %30  : i64
    %32 = llvm.add %29, %31  : i64
    %33 = llvm.urem %32, %25  : i64
    %34 = llvm.sub %32, %33  : i64
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr<f32>
    %36 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %37 = llvm.insertvalue %28, %36[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %39 = llvm.mlir.constant(0 : index) : i64
    %40 = llvm.insertvalue %39, %38[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.insertvalue %20, %40[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.insertvalue %21, %41[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb1(%11 : i64)
  ^bb1(%43: i64):  // 2 preds: ^bb0, ^bb2
    %44 = builtin.unrealized_conversion_cast %43 : i64 to index
    %45 = builtin.unrealized_conversion_cast %44 : index to i64
    %46 = builtin.unrealized_conversion_cast %44 : index to i64
    %47 = llvm.icmp "slt" %45, %10 : i64
    llvm.cond_br %47, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %48 = llvm.extractvalue %16[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %49 = llvm.getelementptr %48[%46] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %50 = llvm.load %49 : !llvm.ptr<f32>
    %51 = llvm.extractvalue %19[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = llvm.getelementptr %51[%46] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %53 = llvm.load %52 : !llvm.ptr<f32>
    %54 = llvm.fadd %50, %53  : f32
    %55 = llvm.extractvalue %42[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %56 = llvm.getelementptr %55[%46] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %54, %56 : !llvm.ptr<f32>
    %57 = llvm.add %45, %13  : i64
    %58 = builtin.unrealized_conversion_cast %57 : i64 to index
    llvm.br ^bb1(%57 : i64)
  ^bb3:  // pred: ^bb1
    %59 = llvm.mlir.constant(1 : index) : i64
    %60 = llvm.alloca %59 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %42, %60 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %61 = llvm.bitcast %60 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %62 = llvm.mlir.constant(1 : index) : i64
    %63 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %64 = llvm.insertvalue %62, %63[0] : !llvm.struct<(i64, ptr<i8>)>
    %65 = llvm.insertvalue %61, %64[1] : !llvm.struct<(i64, ptr<i8>)>
    %66 = builtin.unrealized_conversion_cast %65 : !llvm.struct<(i64, ptr<i8>)> to memref<*xf32>
    %67 = llvm.extractvalue %65[0] : !llvm.struct<(i64, ptr<i8>)>
    %68 = llvm.extractvalue %65[1] : !llvm.struct<(i64, ptr<i8>)>
    llvm.call @refbackend_consume_func_return_mrf32(%67, %68) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_forward(%arg0: !llvm.ptr<struct<(i64, ptr<i8>)>>, %arg1: !llvm.ptr<struct<(i64, ptr<i8>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(i64, ptr<i8>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, ptr<i8>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr<i8>)>
    %3 = llvm.load %arg1 : !llvm.ptr<struct<(i64, ptr<i8>)>>
    %4 = llvm.extractvalue %3[0] : !llvm.struct<(i64, ptr<i8>)>
    %5 = llvm.extractvalue %3[1] : !llvm.struct<(i64, ptr<i8>)>
    llvm.call @forward(%1, %2, %4, %5) : (i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
}


// -----// IR Dump After ConvertControlFlowToLLVM //----- //
module attributes {llvm.data_layout = "", torch.debug_module_name = "VAdd"} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.mlir.global private @global_seed(0 : i64) : i64
  llvm.func @refbackend_consume_func_return_mrf32(%arg0: i64, %arg1: !llvm.ptr<i8>) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr<i8>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr<i8>)>
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i64, ptr<i8>)> : (i64) -> !llvm.ptr<struct<(i64, ptr<i8>)>>
    llvm.store %2, %4 : !llvm.ptr<struct<(i64, ptr<i8>)>>
    llvm.call @_mlir_ciface_refbackend_consume_func_return_mrf32(%4) : (!llvm.ptr<struct<(i64, ptr<i8>)>>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_refbackend_consume_func_return_mrf32(!llvm.ptr<struct<(i64, ptr<i8>)>>) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @forward(%arg0: i64, %arg1: !llvm.ptr<i8>, %arg2: i64, %arg3: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr<i8>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr<i8>)>
    %3 = builtin.unrealized_conversion_cast %2 : !llvm.struct<(i64, ptr<i8>)> to memref<*xf32>
    %4 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %5 = llvm.insertvalue %arg2, %4[0] : !llvm.struct<(i64, ptr<i8>)>
    %6 = llvm.insertvalue %arg3, %5[1] : !llvm.struct<(i64, ptr<i8>)>
    %7 = builtin.unrealized_conversion_cast %6 : !llvm.struct<(i64, ptr<i8>)> to memref<*xf32>
    %8 = builtin.unrealized_conversion_cast %3 : memref<*xf32> to !llvm.struct<(i64, ptr<i8>)>
    %9 = builtin.unrealized_conversion_cast %7 : memref<*xf32> to !llvm.struct<(i64, ptr<i8>)>
    %10 = llvm.mlir.constant(4 : index) : i64
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = builtin.unrealized_conversion_cast %11 : i64 to index
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.extractvalue %8[1] : !llvm.struct<(i64, ptr<i8>)>
    %15 = llvm.bitcast %14 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %16 = llvm.load %15 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %17 = llvm.extractvalue %9[1] : !llvm.struct<(i64, ptr<i8>)>
    %18 = llvm.bitcast %17 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %19 = llvm.load %18 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %20 = llvm.mlir.constant(4 : index) : i64
    %21 = llvm.mlir.constant(1 : index) : i64
    %22 = llvm.mlir.null : !llvm.ptr<f32>
    %23 = llvm.getelementptr %22[%20] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %24 = llvm.ptrtoint %23 : !llvm.ptr<f32> to i64
    %25 = llvm.mlir.constant(128 : index) : i64
    %26 = llvm.add %24, %25  : i64
    %27 = llvm.call @malloc(%26) : (i64) -> !llvm.ptr<i8>
    %28 = llvm.bitcast %27 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %29 = llvm.ptrtoint %28 : !llvm.ptr<f32> to i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.sub %25, %30  : i64
    %32 = llvm.add %29, %31  : i64
    %33 = llvm.urem %32, %25  : i64
    %34 = llvm.sub %32, %33  : i64
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr<f32>
    %36 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %37 = llvm.insertvalue %28, %36[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %39 = llvm.mlir.constant(0 : index) : i64
    %40 = llvm.insertvalue %39, %38[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.insertvalue %20, %40[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.insertvalue %21, %41[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb1(%11 : i64)
  ^bb1(%43: i64):  // 2 preds: ^bb0, ^bb2
    %44 = builtin.unrealized_conversion_cast %43 : i64 to index
    %45 = builtin.unrealized_conversion_cast %44 : index to i64
    %46 = builtin.unrealized_conversion_cast %44 : index to i64
    %47 = llvm.icmp "slt" %45, %10 : i64
    llvm.cond_br %47, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %48 = llvm.extractvalue %16[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %49 = llvm.getelementptr %48[%46] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %50 = llvm.load %49 : !llvm.ptr<f32>
    %51 = llvm.extractvalue %19[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = llvm.getelementptr %51[%46] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %53 = llvm.load %52 : !llvm.ptr<f32>
    %54 = llvm.fadd %50, %53  : f32
    %55 = llvm.extractvalue %42[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %56 = llvm.getelementptr %55[%46] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %54, %56 : !llvm.ptr<f32>
    %57 = llvm.add %45, %13  : i64
    %58 = builtin.unrealized_conversion_cast %57 : i64 to index
    llvm.br ^bb1(%57 : i64)
  ^bb3:  // pred: ^bb1
    %59 = llvm.mlir.constant(1 : index) : i64
    %60 = llvm.alloca %59 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %42, %60 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %61 = llvm.bitcast %60 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %62 = llvm.mlir.constant(1 : index) : i64
    %63 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %64 = llvm.insertvalue %62, %63[0] : !llvm.struct<(i64, ptr<i8>)>
    %65 = llvm.insertvalue %61, %64[1] : !llvm.struct<(i64, ptr<i8>)>
    %66 = builtin.unrealized_conversion_cast %65 : !llvm.struct<(i64, ptr<i8>)> to memref<*xf32>
    %67 = llvm.extractvalue %65[0] : !llvm.struct<(i64, ptr<i8>)>
    %68 = llvm.extractvalue %65[1] : !llvm.struct<(i64, ptr<i8>)>
    llvm.call @refbackend_consume_func_return_mrf32(%67, %68) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_forward(%arg0: !llvm.ptr<struct<(i64, ptr<i8>)>>, %arg1: !llvm.ptr<struct<(i64, ptr<i8>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(i64, ptr<i8>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, ptr<i8>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr<i8>)>
    %3 = llvm.load %arg1 : !llvm.ptr<struct<(i64, ptr<i8>)>>
    %4 = llvm.extractvalue %3[0] : !llvm.struct<(i64, ptr<i8>)>
    %5 = llvm.extractvalue %3[1] : !llvm.struct<(i64, ptr<i8>)>
    llvm.call @forward(%1, %2, %4, %5) : (i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
}


// -----// IR Dump After ReconcileUnrealizedCasts //----- //
module attributes {llvm.data_layout = "", torch.debug_module_name = "VAdd"} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.mlir.global private @global_seed(0 : i64) : i64
  llvm.func @refbackend_consume_func_return_mrf32(%arg0: i64, %arg1: !llvm.ptr<i8>) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr<i8>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr<i8>)>
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i64, ptr<i8>)> : (i64) -> !llvm.ptr<struct<(i64, ptr<i8>)>>
    llvm.store %2, %4 : !llvm.ptr<struct<(i64, ptr<i8>)>>
    llvm.call @_mlir_ciface_refbackend_consume_func_return_mrf32(%4) : (!llvm.ptr<struct<(i64, ptr<i8>)>>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_refbackend_consume_func_return_mrf32(!llvm.ptr<struct<(i64, ptr<i8>)>>) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @forward(%arg0: i64, %arg1: !llvm.ptr<i8>, %arg2: i64, %arg3: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr<i8>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr<i8>)>
    %3 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %4 = llvm.insertvalue %arg2, %3[0] : !llvm.struct<(i64, ptr<i8>)>
    %5 = llvm.insertvalue %arg3, %4[1] : !llvm.struct<(i64, ptr<i8>)>
    %6 = llvm.mlir.constant(4 : index) : i64
    %7 = llvm.mlir.constant(0 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.extractvalue %2[1] : !llvm.struct<(i64, ptr<i8>)>
    %10 = llvm.bitcast %9 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %11 = llvm.load %10 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %12 = llvm.extractvalue %5[1] : !llvm.struct<(i64, ptr<i8>)>
    %13 = llvm.bitcast %12 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %14 = llvm.load %13 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %15 = llvm.mlir.constant(4 : index) : i64
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.mlir.null : !llvm.ptr<f32>
    %18 = llvm.getelementptr %17[%15] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %19 = llvm.ptrtoint %18 : !llvm.ptr<f32> to i64
    %20 = llvm.mlir.constant(128 : index) : i64
    %21 = llvm.add %19, %20  : i64
    %22 = llvm.call @malloc(%21) : (i64) -> !llvm.ptr<i8>
    %23 = llvm.bitcast %22 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %24 = llvm.ptrtoint %23 : !llvm.ptr<f32> to i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.sub %20, %25  : i64
    %27 = llvm.add %24, %26  : i64
    %28 = llvm.urem %27, %20  : i64
    %29 = llvm.sub %27, %28  : i64
    %30 = llvm.inttoptr %29 : i64 to !llvm.ptr<f32>
    %31 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.insertvalue %23, %31[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %33 = llvm.insertvalue %30, %32[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = llvm.mlir.constant(0 : index) : i64
    %35 = llvm.insertvalue %34, %33[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = llvm.insertvalue %15, %35[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %37 = llvm.insertvalue %16, %36[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb1(%7 : i64)
  ^bb1(%38: i64):  // 2 preds: ^bb0, ^bb2
    %39 = llvm.icmp "slt" %38, %6 : i64
    llvm.cond_br %39, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %40 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.getelementptr %40[%38] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %42 = llvm.load %41 : !llvm.ptr<f32>
    %43 = llvm.extractvalue %14[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.getelementptr %43[%38] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %45 = llvm.load %44 : !llvm.ptr<f32>
    %46 = llvm.fadd %42, %45  : f32
    %47 = llvm.getelementptr %30[%38] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %46, %47 : !llvm.ptr<f32>
    %48 = llvm.add %38, %8  : i64
    llvm.br ^bb1(%48 : i64)
  ^bb3:  // pred: ^bb1
    %49 = llvm.mlir.constant(1 : index) : i64
    %50 = llvm.alloca %49 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %37, %50 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %51 = llvm.bitcast %50 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %52 = llvm.mlir.constant(1 : index) : i64
    %53 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %54 = llvm.insertvalue %52, %53[0] : !llvm.struct<(i64, ptr<i8>)>
    %55 = llvm.insertvalue %51, %54[1] : !llvm.struct<(i64, ptr<i8>)>
    llvm.call @refbackend_consume_func_return_mrf32(%52, %51) : (i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_forward(%arg0: !llvm.ptr<struct<(i64, ptr<i8>)>>, %arg1: !llvm.ptr<struct<(i64, ptr<i8>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(i64, ptr<i8>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, ptr<i8>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr<i8>)>
    %3 = llvm.load %arg1 : !llvm.ptr<struct<(i64, ptr<i8>)>>
    %4 = llvm.extractvalue %3[0] : !llvm.struct<(i64, ptr<i8>)>
    %5 = llvm.extractvalue %3[1] : !llvm.struct<(i64, ptr<i8>)>
    llvm.call @forward(%1, %2, %4, %5) : (i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
}


