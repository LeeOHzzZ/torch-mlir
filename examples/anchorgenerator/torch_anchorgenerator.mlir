module attributes {torch.debug_module_name = "DefaultAnchorGenerator"} {
  func.func @forward(%arg0: !torch.vtensor<[1,256,16,20],f32>) -> !torch.vtensor<[4800,4],f32> {
    %0 = torch.vtensor.literal(dense<[[-22.6274166, -11.3137083, 22.6274166, 11.3137083], [-1.600000e+01, -1.600000e+01, 1.600000e+01, 1.600000e+01], [-11.3137083, -22.6274166, 11.3137083, 22.6274166], [-45.2548332, -22.6274166, 45.2548332, 22.6274166], [-3.200000e+01, -3.200000e+01, 3.200000e+01, 3.200000e+01], [-22.6274166, -45.2548332, 22.6274166, 45.2548332], [-67.8822479, -33.941124, 67.8822479, 33.941124], [-4.800000e+01, -4.800000e+01, 4.800000e+01, 4.800000e+01], [-33.941124, -67.8822479, 33.941124, 67.8822479], [-90.5096664, -45.2548332, 90.5096664, 45.2548332], [-6.400000e+01, -6.400000e+01, 6.400000e+01, 6.400000e+01], [-45.2548332, -90.5096664, 45.2548332, 90.5096664], [-113.137085, -56.5685425, 113.137085, 56.5685425], [-8.000000e+01, -8.000000e+01, 8.000000e+01, 8.000000e+01], [-56.5685425, -113.137085, 56.5685425, 113.137085]]> : tensor<15x4xf32>) : !torch.vtensor<[15,4],f32>
    %1 = torch.vtensor.literal(dense<2.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<4.000000e+01> : tensor<f32>) : !torch.vtensor<[],f32>
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %int6 = torch.constant.int 6
    %float2.000000e00 = torch.constant.float 2.000000e+00
    %float6.400000e02 = torch.constant.float 6.400000e+02
    %int0 = torch.constant.int 0
    %cpu = torch.constant.device "cpu"
    %4 = torch.aten.arange.start_step %int0, %float6.400000e02, %float2.000000e00, %int6, %int0, %cpu, %false : !torch.int, !torch.float, !torch.float, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[320],f32>
    %5 = torch.aten.div.Tensor %4, %3 : !torch.vtensor<[320],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[320],f32>
    %6 = torch.aten.floor %5 : !torch.vtensor<[320],f32> -> !torch.vtensor<[320],f32>
    %7 = torch.aten.mul.Tensor %6, %3 : !torch.vtensor<[320],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[320],f32>
    %8 = torch.aten.sub.Tensor %4, %7, %int1 : !torch.vtensor<[320],f32>, !torch.vtensor<[320],f32>, !torch.int -> !torch.vtensor<[320],f32>
    %9 = torch.aten.add.Tensor %8, %2, %int1 : !torch.vtensor<[320],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[320],f32>
    %10 = torch.aten.arange.start_step %int0, %float6.400000e02, %float2.000000e00, %int6, %int0, %cpu, %false : !torch.int, !torch.float, !torch.float, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[320],f32>
    %11 = torch.aten.div.Tensor %10, %3 : !torch.vtensor<[320],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[320],f32>
    %12 = torch.aten.floor %11 : !torch.vtensor<[320],f32> -> !torch.vtensor<[320],f32>
    %13 = torch.aten.mul.Tensor %12, %1 : !torch.vtensor<[320],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[320],f32>
    %14 = torch.aten.add.Tensor %13, %2, %int1 : !torch.vtensor<[320],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[320],f32>
    %15 = torch.aten.unsqueeze %9, %int1 : !torch.vtensor<[320],f32>, !torch.int -> !torch.vtensor<[320,1],f32>
    %16 = torch.aten.unsqueeze %14, %int1 : !torch.vtensor<[320],f32>, !torch.int -> !torch.vtensor<[320,1],f32>
    %17 = torch.prim.ListConstruct %15, %16, %15, %16 : (!torch.vtensor<[320,1],f32>, !torch.vtensor<[320,1],f32>, !torch.vtensor<[320,1],f32>, !torch.vtensor<[320,1],f32>) -> !torch.list<vtensor>
    %18 = torch.aten.cat %17, %int1 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[320,4],f32>
    %19 = torch.aten.unsqueeze %18, %int1 : !torch.vtensor<[320,4],f32>, !torch.int -> !torch.vtensor<[320,1,4],f32>
    %20 = torch.aten.unsqueeze %0, %int0 : !torch.vtensor<[15,4],f32>, !torch.int -> !torch.vtensor<[1,15,4],f32>
    %21 = torch.aten.add.Tensor %19, %20, %int1 : !torch.vtensor<[320,1,4],f32>, !torch.vtensor<[1,15,4],f32>, !torch.int -> !torch.vtensor<[320,15,4],f32>
    %22 = torch.aten.flatten.using_ints %21, %int0, %int1 : !torch.vtensor<[320,15,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4800,4],f32>
    return %22 : !torch.vtensor<[4800,4],f32>
  }
}
