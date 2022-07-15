#map0 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "Conv2d"} {
  func.func @forward(%arg0: tensor<3x3x32x32xf32>) -> tensor<3x3x30x30xf32> {
    %cst = arith.constant dense<[[[[0.10544271, -0.0632548705, 0.155782789], [0.149627611, -0.174780861, -0.0418872535], [-0.0906655266, 0.040773496, -0.11680039]], [[-0.170247212, 0.147964194, 0.0955948085], [0.16221562, 0.123264179, -0.171233669], [-0.149017662, 0.117653273, -0.0985492393]], [[0.0153566934, -0.178193122, 0.0848361403], [0.12641643, 0.148512617, 0.129268169], [-0.088901095, -0.17654486, -0.142492354]]], [[[-0.0374182314, 0.0986094102, -0.0577251315], [-0.0560243614, 0.131479472, 0.116924271], [-0.117099918, -0.0475517772, -0.00175973075]], [[0.17515704, -0.022550635, -0.0670964569], [-0.0284233782, -0.127548784, -0.0599880889], [0.166481301, -0.0813012421, -0.0646878853]], [[-0.157592237, 6.752100e-02, 0.0271486603], [-0.020048758, 0.0247509405, -0.0860386863], [0.152274042, 0.0267280918, -0.00991346687]]], [[[0.0988490208, -0.010171975, 0.0832737833], [-0.0220279731, -0.03541518, 0.0541252345], [-0.174909547, -0.113369323, -0.00568466634]], [[-0.0633686408, 0.165419623, 0.119426101], [0.16138503, 0.105015323, -0.0401478037], [0.0357512757, 0.0698492751, 0.00555766048]], [[0.181034967, -0.176028982, -0.0634875447], [0.140753478, 0.114295989, 0.18496187], [0.0485140048, 0.00907306094, 0.0741938576]]]]> : tensor<3x3x3x3xf32>
    %cst_0 = arith.constant dense<[0.189480066, -0.124956347, 0.0124727432]> : tensor<3xf32>
    %0 = linalg.init_tensor [3, 3, 30, 30] : tensor<3x3x30x30xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_0 : tensor<3xf32>) outs(%0 : tensor<3x3x30x30xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<3x3x30x30xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %cst : tensor<3x3x32x32xf32>, tensor<3x3x3x3xf32>) outs(%1 : tensor<3x3x30x30xf32>) -> tensor<3x3x30x30xf32>
    return %2 : tensor<3x3x30x30xf32>
  }
}
