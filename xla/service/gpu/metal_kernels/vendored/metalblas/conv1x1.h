// conv1x1.h - 1x1 convolution as GEMM (very-thin-N path via convolution2d).
#ifdef MB_BUILD_CONV1X1
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T   __IN_T__
#define OUT_T  __OUT_T__
#define BMW    __BMW__
#define BNO    __BNO__
#define NSG    __NSG__
#define KCONST __KCONST__

kernel void conv1x1_gemm(
    device IN_T   *A   [[buffer(0)]],
    device IN_T   *B   [[buffer(1)]],
    device OUT_T  *C   [[buffer(2)]],
    constant int4& gP  [[buffer(3)]],   // packed (gM, gN, gK)
    uint3 tgid         [[threadgroup_position_in_grid]])
{
    int gM = gP.x, gN = gP.y, gK = gP.z;
    tensor<device IN_T, dextents<int32_t, 4>, tensor_inline> tA(A, dextents<int32_t, 4>(gK, gM, 1, 1));
    tensor<device IN_T, dextents<int32_t, 4>, tensor_inline> tW(B, dextents<int32_t, 4>(gN, gK, 1, 1));
    tensor<device OUT_T, dextents<int32_t, 4>, tensor_inline> tC(C, dextents<int32_t, 4>(gN, gM, 1, 1));

    constexpr auto desc = convolution2d_descriptor(
        int4(BNO, BMW, 1, 1), int4(KCONST, BMW, 1, 1), int2(1, 1),
        convolution2d_activation_layout::nhwc, convolution2d_weights_layout::hwio,
        int2(1, 1), int2(1, 1), 1, true, convolution2d_descriptor::mode::multiply);
    convolution2d<desc, execution_simdgroups<NSG>> op;

    int tiles_o = (gN + BNO - 1) / BNO;
    int tiles_w = (gM + BMW - 1) / BMW;
    if (int(tgid.x) >= tiles_o || int(tgid.y) >= tiles_w) return;
    int o_off = int(tgid.x) * BNO;
    int w_off = int(tgid.y) * BMW;

    auto mA = tA.slice(0, w_off, 0, 0);
    auto mW = tW.slice(o_off, 0, 0, 0);
    auto mC = tC.slice(o_off, w_off, 0, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mW), float>();
    op.run(mA, mW, cT);
    auto cO = op.get_destination_cooperative_tensor<decltype(mA), decltype(mW), OUT_T>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cO[i] = (OUT_T)cT[i];
    cO.store(mC);
}
#endif  // MB_BUILD_CONV1X1
