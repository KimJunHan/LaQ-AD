// 수정(2026-11-04): CUDA Graph 캡처용 — 원본 커널(mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu)을 그대로 #include 하고
//   실행만 "현재 스트림"(at::cuda::getCurrentCUDAStream)에 올린다. 원본 실행 함수는 스트림 인자 없이 legacy 기본 스트림에
//   올려서 비기본 스트림 캡처(torch.cuda.graph)에 들어가지 않는다. 커널 코드·격자·블록 크기는 원본과 같다(결과 비트 동일).
#include "../../../mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu"
#include <ATen/cuda/CUDAContext.h>

void deformable_aggregation_stream(
    float* output, const float* mc_ms_feat, const int* spatial_shape, const int* scale_start_index,
    const float* sample_location, const float* weights,
    int batch_size, int num_cams, int num_feat, int num_embeds, int num_scale, int num_anchors, int num_pts, int num_groups
) {
    const long long num_kernels = (long long)batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    deformable_aggregation_kernel
        <<<(int)ceil(((double)num_kernels/128)), 128, 0, stream>>>(
        num_kernels, output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
