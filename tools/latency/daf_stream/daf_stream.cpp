// 수정(2026-11-04): CUDA Graph 캡처용 deformable_aggregation 순방향(현재 스트림). 인자 해석·출력 할당은 원본
//   mmdet3d_plugin/ops/src/deformable_aggregation.cpp 의 deformable_aggregation_forward 와 같다(추론 전용 — 역방향 없음).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void deformable_aggregation_stream(
    float* output, const float* mc_ms_feat, const int* spatial_shape, const int* scale_start_index,
    const float* sample_location, const float* weights,
    int batch_size, int num_cams, int num_feat, int num_embeds, int num_scale, int num_anchors, int num_pts, int num_groups);

at::Tensor forward(const at::Tensor &_mc_ms_feat, const at::Tensor &_spatial_shape, const at::Tensor &_scale_start_index,
                   const at::Tensor &_sampling_location, const at::Tensor &_weights) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  TORCH_CHECK(_mc_ms_feat.is_contiguous() && _sampling_location.is_contiguous() && _weights.is_contiguous()
              && _spatial_shape.is_contiguous() && _scale_start_index.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(_mc_ms_feat.scalar_type() == at::kFloat && _sampling_location.scalar_type() == at::kFloat
              && _weights.scalar_type() == at::kFloat, "float32 inputs expected");
  TORCH_CHECK(_spatial_shape.scalar_type() == at::kInt && _scale_start_index.scalar_type() == at::kInt, "int32 shapes expected");
  int batch_size = _mc_ms_feat.size(0);
  int num_feat = _mc_ms_feat.size(1);
  int num_embeds = _mc_ms_feat.size(2);
  int num_cams = _spatial_shape.size(0);
  int num_scale = _spatial_shape.size(1);
  int num_anchors = _sampling_location.size(1);
  int num_pts = _sampling_location.size(2);
  int num_groups = _weights.size(5);
  auto output = at::zeros({batch_size, num_anchors, num_embeds}, _mc_ms_feat.options());
  deformable_aggregation_stream(output.data_ptr<float>(), _mc_ms_feat.data_ptr<float>(), _spatial_shape.data_ptr<int>(),
      _scale_start_index.data_ptr<int>(), _sampling_location.data_ptr<float>(), _weights.data_ptr<float>(),
      batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward, "deformable_aggregation forward on current stream"); }
