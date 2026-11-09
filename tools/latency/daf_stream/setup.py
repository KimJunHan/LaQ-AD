# 수정(2026-11-04): CUDA Graph 캡처용 deformable_aggregation(현재 스트림) 확장 빌드 — 원본 ops/setup.py 와 같은 nvcc 매크로, sm_90.
#   사용: cd tools/latency/daf_stream && python setup.py build_ext --inplace
import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_PFX = os.path.dirname(os.path.dirname(sys.executable))  # conda env (원본 빌드처럼 targets/x86_64-linux/include)
_INC = [os.path.join(_PFX, "targets", "x86_64-linux", "include")]
_LIB = [os.path.join(_PFX, "targets", "x86_64-linux", "lib")]
nvcc = ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
        "-gencode=arch=compute_90,code=sm_90"]
setup(
    name="laq_daf_stream_ext",
    ext_modules=[CUDAExtension("laq_daf_stream_ext", ["daf_stream.cpp", "daf_stream_cuda.cu"],
                               define_macros=[("WITH_CUDA", None)], include_dirs=_INC, library_dirs=_LIB,
                               extra_compile_args={"cxx": [], "nvcc": nvcc})],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
