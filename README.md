# LaQ-AD: Language-to-Query Fusion for Sparse End-to-End Autonomous Driving

LaQ-AD injects the hidden token sequence of a frozen vision--language model
(Qwen2-VL-2B) into the shared query pool of a sparse single-decoder driving
model through a cross-attention layer with a **zero-initialized gate**
(decoder-layer × task × channel). The gate makes the fused model bit-identical
to its unfused counterpart at step zero, so the language pathway is credited
only with what it subsequently earns — and the same construction yields
retraining-free ablations from a single trained checkpoint.

Key components:
- `mmdet3d_plugin/models/qwen_token_attn.py` — language–query cross-attention
  with the decomposed zero-init gate and 2-layer K/V projection.
- `tools/qwen_cache.py` — offline VLM token cache (intermediate-layer hidden
  states; the VLM is never run during training).
- `configs/` — nuScenes training/evaluation configs (mmcv style).

Built on [HiP-AD](https://github.com/nullmax-vision/HiP-AD) (single-decoder
sparse architecture) with the nuScenes data path ported from
[SparseDrive](https://github.com/swc-17/SparseDrive). See `LICENSE`.

## Setup / Train / Eval
```bash
cd mmdet3d_plugin/ops && python setup.py develop && cd ../..   # CUDA op
python tools/qwen_cache.py --layers 18 --out-dir data/nuscenes/qwen \
    --ann-files data/infos/nuscenes_infos_train.pkl data/infos/nuscenes_infos_val.pkl
python tools/train.py configs/hipad_nusc_vla_s1hr.py --no-validate --work-dir results/laq_s1
python tools/test.py  configs/hipad_nusc_vla_s1hr.py results/laq_s1/latest.pth --eval bbox
```
