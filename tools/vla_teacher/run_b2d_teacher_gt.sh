#!/usr/bin/env bash
# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   B2D 교사 정답(v2-sensor-b2d) 전체 생성 순서. CPU 전용(GPU 사용 안 함). 저장소 루트에서 실행.
#   출력: OUT_DIR(기본 NAS phd_thesis/bench2drive/teacher/gt_v2_sensor_b2d = data/vla_teacher/b2d/gt_v2_sensor_b2d),
#         중간 산출물 results/b2d_prep/(raw pkl·규약·TTC·코드북).
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT_DIR=${OUT_DIR:-data/vla_teacher/b2d/gt_v2_sensor_b2d}
W=${W:-10}
G=tools/vla_teacher/gen_teacher_gt_b2d.py
[ -f results/b2d_prep/teacher_gt_raw/train.pkl ] || python -u $G extract --split train --workers $W
[ -f results/b2d_prep/teacher_gt_raw/val.pkl ]   || python -u $G extract --split val --workers $W
python -u $G conventions --split train
python -u $G explore_ttc --workers $W
[ -f results/b2d_prep/b2d_action_codebook.npz ] || python -u tools/vla_teacher/build_b2d_action_codebook.py
python -u $G answers --out_dir "$OUT_DIR" --workers $W ${FORCE:+--force}
cp results/b2d_prep/b2d_action_codebook.npz results/b2d_prep/b2d_conventions_train.json results/b2d_prep/b2d_ttc_explore.json "$OUT_DIR"/
OMP_NUM_THREADS=4 python -u tools/vla_teacher/check_teacher_gt_b2d.py --data_dir "$OUT_DIR"
python -u tools/vla_teacher/make_fold_split_b2d_kf.py --gt_dir "$OUT_DIR" ${FORCE:+--force}
