# 수정(2026-11-05): navhard_two_stage 평가 준비 — 학생 지도 pkl(NavsimTeacherDataset teacher_map_files). 새 파일, CPU, env navsim.
"""tools/vla_teacher/navsim_teacher_map.py(무수정)를 navhard info 에 그대로 실행한다. 바꾸는 것은 지도 경로뿐:
그 파일의 MAPS_ROOT(…/nas_mirror/…/navsim/maps)는 이 서버에 없어서 모듈 변수를 이 서버의 nuPlan 지도(navsim_env.sh 의
NUPLAN_MAPS_ROOT = datasets/navsim/maps, nuplan-maps-v1.0)로 덮어쓴 뒤 main() 을 부른다(Pool 은 fork 라 작업자도 같은 값).

NavsimTeacherDataset 은 test 에서도 모든 프레임 token 의 지도 pkl 항목을 요구한다(학습 손실·평가 지표에는 안 쓰임; 계획 EPDMS 와 무관).
합성 프레임: ego2global = SE(2)(x, y, heading) — frame_map 은 2D(x, y, yaw)만 쓰므로 원본과 같은 규칙. lane_change 는 미래 GT 가
없어 'future_incomplete'(라벨용 신호, 입력 아님).
실행: source scripts/navsim/navsim_env.sh && python -I tools/navsim/build_navhard_teacher_map.py \
        --infos data/infos/navsim_navhard_20261105/navsim_navhard_two_stage.pkl \
        --out data/infos/navsim_navhard_20261105/teacher_map_navhard.pkl --workers 8
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vla_teacher"))
import navsim_teacher_map as M  # noqa: E402

if __name__ == "__main__":
    root = os.environ.get("NUPLAN_MAPS_ROOT", "/home1/irteam/data-vol1/datasets/navsim/maps")
    if not os.path.isdir(os.path.join(root, "us-nv-las-vegas-strip")):
        raise SystemExit(f"nuPlan 지도 없음: {root}")
    if "--workers" in sys.argv and int(sys.argv[sys.argv.index("--workers") + 1]) > 8:
        raise SystemExit("--workers ≤ 8 (공용 서버)")
    M.MAPS_ROOT = root
    M.main()
