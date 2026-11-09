# 수정(2026-11-05): navhard_two_stage 평가 준비 — navtest test config 에서 navhard 추론 config 를 만든다(새 파일; configs/ 무수정).
"""base = configs/navsim/v2_teacher/stage{1,2}/test.py(navtest). 바꾸는 것만:
  data.test.ann_file        → navhard info pkl (build_navhard_infos.py)
  data.test.teacher_map_files → [navhard 지도 pkl] (build_navhard_teacher_map.py)
  test pipeline LoadTeacherCacheNusc.cache_root → navhard 교사 캐시 루트(extract_teacher_cache_navhard.py, 기본 data/navsim/laq_cache_navhard)
  evaluation.jsonfile_prefix → 출력 폴더
  data.test.ego_input_file  → navhard ego_input pkl(build_navsim_ego_input.py, base 에 ego_input_file 이 있을 때만; E2)  # 수정(2026-11-06): navhard E2 ego_input
나머지(모델·가중치 규칙·카메라 3 대·batch 1·시간 뱅크 장면 경계 초기화·plan_key)는 navtest 와 같다. 덮어쓴 값을 다시 읽어 확인한다.
실행(hipad_bw): python tools/navsim/make_navhard_test_cfg.py --base configs/navsim/v2_teacher/stage2/test.py --out <dir>/navhard_test.py
"""
import argparse
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
D = os.path.join(REPO, "data/infos/navsim_navhard_20261105")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--infos", default=os.path.join(D, "navsim_navhard_two_stage.pkl"))
    ap.add_argument("--map", default=os.path.join(D, "teacher_map_navhard.pkl"))
    ap.add_argument("--cache_root", default="data/navsim/laq_cache_navhard")
    ap.add_argument("--jsonfile_prefix", required=False)
    ap.add_argument("--ego_input", default=os.path.join(D, "navsim_ego_input_navhard.pkl"))  # 수정(2026-11-06): navhard E2 ego_input
    a = ap.parse_args()
    from mmcv import Config
    c = Config.fromfile(a.base, import_custom_modules=False)
    t = c.data.test
    if t.type != "NavsimTeacherDataset":
        raise SystemExit(f"data.test.type {t.type} — NavsimTeacherDataset config 만")
    for p in (a.infos, a.map):
        if not os.path.exists(p):
            raise SystemExit(f"없음: {p}")
    t.ann_file = os.path.abspath(a.infos)
    t.teacher_map_files = [os.path.abspath(a.map)]
    n = 0
    for step in t.pipeline:
        if step["type"] == "LoadTeacherCacheNusc":
            step["cache_root"] = a.cache_root
            n += 1
    if n != 1:
        raise SystemExit(f"test pipeline 의 LoadTeacherCacheNusc {n} 개(1 이어야 함)")
    if c.data.get("samples_per_gpu") != 1 or t.get("samples_per_gpu", 1) != 1:
        raise SystemExit("batch 1 이 아니다 — 장면 경계 시간 뱅크 초기화가 깨진다")
    # 수정(2026-11-06): navhard E2 ego_input — base(E2)의 ego_input_file 은 navtest 표라 navhard 프레임이 없다 → navhard 표로 바꾼다.
    #   E2 판정 = data.test.ego_input_file 이 있거나 test pipeline Collect keys 에 "ego_input". 둘 다 없으면(비 E2) 아무것도 바꾸지 않는다.
    collect_ego = any(s["type"] == "Collect" and "ego_input" in s.get("keys", ()) for s in t.pipeline)
    if t.get("ego_input_file") is not None or collect_ego:
        if t.get("ego_input_file") is None:
            raise SystemExit("test pipeline Collect 에 ego_input 이 있는데 data.test.ego_input_file 이 없다(E2 config 불일치)")
        if not os.path.exists(a.ego_input):
            raise SystemExit(f"E2 config 인데 navhard ego_input pkl 없음: {a.ego_input} (tools/navsim/build_navsim_ego_input.py 로 생성)")
        import pickle
        tab = pickle.load(open(a.ego_input, "rb"))["ego_input"]
        miss = {i["token"] for i in pickle.load(open(t.ann_file, "rb"))["infos"]} - set(tab)
        if miss:
            raise SystemExit(f"{a.ego_input}: navhard info 프레임 {len(miss)} 개가 표에 없음(예 {sorted(miss)[:3]})")
        t.ego_input_file = os.path.abspath(a.ego_input)
    if a.jsonfile_prefix:
        c.evaluation.jsonfile_prefix = a.jsonfile_prefix
    c.navhard_eval_only = True
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    c.dump(a.out)
    r = Config.fromfile(a.out, import_custom_modules=False)
    got = [s["cache_root"] for s in r.data.test.pipeline if s["type"] == "LoadTeacherCacheNusc"]
    assert r.data.test.get("ego_input_file") == t.get("ego_input_file")  # 수정(2026-11-06): navhard E2 ego_input
    assert r.data.test.ann_file == t.ann_file and list(r.data.test.teacher_map_files) == t.teacher_map_files and got == [a.cache_root]
    print(f"OK {a.out}: ann_file={t.ann_file} map={t.teacher_map_files[0]} cache_root={a.cache_root} ego_input_file={t.get('ego_input_file')}")  # 수정(2026-11-06): navhard E2 ego_input


if __name__ == "__main__":
    main()
