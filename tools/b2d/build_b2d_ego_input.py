# 수정(2026-11-06): 사용자 결정 "B2D E2" — tools/navsim/build_navsim_ego_input.py 의 Bench2Drive 판.
#   교사 프롬프트의 자차 상태 줄과 같은 숫자(속도·종방향 가속도·yaw rate)를 학생 ego 쿼리에 직접 넣기 위한 표(작은 pkl).
#   출처 = B2D 교사 GT jsonl(data/vla_teacher/b2d/gt_v2_sensor_b2d/teacher_gt_b2d_<split>.jsonl = 교사 캐시 추출기
#   extract_teacher_cache_b2d.py 가 읽는 파일; 캐시 프롬프트 = build_user_text(r) = r["prompt"]["user"] 그대로, dropout 없음).
#   값은 프롬프트 user 텍스트 안의 자차 상태 줄(prompt_parts["ego"], 정확히 한 줄)을 그대로 읽는다(표기 정밀도 speed vv.v,
#   accel +aa.aa, yaw +w.www) → 교사가 받은 숫자와 비트 단위로 같은 10 진 값. 결측("Ego state: unavailable.")은 [0,0,0,0]
#   (B2D 실제 결측 없음 — common_b2d 주석; 나오면 그대로 처리).
#   미래 정보 없음 검사: 각 키프레임의 ego_status.raw(ego_vel·ego_accel·ego_rot)가 infos 의 같은 프레임(현재 프레임) 값과 같고,
#     speed = max(ego_vel[0], 0), accel = ego_accel[0], yaw = sign × ego_rotation_rate[2] 를 다시 계산해 반올림 범위 안인지 본다
#     (gen_teacher_gt_b2d.ego_inputs_b2d 와 같은 식; sign 은 b2d_conventions_train.json 의 데이터셋 전체 상수 = 좌표 규약).
#   10 Hz 학생 프레임 → 2 Hz 키프레임 사상 = LoadTeacherCacheB2D 와 같은 규칙: 같은 클립의 키프레임(= GT 레코드 = 캐시 표본) 중
#     kf ≤ f 인 가장 큰 것, f − kf ≤ 4(없으면 실패 — 대체값 없음). 즉 학생은 그 프레임에 쓰이는 교사 출력의 프롬프트 숫자를 받는다
#     (교사 나이 0~0.4 s 와 같은 나이; 폐쇄루프에서도 교사 갱신 시점 값을 재사용해야 학습과 같다).
#   출력 키 = "<clip>/<frame_idx:05d>"(학생 프레임; clip = infos folder 의 마지막 이름). 값 = float32 [speed, accel_long, yaw_rate, available].
#   사용:
#     python -I tools/b2d/build_b2d_ego_input.py build <teacher_gt_b2d_<split>.jsonl> <b2d_infos_<split>.pkl> <conventions.json> <out pkl>
#     python -I tools/b2d/build_b2d_ego_input.py check_index <out pkl> <cache_root>   # 캐시 색인 키프레임으로 같은 사상인지 확인
# pickle 은 저장소 안 자체 산출물(infos·이 도구 출력)만 읽는다(외부 입력 아님).
import bisect
import glob
import hashlib
import json
import os
import pickle
import re
import sys

import numpy as np

STRIDE = 5
FIELDS = ["speed_mps", "accel_long_mps2", "yaw_rate_rps", "available"]
# common_b2d.ego_line 의 형식: speed {:04.1f}, accel {:+06.2f}, yaw {:+.3f}
PAT = re.compile(r"Ego state: speed ([0-9]{2}\.[0-9]) m/s, longitudinal acceleration ([+-][0-9]{2}\.[0-9]{2}) m/s\^2, "
                 r"yaw rate ([+-][0-9]\.[0-9]{3}) rad/s \(positive = turning left\)\.")
UNAVAIL = "Ego state: unavailable."


def clip_name(folder):
    c = os.path.basename(str(folder).rstrip("/"))
    if not c:
        raise SystemExit(f"빈 folder 이름 {folder!r}")
    return c


def keyframe_for(kfs, f):   # models/encoder/vl_tokens_cache/loading_b2d.b2d_keyframe_for 와 같은 식
    i = bisect.bisect_right(kfs, f) - 1
    if i < 0 or f - kfs[i] >= STRIDE:
        return None
    return kfs[i]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(gt, info, conv, out):
    sgn = 1.0 if json.load(open(conv))["yaw_rate_regress_slope"] > 0 else -1.0
    infos = pickle.load(open(info, "rb"))
    infos = infos["infos"] if isinstance(infos, dict) else infos
    by_frame = {}
    for i in infos:
        k = (clip_name(i["folder"]), int(i["frame_idx"]))
        if k in by_frame:
            raise SystemExit(f"infos 프레임 중복 {k}")
        by_frame[k] = i
    tab, kfs = {}, {}
    with open(gt) as fh:
        for ln in fh:
            if not ln.strip():
                continue
            r = json.loads(ln)
            tok = r["token"]
            clip, _, kf = tok.rpartition("/")
            if not clip or len(kf) != 5 or not kf.isdigit():
                raise SystemExit(f"token 형식 {tok!r}")
            kf = int(kf)
            if tok in tab:
                raise SystemExit(f"token 중복 {tok}")
            if int(r["frame_idx"]) != kf or clip_name(r["scene_token"]) != clip:
                raise SystemExit(f"{tok}: frame_idx/scene_token 불일치")
            e = r["ego_status"]
            ego_txt = (r.get("prompt_parts") or {}).get("ego")
            lines = r["prompt"]["user"].split("\n")
            ego_lines = [l for l in lines if l.startswith("Ego state:")]
            if len(ego_lines) != 1 or ego_txt != ego_lines[0]:
                raise SystemExit(f"{tok}: 프롬프트 자차 상태 줄이 정확히 하나가 아니거나 prompt_parts.ego 와 다르다: {ego_lines}")
            line = ego_lines[0]
            m = PAT.fullmatch(line)
            if e["available"]:
                if m is None:
                    raise SystemExit(f"{tok}: 자차 상태 줄 형식 불일치: {line}")
                v = [float(m.group(1)), float(m.group(2)), float(m.group(3)), 1.0]
                if abs(v[0] - e["speed"]) > 0.0501 or abs(v[1] - e["accel_long"]) > 0.00501 or abs(v[2] - e["yaw_rate"]) > 0.000501:
                    raise SystemExit(f"{tok}: 프롬프트 값 {v[:3]} 과 ego_status {e} 가 반올림 범위 밖")
                # 미래 정보 없음: 현재 프레임 infos 값으로 다시 계산
                inf = by_frame.get((clip, kf))
                if inf is None:
                    raise SystemExit(f"{tok}: infos 에 이 키프레임이 없다")
                raw = e["raw"]
                for a, b in (("ego_vel", "ego_vel"), ("ego_accel", "ego_accel"), ("ego_rot", "ego_rotation_rate")):
                    if not np.array_equal(np.asarray(raw[a], np.float64), np.asarray(inf[b], np.float64)):
                        raise SystemExit(f"{tok}: ego_status.raw.{a} != infos 현재 프레임 {b}")
                re_ = (max(float(inf["ego_vel"][0]), 0.0), float(inf["ego_accel"][0]), sgn * float(inf["ego_rotation_rate"][2]))
                if abs(re_[0] - v[0]) > 0.0501 or abs(re_[1] - v[1]) > 0.00501 or abs(re_[2] - v[2]) > 0.000501:
                    raise SystemExit(f"{tok}: 현재 프레임 재계산 {re_} 과 프롬프트 {v[:3]} 불일치")
            else:
                if not line.startswith(UNAVAIL):
                    raise SystemExit(f"{tok}: available=False 인데 프롬프트 줄이 {line!r}")
                v = [0.0, 0.0, 0.0, 0.0]
            tab[tok] = np.asarray(v, np.float32)
            kfs.setdefault(clip, []).append(kf)
    kfs = {c: sorted(v) for c, v in kfs.items()}
    sub, src, ages = {}, {}, []
    for (clip, f) in by_frame:
        ks = kfs.get(clip)
        if ks is None:
            raise SystemExit(f"{info}: 클립 {clip} 의 교사 GT 키프레임이 없다")
        kf = keyframe_for(ks, f)
        if kf is None:
            raise SystemExit(f"{info}: {clip} frame {f} 에 쓸 키프레임(kf ≤ f, f−kf ≤ {STRIDE - 1})이 없다")
        key = f"{clip}/{f:05d}"
        sub[key] = tab[f"{clip}/{kf:05d}"]
        src[key] = f"{clip}/{kf:05d}"
        ages.append(f - kf)
    unused = set(tab) - set(src.values())
    if unused:
        raise SystemExit(f"쓰이지 않는 GT 키프레임 {len(unused)} (예 {sorted(unused)[:3]}) — infos 분할이 GT 와 다르다")
    a = np.stack(list(sub.values()))
    av = a[:, 3] == 1
    meta = dict(source=os.path.abspath(gt), source_sha256=sha256(gt), info=info, conventions=conv, fields=FIELDS,
                rounding="parsed from teacher prompt ego line (speed 1, accel 2, yaw 3 decimals)",
                keyframe_rule="LoadTeacherCacheB2D: same clip, largest teacher keyframe kf <= frame_idx, frame_idx - kf <= 4",
                n=len(sub), n_keyframes=len(tab), n_unavailable=int((~av).sum()),
                age_hist={int(k): int(c) for k, c in zip(*np.unique(ages, return_counts=True))},
                mean=a[av, :3].mean(0).round(4).tolist(), std=a[av, :3].std(0).round(4).tolist(),
                abs_p99=np.percentile(np.abs(a[av, :3]), 99, axis=0).round(4).tolist(), abs_max=np.abs(a[av, :3]).max(0).round(4).tolist())
    tmp = out + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(tmp, "wb") as fh:
        pickle.dump(dict(metadata=meta, ego_input=sub, source_keyframe=src), fh)
    os.replace(tmp, out)
    print(json.dumps(meta, ensure_ascii=False))


def check_index(pkl, cache_root):
    d = pickle.load(open(pkl, "rb"))
    files = sorted(glob.glob(os.path.join(cache_root, "_index", "teacher_*.jsonl")))
    if not files:
        raise SystemExit(f"{cache_root}/_index/teacher_*.jsonl 없음")
    kfs = {}
    for p in files:
        for ln in open(p):
            if ln.strip():
                clip, _, kf = json.loads(ln)["token"].rpartition("/")
                kfs.setdefault(clip, set()).add(int(kf))
    kfs = {c: sorted(v) for c, v in kfs.items()}
    bad = []
    for key, s in d["source_keyframe"].items():
        clip, _, f = key.rpartition("/")
        kf = keyframe_for(kfs.get(clip, []), int(f))
        if kf is None or f"{clip}/{kf:05d}" != s:
            bad.append((key, s, kf))
    if bad:
        raise SystemExit(f"[check_index] FAIL {len(bad)} 프레임이 캐시 색인 키프레임과 다르게 사상됨 (예 {bad[:3]})")
    print(f"[check_index] OK {pkl}: {len(d['source_keyframe'])} 프레임 모두 캐시 색인({len(files)} 파일) 키프레임과 같은 사상")


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "build":
        build(*sys.argv[2:6])
    elif len(sys.argv) == 4 and sys.argv[1] == "check_index":
        check_index(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("사용법: 파일 머리 주석(build / check_index)")
