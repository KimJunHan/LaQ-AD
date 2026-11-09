# 수정(2026-11-05): 사용자 결정 "E2 자차 상태 직접 입력" — 교사 프롬프트의 자차 상태 줄과 같은 값(속도·종방향 가속도·yaw rate)을
#   학생 ego 쿼리에 직접 넣기 위한 token → 값 표(작은 pkl). 출처 = 교사 GT jsonl(teacher_gt_<split>.jsonl)의 ego_status
#   (gen_teacher_gt_navsim.py ego_inputs: speed = |(vx,vy)|, accel_long = ax, yaw rate = 직전 로그 프레임 포즈와의 yaw 차 / Δt).
#   값은 교사 프롬프트의 자차 상태 줄에서 직접 읽는다(표기 정밀도 speed vv.v, accel +a.aa, yaw +w.www) → 교사가 받은 숫자와 같다. 결측(교사 "Ego state: unavailable")은 avail=0.
#   info 의 모든 프레임이 표에 있어야 한다(없으면 실패 — 대체값 없음).
#   사용: python -I tools/navsim/build_navsim_ego_input.py <teacher_gt_<split>.jsonl> <info pkl> <out pkl>
import json, pickle, re, sys
import numpy as np

gt, info, out = sys.argv[1:4]
# 교사 프롬프트 마지막 줄을 그대로 읽는다(교사가 받은 숫자 그 자체 — jsonl 4 자리 값을 다시 반올림하면 이중 반올림으로 어긋난다)
PAT = re.compile(r"Ego state: speed ([0-9.]+) m/s, longitudinal acceleration ([+-][0-9.]+) m/s\^2, yaw rate ([+-][0-9.]+) rad/s \(positive = turning left\)\.")
tab = {}
with open(gt) as fh:
    for ln in fh:
        if not ln.strip():
            continue
        r = json.loads(ln)
        e = r["ego_status"]
        if r["token"] in tab:
            raise SystemExit(f"token 중복 {r['token']}")
        line = r["prompt"]["user"].rstrip("\n").split("\n")[-1]
        m = PAT.fullmatch(line)
        if e["available"]:
            if m is None:
                raise SystemExit(f"{r['token']}: 자차 상태 줄 형식 불일치: {line}")
            v = [float(m.group(1)), float(m.group(2)), float(m.group(3)), 1.0]
            if abs(v[0] - e["speed"]) > 0.051 or abs(v[1] - e["accel_long"]) > 0.0051 or abs(v[2] - e["yaw_rate"]) > 0.00051:
                raise SystemExit(f"{r['token']}: 프롬프트 값 {v[:3]} 과 ego_status {e} 가 반올림 범위 밖")
        else:
            if not line.startswith("Ego state: unavailable."):
                raise SystemExit(f"{r['token']}: available=False 인데 프롬프트 줄이 {line!r}")
            v = [0.0, 0.0, 0.0, 0.0]
        tab[r["token"]] = np.asarray(v, np.float32)
infos = pickle.load(open(info, "rb"))["infos"]
miss = [i["token"] for i in infos if i["token"] not in tab]
if miss:
    raise SystemExit(f"{info}: 표에 없는 프레임 {len(miss)} (예 {miss[:3]})")
sub = {i["token"]: tab[i["token"]] for i in infos}
a = np.stack(list(sub.values()))
meta = dict(source=gt, info=info, fields=["speed_mps", "accel_long_mps2", "yaw_rate_rps", "available"],
            rounding="parsed from teacher prompt line (1, 2, 3 decimals)", n=len(sub), n_unavailable=int((a[:, 3] == 0).sum()))
pickle.dump(dict(metadata=meta, ego_input=sub), open(out, "wb"))
print(meta, "mean", a[a[:, 3] == 1, :3].mean(0).round(3), "std", a[a[:, 3] == 1, :3].std(0).round(3))
