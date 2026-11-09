# 수정(2026-11-09): C3 GRPO 검토 B (1)(2) — 체인 보조(관문 강제·RUNS 분배 검사). 신규 파일(기존 파일 무수정).
"""run_grpo_navsim_chain.sh 가 부르는 판정 도구. numpy·torch 불필요(어느 env 든).

gates  --gates-dir D [--approved F]
  MODE=gates 산출을 읽어 관문 통과 여부를 판정한다. 하나라도 실패면 종료 코드 1(체인은 die). 결과 D/gates_check.json.
  기준(엄격 = 설계 v4.1 원문):
    G1      g1.json decision == go
    G2-1    g2_1.pass_ (≤ 1e-4 m + 인덱스 일치), deterministic_daf == true
    G2-2    g2_2.pass_ (배치 8 vs 1 ≤ 1e-4 m + 인덱스 일치)
    G2-3a/b/d  g2_3a_pass, g2_3b.pass_, g2_3d_bank_tensors_with_grad == 0
    G2-3c   g2_3c_det: 같은 모드 반복 기울기 상대 차 최대 ≤ 1e-5(켬·끔 둘 다), 켬/끔 μ 차 ≤ 1e-4 m, 기울기 상대 차 최대 ≤ 1e-3
    G2-7    g2_7_on/mem_summary.json frac ≤ 0.8 (micro-batch 최대 메모리 / 장치 또는 상한), 격리 끔은 기록만
    G2-5    sigma_ref.json 의 sigma_ref 유한·양수
    G2-6    기록만(판정 없음)
  --approved F(사용자 결정 기록 파일, 예: results/.grpo_gate_criteria_approved)가 **유효할 때만** 바뀐 기준(impl_log §8·§8.1 제안).
    유효 = 비어 있지 않고, 결정 일자(YYYY-MM-DD)와 대상 시작 체크포인트(--init 의 경로 그대로 또는 절대 경로)가 적혀 있음.
    그 밖(없음·빈 파일·일자 또는 INIT 누락)은 불승인 → 엄격 기준(사유는 gates_check.json approval_invalid).
    G2-2    g2_1 통과 ∧ g2_2_partner_independence.max_dmu_m == 0 ∧ max_draw == 0 (배치 짝 누설 없음). 배치 8 vs 1 차는 기록
    G2-3c   반복 ≤ 1e-5(켬·끔), 켬/끔 μ 차 ≤ 1e-3 m, 켬/끔 기울기 상대 차 **중앙** ≤ 1e-3 (큰 차 매개변수 목록은 기록)
claim  --dir D --mode runs|eval --gpu N --runs "r1 r2 .." --pid P
  D/.claims/<mode>_gpu<N>.json 에 (pid, 프로세스 시작 시각, runs) 를 기록한다. 살아 있는 다른 체인의 claim(runs·eval 모두)과 런이
  겹치면 종료 코드 1. 검사와 기록은 D/.claims.lock flock 아래에서 한다. 죽은 pid(또는 시작 시각이 다른 pid)의 claim 은 무시·덮어씀.
"""
import argparse
import fcntl
import json
import math
import os
import re
import sys
from pathlib import Path

ALL_RUNS = ["full_s0", "b1_s0", "c1_s0", "full_s1", "b1_s1", "aa_s0", "ad_s0", "b2_s0", "ab_s0", "r_s0", "p_s0", "c2_s0"]


def _load(p):
    p = Path(p)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def judge_gates(gd, approved):
    gd = Path(gd)
    res, fails = {}, []

    def put(name, ok, detail):
        res[name] = dict(ok=bool(ok), detail=detail)
        if not ok:
            fails.append(name)

    g1 = _load(gd / "g1.json")
    put("G1", g1 is not None and g1.get("decision") == "go", None if g1 is None else dict(decision=g1.get("decision"), kappa=g1.get("kappa")))
    g2 = _load(gd / "g2_123.json")
    if g2 is None:
        put("G2-1·2·3", False, "g2_123.json 없음")
    else:
        put("G2-DAF", g2.get("deterministic_daf") is True, dict(deterministic_daf=g2.get("deterministic_daf")))
        g21 = g2.get("g2_1", {})
        put("G2-1", g21.get("pass_") is True, g21)
        g22 = g2.get("g2_2", {})
        pi = g2.get("g2_2_partner_independence")
        if approved:
            ok = g21.get("pass_") is True and pi is not None and pi.get("max_dmu_m") == 0.0 and pi.get("max_draw") == 0.0
            put("G2-2", ok, dict(criterion="승인 기준: G2-1 통과 ∧ 짝 독립 = 0", partner_independence=pi,
                                 recorded_batch8_vs_1=dict(max_dmu_m=g22.get("max_dmu_m"), argmax_match_frac=g22.get("argmax_match_frac"))))
        else:
            put("G2-2", g22.get("pass_") is True, dict(criterion="엄격: ≤ 1e-4 m + 인덱스 일치", **g22))
        put("G2-3a", g2.get("g2_3a_pass") is True, g2.get("g2_3a_requires_grad"))
        put("G2-3b", g2.get("g2_3b", {}).get("pass_") is True, g2.get("g2_3b"))
        put("G2-3d", g2.get("g2_3d_bank_tensors_with_grad") == 0, g2.get("g2_3d_bank_tensors_with_grad"))
        c = g2.get("g2_3c_det")
        if c is None:
            put("G2-3c", False, "g2_3c_det 없음(--deterministic-daf 로 다시 돌릴 것)")
        else:
            rep_ok = c["rep_on"]["grad_rel_max"] <= 1e-5 and c["rep_off"]["grad_rel_max"] <= 1e-5 \
                and c["rep_on"]["max_dmu_m"] == 0.0 and c["rep_off"]["max_dmu_m"] == 0.0
            o = c["on_vs_off"]
            if approved:
                ok = rep_ok and o["max_dmu_m"] <= 1e-3 and o["grad_rel_median"] <= 1e-3
                crit = "승인 기준: 반복 ≤ 1e-5, 켬/끔 μ ≤ 1e-3 m, 기울기 상대 차 중앙 ≤ 1e-3"
            else:
                ok = rep_ok and o["max_dmu_m"] <= 1e-4 and o["grad_rel_max"] <= 1e-3
                crit = "엄격: 반복 ≤ 1e-5, 켬/끔 μ ≤ 1e-4 m, 기울기 상대 차 최대 ≤ 1e-3"
            put("G2-3c", ok, dict(criterion=crit, rep_on_max=c["rep_on"]["grad_rel_max"], rep_off_max=c["rep_off"]["grad_rel_max"],
                                  on_vs_off={k: o[k] for k in ("max_dmu_m", "grad_rel_median", "grad_rel_max")},
                                  params_rel_gt_1e2=o.get("params_rel_gt_1e2")))
    m_on = _load(gd / "g2_7_on" / "mem_summary.json")
    m_off = _load(gd / "g2_7_off" / "mem_summary.json")
    off_rc = (gd / "g2_7_off" / "rc").read_text().strip() if (gd / "g2_7_off" / "rc").exists() else None
    put("G2-7", m_on is not None and m_on.get("hold_all_graphs") and m_on.get("isolation") and m_on["frac"] <= 0.8,
        dict(on=m_on, off=m_off, off_rc=off_rc, criterion="격리 켬 6 그래프 동시 보유 최대 ≤ 0.8 × (상한 또는 장치)"))
    sr = _load(gd / "sigma_ref.json")
    v = None if sr is None else sr.get("sigma_ref")
    put("G2-5", v is not None and isinstance(v, (int, float)) and math.isfinite(v) and v > 0, sr)
    res["G2-6(기록)"] = dict(ok=True, detail=_load(gd / "g2_6.json"))
    out = dict(approved_criteria=bool(approved), fails=fails, passed=not fails, checks=res)
    return out


def approval_valid(path, init):
    """→ (유효 여부, 사유). 빈 파일·결정 일자 없음·대상 INIT 없음이면 불승인."""
    if path is None or not Path(path).is_file():
        return False, "승인 파일 없음"
    txt = Path(path).read_text(errors="ignore")
    if not txt.strip():
        return False, "빈 승인 파일"
    if not re.search(r"\b20\d\d-\d\d-\d\d\b", txt):
        return False, "결정 일자(YYYY-MM-DD) 없음"
    if not init or not (init in txt or os.path.abspath(init) in txt):
        return False, f"대상 INIT({init}) 기재 없음"
    return True, "유효"


def _proc_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def claim(d, mode, gpu, runs, pid):
    d = Path(d)
    cd = d / ".claims"
    cd.mkdir(parents=True, exist_ok=True)
    mine = cd / f"{mode}_gpu{gpu}.json"
    with open(d / ".claims.lock", "a") as lk:
        fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        conflicts = []
        for f in sorted(cd.glob("*.json")):
            if f == mine:
                continue
            c = json.loads(f.read_text())
            alive = _proc_start(c["pid"]) is not None and _proc_start(c["pid"]) == c.get("start")
            if not alive:
                continue
            ov = sorted(set(c["runs"]) & set(runs))
            if ov:
                conflicts.append(dict(claim=f.name, pid=c["pid"], overlap=ov))
        if conflicts:
            return False, conflicts
        old = _load(mine)
        if old is not None and old["pid"] != pid and _proc_start(old["pid"]) == old.get("start") and old.get("start") is not None:
            return False, [dict(claim=mine.name, pid=old["pid"], overlap="같은 mode·GPU 체인이 살아 있다")]
        tmp = mine.with_name(mine.name + ".tmp")
        tmp.write_text(json.dumps(dict(pid=pid, start=_proc_start(pid), mode=mode, gpu=gpu, runs=list(runs))))
        os.replace(tmp, mine)
    return True, []


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gates")
    g.add_argument("--gates-dir", required=True)
    g.add_argument("--approved", default=None, help="사용자 결정 기록 파일 — 유효할 때만 바뀐 기준")
    g.add_argument("--init", required=True, help="시작 체크포인트(승인 파일에 적혀 있어야 함)")
    c = sub.add_parser("claim")
    c.add_argument("--dir", required=True)
    c.add_argument("--mode", required=True, choices=["runs", "eval"])
    c.add_argument("--gpu", required=True)
    c.add_argument("--runs", required=True)
    c.add_argument("--pid", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "gates":
        appr, why = approval_valid(a.approved, a.init)
        r = judge_gates(a.gates_dir, appr)
        r["approval_status"] = why
        if a.approved and Path(a.approved).is_file():
            r["approval_record"] = Path(a.approved).read_text(errors="ignore")[:2000]
        Path(a.gates_dir, "gates_check.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(json.dumps(dict(passed=r["passed"], fails=r["fails"], approved_criteria=appr), ensure_ascii=False))
        sys.exit(0 if r["passed"] else 1)
    runs = a.runs.split()
    bad = [r for r in runs if r not in ALL_RUNS]
    if bad:
        print(f"알 수 없는 런 {bad}"); sys.exit(1)
    ok, conf = claim(a.dir, a.mode, a.gpu, runs, a.pid)
    if not ok:
        print(json.dumps(dict(ok=False, conflicts=conf), ensure_ascii=False)); sys.exit(1)
    print(json.dumps(dict(ok=True, mode=a.mode, gpu=a.gpu, runs=runs), ensure_ascii=False))


if __name__ == "__main__":
    main()
