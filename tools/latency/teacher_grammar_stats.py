# 수정(2026-11-04): 교사 생성 비용 분석 — 정답 문법(answer_grammar_v1)의 위치·폴더별 토큰 수, 문법이 한 토큰만 허용하는
#   '강제' 위치(머리말·<none>/<na> 뒤 채움)의 비율을 val 표본의 실제 생성 답(_index gen_ids)으로 잰다. CPU 전용, 읽기만 한다.
#   강제 위치는 탐욕 제약 생성에서도 값이 정해져 있으므로 샘플링 없이 한 번의 순전파로 묶어 넣을 수 있다(jump-forward) — 결과·은닉 동일.
#   사용: CUDA_VISIBLE_DEVICES="" python tools/latency/teacher_grammar_stats.py --out results/latency/teacher_grammar_stats.json
import argparse
import json
import os
import pickle
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "vla_teacher"))
os.chdir(REPO)
import laq_cache_layout as LY  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import answer_grammar as AG
    tok = AG._tokenizer()
    g = AG.load_grammar(tok, check_train_sha=False)
    tmpl = LY.answer_template(tok)
    pos2f = {}
    for f, ps in tmpl.items():
        for p in ps:
            pos2f.setdefault(p, []).append(f)
    val = {i["token"] for i in pickle.load(open("data/infos/nuscenes_infos_val.pkl", "rb"))["infos"]}
    n = 0
    forced = [0] * LY.ANSWER_TOKENS
    nonempty_slots = []
    last_nonforced = {f: [] for f in LY.ANSWER_FOLDERS}
    for T in ("A", "B"):
        for line in open(f"data/nuscenes/laq_cache/_index/teacher_{T}.jsonl"):
            e = json.loads(line)
            if e["token"] not in val:
                continue
            ids = e["gen_ids"]
            keys = g.replay_keys(ids)
            fz = [len(g._set(k)) == 1 for k in keys]
            for t in range(LY.ANSWER_TOKENS):
                forced[t] += fz[t]
            ne = sum(1 for k in range(LY.N_SLOTS) if ids[3 + k * LY.SLOT] != g.none)
            nonempty_slots.append(ne)
            n += 1
            if a.limit and n >= a.limit:
                break
        if a.limit and n >= a.limit:
            break
    per_folder = {}
    for f, ps in tmpl.items():
        per_folder[f] = dict(n_tokens=len(ps), first=min(ps), last=max(ps),
                             forced_frac=sum(forced[p] for p in ps) / (n * len(ps)))
    headers = [t for t in range(LY.ANSWER_TOKENS) if t not in pos2f]
    import collections
    out = dict(n_val=n, per_folder=per_folder, header_positions=headers,
               forced_frac_all=sum(forced) / (n * LY.ANSWER_TOKENS),
               mean_forced_tokens=sum(forced) / n,
               forced_per_pos=[x / n for x in forced],
               prefix_cost={str(c): dict(steps=c, mean_forced=sum(forced[:c]) / n, mean_free=c - sum(forced[:c]) / n)
                            for c in (423, 428, 458, 459, 531, 547)},
               nonempty_slots_hist=dict(sorted(collections.Counter(nonempty_slots).items())))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
