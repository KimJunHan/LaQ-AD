#수정: VLA-FULL-LITE — full nuScenes info pkl 을 scene 단위로 deterministic 부분집합.
#  목적: full(28130/6019)은 캐시 116GB·다GPU-시간으로 무겁다. mini(323/81)는 eval 노이즈
#  바닥이 너무 높다. 그 사이 "light" 지점 = trainval 의 1/N scene 만 사용.
#  scene 을 통째로(샘플 순서 유지) 잘라 temporal instance propagation 을 깨지 않는다.
#  매 N번째 고유 scene 을 고름(정렬된 scene_token 기준) → train/val 동일 규칙, 재현 가능.
#
#  사용:
#    python tools/data_converter/subset_nusc_infos.py \
#        --in data/infos/nuscenes_infos_train.pkl --out data/infos/nuscenes_infos_train_lite.pkl --every 4
#    python tools/data_converter/subset_nusc_infos.py \
#        --in data/infos/nuscenes_infos_val.pkl   --out data/infos/nuscenes_infos_val_lite.pkl   --every 4
import argparse
import pickle


def subset(in_path, out_path, every):
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    is_dict = isinstance(data, dict)
    infos = data["infos"] if is_dict else data

    # 등장 순서 보존하면서 고유 scene 수집 (정렬은 안 함 — 변환 시 scene 묶음 순서 유지가
    # temporal 연속성에 자연스럽다. deterministic = 입력 pkl 이 deterministic 하므로 보장).
    seen = []
    seen_set = set()
    for it in infos:
        st = it["scene_token"]
        if st not in seen_set:
            seen_set.add(st)
            seen.append(st)
    keep = set(seen[::every])  # 매 N번째 scene

    sub = [it for it in infos if it["scene_token"] in keep]
    n_scene_total, n_scene_keep = len(seen), len(keep)
    print(f"{in_path}: scenes {n_scene_total} -> {n_scene_keep} (every {every}) | "
          f"samples {len(infos)} -> {len(sub)}")

    if is_dict:
        out = dict(data)
        out["infos"] = sub
    else:
        out = sub
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"  saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--every", type=int, default=4, help="매 N번째 scene 만 보존 (1/N 부분집합)")
    args = ap.parse_args()
    subset(args.in_path, args.out_path, args.every)


if __name__ == "__main__":
    main()
