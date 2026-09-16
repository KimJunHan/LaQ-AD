# 수정: VLA-v0 — Bench2Drive visualize.py 가 뽑은 per-cam 3D-bbox 프레임들을
# HiP-AD 논문(hipad.pdf) qualitative figure 처럼 "6-cam(2x3) | top-down" 한 장으로 합친다.
# 입력 : visualize.py 출력 디렉토리 (<scenario>/camera/{rgb_*_3d_bbox, rgb_top_down_3d_bbox}/XXXXX.jpg)
# 출력 : <scenario>/combine/XXXXX.jpg  +  <scenario>/combine.mp4
# 레이아웃:
#   [ FRONT_LEFT  FRONT  FRONT_RIGHT ]
#   [ BACK_LEFT   BACK   BACK_RIGHT  ]  | [ TOP_DOWN ]
import argparse
import glob
import os

import cv2
import numpy as np

# 2x3 surround grid 순서 (nuScenes/HiP-AD figure 관례)
GRID = [
    ["rgb_front_left_3d_bbox", "rgb_front_3d_bbox", "rgb_front_right_3d_bbox"],
    ["rgb_back_left_3d_bbox",  "rgb_back_3d_bbox",  "rgb_back_right_3d_bbox"],
]
LABELS = [
    ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"],
    ["BACK_LEFT",  "BACK",  "BACK_RIGHT"],
]
TOPDOWN = "rgb_top_down_3d_bbox"


def _put_label(img, text):
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def combine_one(cam_dir, step, cam_w=480, pad=4, bev_size=200):
    """한 프레임 합성. 누락 패널은 검정으로 채움."""
    cam_h = None  # cam_w 비율로 결정

    def load(sub):
        p = os.path.join(cam_dir, sub, f"{step:05}.jpg")
        return cv2.imread(p) if os.path.exists(p) else None

    # 그리드 셀 크기 기준 = FRONT 의 종횡비
    ref = load("rgb_front_3d_bbox")
    if ref is None:
        # 아무 셀이나
        for row in GRID:
            for sub in row:
                ref = load(sub)
                if ref is not None:
                    break
            if ref is not None:
                break
    if ref is None:
        return None
    h0, w0 = ref.shape[:2]
    cam_h = int(cam_w * h0 / w0)

    def cell(sub, label):
        im = load(sub)
        if im is None:
            im = np.zeros((h0, w0, 3), dtype=np.uint8)
        im = cv2.resize(im, (cam_w, cam_h))
        return _put_label(im, label)

    rows = []
    for r in range(2):
        cells = [cell(GRID[r][c], LABELS[r][c]) for c in range(3)]
        cells = [np.pad(c, ((0, 0), (pad, pad), (0, 0))) for c in cells]
        rows.append(np.concatenate(cells, axis=1))
    grid = np.concatenate([np.pad(rows[0], ((pad, pad), (0, 0), (0, 0))),
                           np.pad(rows[1], ((pad, pad), (0, 0), (0, 0)))], axis=0)

    grid_h = grid.shape[0]
    # 수정: top-down(BEV) 은 중앙 정사각 크롭 후 bev_size x bev_size 로 고정(기본 200x200).
    #       왜곡 방지 위해 짧은 변 기준 center-crop. 그리드 높이에 맞춰 세로 가운데 정렬(검정 패딩).
    td = load(TOPDOWN)
    if td is None:
        td = np.zeros((900, 1600, 3), dtype=np.uint8)
    th, tw = td.shape[:2]
    side = min(th, tw)
    y0 = (th - side) // 2
    x0 = (tw - side) // 2
    td = td[y0:y0 + side, x0:x0 + side]          # center square crop
    td = cv2.resize(td, (bev_size, bev_size))    # 200x200 고정
    td = _put_label(td, "BEV")
    # 그리드 높이에 맞춰 세로 가운데, 좌측 여백 패딩
    pad_top = max((grid_h - bev_size) // 2, 0)
    pad_bot = max(grid_h - bev_size - pad_top, 0)
    td = np.pad(td, ((pad_top, pad_bot), (pad * 2, pad * 2), (0, 0)))
    # 혹시 라운딩으로 높이 안 맞으면 보정
    if td.shape[0] != grid_h:
        td = cv2.resize(td, (td.shape[1], grid_h))

    return np.concatenate([grid, td], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", "-s", required=True,
                    help="visualize.py 가 만든 <scenario> 디렉토리 (camera/ 하위 포함)")
    ap.add_argument("--cam-w", type=int, default=480, help="그리드 셀 한 칸 가로 px")
    ap.add_argument("--bev-size", type=int, default=200, help="우측 BEV(top-down) 정사각 크기 px")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--interval", type=int, default=1)
    args = ap.parse_args()

    cam_dir = os.path.join(args.scenario_dir, "camera")
    front = sorted(glob.glob(os.path.join(cam_dir, "rgb_front_3d_bbox", "*.jpg")))
    if not front:
        raise SystemExit(f"no rendered frames under {cam_dir}/rgb_front_3d_bbox — run visualize.py first")
    n = len(front)
    end = args.end if args.end is not None else n
    steps = list(range(args.start, end, args.interval))

    out_dir = os.path.join(args.scenario_dir, "combine")
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for k, step in enumerate(steps):
        img = combine_one(cam_dir, step, cam_w=args.cam_w, bev_size=args.bev_size)
        if img is None:
            continue
        sp = os.path.join(out_dir, f"{step:05}.jpg")
        cv2.imwrite(sp, img)
        saved.append(sp)
        if k % 20 == 0:
            print(f"[{k+1}/{len(steps)}] {os.path.basename(sp)}  {img.shape}")

    if saved:
        H, W = cv2.imread(saved[0]).shape[:2]
        vp = os.path.join(args.scenario_dir, "combine.mp4")
        vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        for p in saved:
            vw.write(cv2.imread(p))
        vw.release()
        print(f"video: {vp}  ({len(saved)} frames @ {args.fps}fps, {W}x{H})")
    print(f"DONE: {len(saved)} combined imgs in {out_dir}")


if __name__ == "__main__":
    main()
