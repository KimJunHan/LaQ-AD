# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   B2D 교사 캐시(7 폴더) 로더. LoadTeacherCacheNusc(./loading.py, 수정하지 않음)의 엄격 검사(파일·형식·폴더별 고정 길이·
#   _meta.json 기대값·색인 layout_ok/constrained_decoding)를 그대로 물려받고, "어느 파일을 읽는가" 만 B2D 규칙으로 바꾼다.
#
#   [2 Hz 키프레임 규칙 — 사용자 결정 2026-10-16/19, 상위 작업자 사양 변경 2026-10-19]
#     교사 출력은 2 Hz 키프레임에서만 만든다(캐시도 키프레임만). 클립마다 f0 = infos 에서 그 클립의 첫 frame_idx,
#     키프레임 = f0, f0+5, f0+10, ... (대부분 f0 = 0 → frame_idx % 5 == 0; 일부 클립은 1 에서 시작 — 예 val
#     VehicleTurningRoutePedestrian_Town15_Route481_Weather19). 폐쇄루프에서 경로 첫 틱이 갱신 시점인 것과 같다.
#     10 Hz 학습·평가 프레임 f 는 "같은 클립에서 색인에 있는 키프레임 중 kf ≤ f 인 가장 큰 것" 을 쓴다.
#     f − kf > 4(0.4 s 초과) 이거나 그런 키프레임이 없으면 즉시 예외(다른 키프레임으로 대체하지 않음 — skip 금지).
#     키프레임 목록은 캐시 색인(_index/teacher_*.jsonl 의 token "<clip>/<kf:05d>")에서 클립별로 정렬해 만든다
#     → 키프레임 규칙(f0 기준)은 캐시 추출기가 정하고 로더는 그것을 그대로 따른다.
#   [경로] <cache_root>/<folder>/<clip>/<kf:05d>.pt   (clip = info['folder'] 의 마지막 이름, 'v1/' 제거)
#   results['teacher_age'] = (f - kf) / 10 s, results['teacher_key'] = "<clip>/<kf:05d>" 를 기록한다(진단용).
import os

from mmdet.datasets.builder import PIPELINES

from .loading import LoadTeacherCacheNusc

__all__ = ["LoadTeacherCacheB2D", "b2d_clip_name", "b2d_keyframes", "b2d_keyframe_for", "KEYFRAME_STRIDE"]

KEYFRAME_STRIDE = 5   # 10 Hz → 2 Hz (최대 나이 = STRIDE-1 프레임 = 0.4 s)


def b2d_clip_name(folder):
    clip = os.path.basename(str(folder).rstrip("/"))
    if not clip:
        raise ValueError(f"LoadTeacherCacheB2D: folder 이름이 비었다: {folder!r}")
    return clip


def b2d_keyframes(frame_idxs, stride=KEYFRAME_STRIDE):
    """한 클립의 infos frame_idx 목록 → 키프레임 목록(f0, f0+stride, ... ≤ 마지막 프레임). 캐시 추출기·폐쇄루프와 같은 규칙."""
    fs = sorted(int(f) for f in frame_idxs)
    if not fs:
        return []
    return list(range(fs[0], fs[-1] + 1, stride))


def b2d_keyframe_for(kfs, frame_idx, stride=KEYFRAME_STRIDE):
    """정렬된 키프레임 목록에서 kf ≤ f 인 가장 큰 것. 없거나 f − kf ≥ stride 이면 None."""
    import bisect
    f = int(frame_idx)
    i = bisect.bisect_right(kfs, f) - 1
    if i < 0 or f - kfs[i] >= stride:
        return None
    return kfs[i]


@PIPELINES.register_module()
class LoadTeacherCacheB2D(LoadTeacherCacheNusc):
    """LoadTeacherCacheNusc 와 같은 출력(qwen_hidden/qwen_mask/qwen_type, 고정 길이·pad 없음)·같은 검사.
    차이: 조회 키 = 현재 프레임의 2 Hz 키프레임 "<clip>/<kf:05d>" (클립별 색인 키프레임 중 kf ≤ frame_idx 최대, 나이 ≤ 0.4 s)."""

    def __init__(self, *args, keyframe_stride=KEYFRAME_STRIDE, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyframe_stride = int(keyframe_stride)
        if self.keyframe_stride < 1:
            raise ValueError(f"LoadTeacherCacheB2D: keyframe_stride {keyframe_stride} < 1")
        if self.index is None:
            # 키프레임 목록의 출처가 색인이므로 B2D 는 색인 필수(require_index=False 로 우회하지 않는다)
            raise ValueError("LoadTeacherCacheB2D: require_index=False 불가 — 키프레임 목록을 캐시 색인에서 만든다")
        kfs = {}
        for tok in self.index:
            clip, _, kf = tok.rpartition("/")
            if not clip or not kf.isdigit() or len(kf) != 5:
                raise ValueError(f"LoadTeacherCacheB2D: 색인 token 형식 오류 {tok!r} (기대 '<clip>/<kf:05d>')")
            kfs.setdefault(clip, []).append(int(kf))
        self.clip_keyframes = {c: sorted(v) for c, v in kfs.items()}
        # 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU) — 빈 프레임 보충 키프레임 허용.
        #   B2D train infos 에는 clip 중간에 프레임이 하나 빠진 곳이 있다(ConstructionObstacleTwoWays_Town12_Route1410 frame 325,
        #   …Route1413 frame 315 — gen_teacher_gt_b2d extract 실측). 격자 키프레임이 없으면 그 뒤 4 프레임이 쓸 교사 출력이 없으므로
        #   생성기가 "격자 칸 안의 첫 실제 프레임"(326, 316)을 보충 키프레임으로 만든다(나이 ≤ 0.4 s 유지, 폐쇄루프 규칙과 같은 상한).
        #   색인 정합: 격자 밖 키프레임은 그 칸의 격자 키프레임이 없고 칸 안(격자 + 1..stride−1)에 있을 때만 허용.
        for c, v in self.clip_keyframes.items():
            on = set(b for b in v if (b - v[0]) % self.keyframe_stride == 0)
            for b in v:
                if b in on:
                    continue
                g = v[0] + self.keyframe_stride * ((b - v[0]) // self.keyframe_stride)
                if g in on or sum(1 for x in v if g < x < g + self.keyframe_stride) != 1:
                    raise ValueError(f"LoadTeacherCacheB2D: 색인 {c} 키프레임 {b} 가 f0={v[0]} 기준 {self.keyframe_stride} 간격 격자 밖이고 "
                                     f"빈 격자 칸 보충도 아니다: {v[:8]}")

    def __call__(self, input_dict):
        if "folder" not in input_dict or "frame_idx" not in input_dict:
            raise RuntimeError("LoadTeacherCacheB2D: input_dict 에 folder/frame_idx 가 없다(Bench2DriveDataset.get_data_info 확인).")
        clip = b2d_clip_name(input_dict["folder"])
        f = int(input_dict["frame_idx"])
        kfs = self.clip_keyframes.get(clip)
        if kfs is None:
            raise RuntimeError(f"교사 캐시 색인에 클립 {clip} 이 없다(추출 미완료) — {self.cache_root}/_index")
        kf = b2d_keyframe_for(kfs, f, self.keyframe_stride)
        if kf is None:
            raise RuntimeError(f"교사 캐시: {clip} frame {f} 에 쓸 키프레임(kf ≤ f, f−kf ≤ {self.keyframe_stride - 1}) 이 색인에 없다 "
                               f"(색인 키프레임 {kfs[:3]}…{kfs[-3:]})")
        key = f"{clip}/{kf:05d}"
        orig = input_dict.get("token")
        input_dict["token"] = key            # 부모 검사·경로·색인 조회가 이 키를 쓴다
        try:
            out = super().__call__(input_dict)
        finally:
            input_dict["token"] = orig       # 다른 단계(평가·로그)가 쓰는 원래 token 은 그대로 둔다
        out["teacher_key"] = key
        out["teacher_age"] = (f - kf) / 10.0
        return out
