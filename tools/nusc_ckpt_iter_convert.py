"""수정(2026-10-10): GPU 수·micro-batch 변경 시 체크포인트 재개 위치 환산.
  같은 유효 배치(64)에서 config 마다 micro-iter 대 optimizer step 비율(cumulative_iters)이 다르다.
  lr 일정·warmup·max_iters 는 micro-iter 단위이므로, optimizer step 위치를 보존하도록 meta.iter 를 환산한다.
    새 iter = 옛 iter × (새 cumulative_iters / 옛 cumulative_iters)     (옛 iter 는 누적 경계에서 저장됨을 확인)
  예: 2-GPU(16×2GPU×2) iter 2,000 → 1-GPU(16×4) iter 4,000. 가중치·optimizer 상태는 그대로.
  사용: python tools/nusc_ckpt_iter_convert.py <in.pth> <out.pth> <old_cum> <new_cum>
"""
import sys, torch
src, dst, old_c, new_c = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ck = torch.load(src, map_location="cpu", weights_only=False)
it = ck["meta"]["iter"]
# 수정(2026-10-12): mmcv IterBasedRunner 는 meta.iter 에 "마지막 완료 iter 의 0-기반 번호"(완료 수 - 1)를 저장한다
#   (파일명 iter_{완료 수}.pth). 재개 시 runner._iter = meta.iter. 환산은 완료 수 기준으로 하고 같은 규약으로 되돌린다.
done = it + 1
assert done % old_c == 0, f"완료 iter {done} 가 누적 경계가 아니다(cumulative_iters={old_c}) — 누적 중간 기울기가 버려진다"
ck["meta"]["iter"] = done * new_c // old_c - 1
ck["meta"]["epoch"] = ck["meta"].get("epoch", 0)
ck["meta"]["converted_from"] = dict(src=src, iter=it, old_cum=old_c, new_cum=new_c)
torch.save(ck, dst)
print(f"완료 iter {done} → {ck['meta']['iter'] + 1} (meta.iter {it} → {ck['meta']['iter']}, optimizer step {done // old_c})  -> {dst}")
