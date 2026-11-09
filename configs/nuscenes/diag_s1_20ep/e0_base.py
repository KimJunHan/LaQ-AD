# 수정(2026-10-27): 사용자 지시 "1단계 nuScenes 미달 원인 후보 1~3 확인" — 20 에폭 비교 실험의 기준(E0).
#   ../v2_teacher/stage1/train.py(본 1단계, 100 에폭)와 레시피·모델이 같고 에폭 수만 20(코사인 일정도 20 에폭에 맞춰 끝남).
#   다른 비교(E1~E4)는 이 파일을 상속해 한 가지씩만 바꾼다. 결과는 E0 과만 비교한다(100 에폭 수치와 직접 비교하지 않음).
_base_ = ["../v2_teacher/stage1/train.py"]
batch_size = 16
cumulative_iters = 4
num_epochs = 20
num_iters_per_epoch = int(28130 // (batch_size * cumulative_iters)) * cumulative_iters   # 1,756 (본 1단계와 같음)
runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)        # 35,120
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=1)
data = dict(workers_per_gpu=8)  # 수정(2026-10-27): 비교 5개 동시 실행 — CPU 경합 완화(값만 읽는 쪽이라 결과 불변)
run_name = "diag_s1_20ep_e0_base"
