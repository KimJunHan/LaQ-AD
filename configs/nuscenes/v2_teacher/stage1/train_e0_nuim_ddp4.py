# 수정(2026-11-02): ./train_e0_ddp4.py + 이미지 백본 초기값만 nuImages 사전학습 R50(./train_e3_nuim_ddp4.py 와 같은 가중치·같은 방식).
#   사전 등록 규칙(scripts/nuscenes/run_stage1_e0_ddp4_20261102.sh): nuImages E3 EMA 가 ImageNet E3 EMA 보다 val mAP·NDS 둘 다 높을 때만 이 config 를 쓴다.
#   [비교 조건] HiP-AD·SparseDrive 원본은 ImageNet 초기값 → 이 config 로 학습하면 "nuImages 사전학습 백본" 으로 따로 표기한다.
_base_ = ["./train_e0_ddp4.py"]
model = dict(img_backbone=dict(
    pretrained=None,
    init_cfg=dict(type="Pretrained", prefix="backbone.",
                  checkpoint="ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth")))
run_name = "hipad_nusc_laq_v2_teacher_stage1_e0_nuim_ddp4"
