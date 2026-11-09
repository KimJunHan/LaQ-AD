# 수정(2026-11-01): 사용자 결정 "5. 학습돌려두자" — E3 4-GPU 1단계(./train_e3_ddp4.py)와 모든 값이 같고 이미지 백본 초기값만
#   ImageNet(ckpts/resnet50-19c8e357.pth) → nuImages 사전학습 R50 으로 바꾼다(100 에폭 처음부터, EMA 함께 저장).
#   가중치: mmdet3d model zoo cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth
#     (sha256 4096396018c0cf59…, 파일명 해시와 일치). 그 안의 backbone.* 318 키(BN num_batches_tracked 포함)만 읽는다 — FPN 은
#     구조가 달라(우리 FPN 은 BN·extra conv) 읽지 않는다. 키·모양은 ImageNet R50 과 같다(fc 제외).
#     그 검출기는 frozen_stages=1·norm_eval=True 로 학습해 stem·layer1 가중치와 BN 이동 통계는 ImageNet 값 그대로이고
#     layer2~4 가중치만 nuImages 로 바뀌어 있다(실측: conv1 차이 0, layer2.0.conv1 평균 |차| 0.031).
#   [비교 조건] HiP-AD·SparseDrive 원본은 ImageNet 초기값 → 이 결과는 "nuImages 사전학습 백본" 으로 따로 표기한다.
_base_ = ["./train_e3_ddp4.py"]
model = dict(img_backbone=dict(
    pretrained=None,
    init_cfg=dict(type="Pretrained", prefix="backbone.",
                  checkpoint="ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth")))
run_name = "hipad_nusc_laq_v2_teacher_stage1_e3_nuim_ddp4"
