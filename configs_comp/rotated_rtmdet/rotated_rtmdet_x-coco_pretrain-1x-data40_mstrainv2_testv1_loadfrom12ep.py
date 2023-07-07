_base_ = './rotated_rtmdet_x-1x-data40_msv2.py'

coco_ckpt = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=coco_ckpt)),
    neck=dict(
        init_cfg=dict(type='Pretrained', prefix='neck.',
                      checkpoint=coco_ckpt)),
    bbox_head=dict(
        init_cfg=dict(
            type='Pretrained', prefix='bbox_head.', checkpoint=coco_ckpt))
)

# batch_size = (2 GPUs) x (4 samples per GPU) = 8
train_dataloader = dict(batch_size=2, num_workers=8)

load_from = "work_dirs/rotated_rtmdet_x-coco_pretrain-1x-data40_ms/epoch_12.pth"

test_evaluator = dict(
    type='DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/rtmdet_r/x_ms_1x_loadep12_mstrainv2_testv1')