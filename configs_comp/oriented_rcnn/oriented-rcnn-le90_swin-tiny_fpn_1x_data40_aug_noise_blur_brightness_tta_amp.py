_base_ = './oriented-rcnn-le90_r50_fpn_1x_dota.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

depths = [2, 2, 6, 2]
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        _delete_=True,
        type='mmdet.FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))


train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # dict(type='mmdet.CachedMosaic', img_scale=(1024, 1024), pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        resize_type='mmdet.Resize',
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=[9, 11]),
    dict(type='mmdet.RandomCrop', crop_size=(1024, 1024)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.CachedMixUp',
        img_scale=(1024, 1024),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        pad_val=(114, 114, 114)),
    dict(type="RandomBlur", prob=0.5, value_range=[3, 15]),
    dict(type="RandomNoise", prob=0.5, sigma_range=[3, 25]),
    dict(type="RandomBrightness", prob=0.5, gamma_range=[0.2, 1.0]),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.RandomResize',
        resize_type='mmdet.Resize',
        scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=[9, 11]),
    dict(type='mmdet.RandomCrop', crop_size=(1024, 1024)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(type="RandomBlur", prob=0.2, value_range=[3, 15]),
    dict(type="RandomNoise", prob=0.2, sigma_range=[3, 25]),
    dict(type="RandomBrightness", prob=0.2, gamma_range=[0.2, 1.0]),
    dict(type='mmdet.PackDetInputs')
]

# batch_size = (2 GPUs) x (4 samples per GPU) = 8
train_dataloader = dict(
    batch_size=1, num_workers=4, dataset=dict(pipeline=train_pipeline))

test_evaluator = dict(
    type='DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix=
    './work_dirs/oriented-rcnn_test/oriented-rcnn-le90_swin-tiny_fpn_1x_data40_aug_noise_blur_brightness_tta'
)

max_epochs = 12
stage2_num_epochs = 6
custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

optim_wrapper = dict(type='AmpOptimWrapper')