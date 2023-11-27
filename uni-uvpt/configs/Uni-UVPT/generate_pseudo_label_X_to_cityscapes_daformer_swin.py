_base_ = [
    '../_base_/models/daformer_swin.py',
    '../_base_/datasets/cityscapes_half_512x512.py',
    '../_base_/default_runtime.py',
]

norm_cfg = dict(type='BN', requires_grad=True)
checkpoint_file = 'model/swin_base_patch4_window7_224_20220317-e9b98025.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(
        num_classes=19,
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))))


env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
# evaluation = dict(metric="mIoU")
checkpoint_config = dict(by_epoch=False, interval=2000)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
optimizer = dict(type='AdamW', lr=3e-06, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0), pos_block=dict(decay_mult=0.0),
                                                     norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='CosineAnnealing',
    warmup=None,
    min_lr_ratio=1e-5)
runner = dict(type='IterBasedRunner', max_iters=40000)
evaluation = dict(interval=1000, metric="mIoU")
workflow = [('train', 1)]
