_base_ = [
    '../_base_/models/daformer_mit-b5.py',
    '../_base_/datasets/gta.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_400k.py'
]


norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
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

# AdamW optimizer
# optimizer_config = None
optimizer = dict(
    type='AdamW',
    lr=0.00002,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


data = dict(samples_per_gpu=2)
