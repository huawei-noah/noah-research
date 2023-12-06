_base_ = [
    '../_base_/models/daformer_mit-b5_prompt.py',
    '../_base_/datasets/cityscapes_half_512x512.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(freeze_backbone=True),
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
    lr=6e-6,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            level_embed=dict(lr_mult=5.0),
            spm=dict(lr_mult=5.0),
            interactions=dict(lr_mult=5.0),
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
            
lr_config = dict(
    policy='CosineAnnealing',
    warmup=None,
    min_lr_ratio=1e-5)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)

