_base_ = [
    '../_base_/models/daformer_mit-b5_aug.py',
    '../_base_/datasets/synthia_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
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
                norm_cfg=norm_cfg))),
    decode_head0=dict(
        num_classes=19,
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))),              
    decode_head1=dict(
        num_classes=19,
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))),                      
    decode_head2=dict(
        num_classes=19,
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))),       
    decode_head3=dict(
        num_classes=19,
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))),                 
    decode_head4=dict(
        num_classes=19,
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))),                   
    decode_head5=dict(
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
    lr=4e-6,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

lr_config = dict(
    policy='poly',
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


data = dict(samples_per_gpu=2)
find_unused_parameters = True