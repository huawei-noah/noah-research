# Modified from https://github.com/lhoyer/DAFormer/


# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='MultiscaleEncoderDecoderPrompt',
    backbone=dict(type='mit_b5_prompt', style='pytorch'),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='conv',
                kernel_size=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg),
        ),
        loss_decode=dict(
            type='GtASelfTrainingLoss', 
            loss_weight=1.0,
            )),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
