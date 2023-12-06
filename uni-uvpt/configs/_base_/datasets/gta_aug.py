# Modified from https://github.com/lhoyer/DAFormer/

# dataset settings
dataset_type = 'GTADataset'

# data_root_source = '/home/ma-user/modelarts/inputs/data_url_0_0/'
# data_root_target = '/home/ma-user/modelarts/inputs/data_url_1_1/'
data_root_source = '/cache/gta/'
data_root_target = '/cache/cityscapes/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
gta_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Adain', adain=0.3),
    dict(type='Styleaug'),
    dict(type='Imgaug', imgaug='snow'),
    dict(type='Imgaug', imgaug='frost'),
    dict(type='Imgaug', imgaug='cartoon'),
    dict(type='Fda', fda=('random', 0.005)),
    dict(type='Fda', fda=('input/style', 0.005)),
    dict(type='NormalizeAug', **img_norm_cfg),
    dict(type='PadAug', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundleAug'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 
    'img_adain', 
    'img_styleaug', 'img_snow', 'img_frost', 'img_cartoon', 
    'img_fda_random', 'img_fda']),
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2, 
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=2, workers_per_gpu=4),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    train=dict(
        type='GTADataset',
        data_root=data_root_source,
        img_dir='images',
        ann_dir='labels',
        pipeline=gta_train_pipeline),
    val=dict(
        type='CityscapesDataset',
        data_root=data_root_target,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type='CityscapesDataset',
        data_root=data_root_target,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))