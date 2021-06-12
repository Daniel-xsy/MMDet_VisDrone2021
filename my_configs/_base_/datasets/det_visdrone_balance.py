dataset_type = 'VisdroneDataset'
classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
data_root = '/data/data1/lishuai/VisDrone2019/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Resize', img_scale=(416, 416), ratio_range=(0.7, 1.4), keep_ratio=True),
    # dict(type='Resize', img_scale=(416, 416),keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2000,1500),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + 'patch_train_coco.json',
            img_prefix=data_root + 'patch/images/', 
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'VisDrone2019-DET-val_coco.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'VisDrone2019-DET-val_coco.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
