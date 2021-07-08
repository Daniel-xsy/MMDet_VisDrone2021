_base_ = [
    # '../my_configs/_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/det_visdrone_p3.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='SABLRetinaHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        approx_anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        square_anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[4],
            strides=[4, 8, 16, 32, 64]),
        # norm_cfg = dict(type='GN', num_groups=32, requires_grad=True),
        bbox_coder=dict(
            type='BucketingBBoxCoder', num_buckets=14, scale_factor=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.5),
        loss_bbox_reg=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.5)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='ApproxMaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=4000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=500))

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
work_dir_prefix = '/data/data1/lishuai/work_dir/ICCV2021_Workshop_VisDrone'
work_dir = work_dir_prefix + '/sabl_retina_baseline'
