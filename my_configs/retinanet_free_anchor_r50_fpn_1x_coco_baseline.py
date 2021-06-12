_base_ = ['../my_configs/_base_/models/retinanet_r50_fpn.py',
          '../my_configs/_base_/datasets/det_visdrone.py',
          '../my_configs/_base_/schedules/schedule_1x.py',
          '../my_configs/_base_/default_runtime.py'


]

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

model = dict(
    bbox_head=dict(
        _delete_=True,
        type='FreeAnchorRetinaHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.75)))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
