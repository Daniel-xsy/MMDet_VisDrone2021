_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=100)))

dataset_type='VisDrone2019'
data_root='/data/VisDrone2021/'
data = dict(
    sample_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root+'patch/patch_train_coco.json',
        img_prefix=data_root+'patch/images/'
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root+'VisDrone2019-DET_val_coco.json',
        img_prefix=data_root+'VisDrone2019-DET-val/images/'
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root+'VisDrone2019-DET_test_coco.json',
        img_prefix=data_root+'VisDrone2019-DET-test-dev/images/'
    )
)