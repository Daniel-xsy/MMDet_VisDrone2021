_base_ = [
    '../my_configs/_base_/models/retinanet_r50_fpn.py',
    '../my_configs/_base_/datasets/det_visdrone.py',
    '../my_configs/_base_/schedules/schedule_1x.py', 
    '../my_configs/_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        num_classes=10
    )
)

optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)

gpu_ids = 1