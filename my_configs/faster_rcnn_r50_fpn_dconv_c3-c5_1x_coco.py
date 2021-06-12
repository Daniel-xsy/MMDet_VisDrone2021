#_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

_base_ = [
    '../my_configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../my_configs/_base_/datasets/det_visdrone.py',
    '../my_configs/_base_/schedules/schedule_1x.py', 
    '../my_configs/_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
