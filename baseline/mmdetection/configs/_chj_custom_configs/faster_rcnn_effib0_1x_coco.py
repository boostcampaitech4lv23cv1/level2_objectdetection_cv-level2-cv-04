_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_ = True,
        type='EfficientNet',
        arch ='b0',
        out_indices=(0, 1, 3, 6)),
    neck = dict(
        type = 'PAFPN',
        in_channels = [24, 40, 112, 1280]
    )
        )