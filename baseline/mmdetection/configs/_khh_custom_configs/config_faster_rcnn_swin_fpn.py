# need to copy and paste config faster_rcnn_r50_fpn.py
_base_ = [
    './_base_/models/swin.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]