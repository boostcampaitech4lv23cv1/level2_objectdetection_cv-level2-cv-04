# need to copy and paste config faster_rcnn_r50_fpn.py
_base_ = [
    './_base_/models/faster_rcnn_r101_fpn_1x_coco.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]