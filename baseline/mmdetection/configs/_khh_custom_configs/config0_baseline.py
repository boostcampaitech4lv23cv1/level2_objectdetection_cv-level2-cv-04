# need to copy and paste config faster_rcnn_r50_fpn.py
_base_ = [
    './_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/datasets/trash_detection.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/runtimes/default_runtime.py'
]