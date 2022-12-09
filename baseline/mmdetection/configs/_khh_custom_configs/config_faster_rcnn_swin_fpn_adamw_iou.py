# need to copy and paste config faster_rcnn_r50_fpn.py
_base_ = ['./_base_/models/faster_rcnn_swin_tiny_fpn_iou.py',
          './_base_/datasets/coco_detection.py',
          './_base_/schedules/schedule_1x_adamw.py',
          './_base_/default_runtime.py']

optimizer = dict(betas=(0.9, 0.999), lr=0.0001, type='AdamW', weight_decay=0.05)