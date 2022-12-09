_base_ = [
    '/opt/ml/level2_objectdetection_cv-level2-cv-04/baseline/mmdetection/configs/_chj_custom_configs/faster_rcnn_r50_PAFPN.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]