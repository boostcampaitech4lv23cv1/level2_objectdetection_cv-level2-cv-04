from mmdet.utils import get_device

## detector를 cascade로 교체하는 경우

_base_ = [
    './_base_/models/cascade_rcnn_r50_fpn.py',
    './_base_/datasets/trash_detection.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/runtimes/default_runtime.py'
]


# "./_base_/models/cascade_rcnn_r50_fpn.py"에 들어가서 num_classes=10으로 모두 변경


seed = 42
gpu_ids = [0]
log_dir_path = '/opt/level2_objectdetection_cv-level2-cv-04/baseline/mmdetection/configs/_khh_custom_configs/_log'
work_dir = log_dir_path + '/log_config3_cascade_rcnn' 
runner = dict(type='EpochBasedRunner', max_epochs=15)
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
device = get_device()