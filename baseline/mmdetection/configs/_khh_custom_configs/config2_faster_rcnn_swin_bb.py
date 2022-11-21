from mmdet.utils import get_device

# backbone만 swintransformer 교체하는 경우

_base_ = [
    './_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/datasets/trash_detection.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/runtimes/default_runtime.py'
]

seed = 42
gpu_ids = [0]
device = get_device()

log_dir_path = '/opt/level2_objectdetection_cv-level2-cv-04/baseline/mmdetection/configs/_khh_custom_configs/_log'
work_dir = log_dir_path + '/log_config2_faster_rcnn_swin' # save in dir where "/log_1_config1_faster_rcnn"

checkpoint_config = dict(max_keep_ckpts=3, interval=1)

lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=15)


pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    type='FasterRCNN',
    backbone=dict(
        _delete_=True, # 기존 bb과 그 안에 있는 params를 모두 삭제하겠다.
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head = dict(bbox_head = dict(num_classes = 10))
    )




