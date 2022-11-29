## mosaic config는 두번째 train_script에 종속됩니다.
## 즉, train_script에서 강제로 img_size를 할당을 할때, 이 config를 수정하지 않으면 오류가 발생합니다.
## 지금은 리사이즈 이후가 512, 512임을 가정합니다

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

_base_ = [
    './_base_/models/faster_rcnn_swin_tiny_fpn.py',
    './_base_/datasets/coco_detection_size512.py',
    './_base_/schedules/schedule_1x_adamw.py',
    '../_base_/default_runtime.py'
]

img_scale = (512,512) # ⭐️ ← parser로 전달한 img_size와 일치해야 합니다.

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

optimizer = dict(betas=(0.9, 0.999), lr=0.0001, type='AdamW', weight_decay=0.05)

### Mosaic augmentation
train_pipeline = [
    # dict(type='Mosaic', img_scale=img_scale, pad_val=0), # , pad_val=114.0
    dict(type='MixUp', img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=0),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    _delete_ = True, # remove unnecessary Settings
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        classes = classes,
        ann_file='/opt/bro/folded_anns/train_3.json',
        img_prefix='/opt/bro/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline
    )

data = dict(
    train=train_dataset
    )