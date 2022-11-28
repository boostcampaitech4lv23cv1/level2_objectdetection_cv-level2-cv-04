_base_ = [
    './_base_/models/swin.py',
    './_base_/datasets/coco_detection_size512.py',
    './_base_/schedules/schedule_1x_adamw.py',
    '../_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.5),
    
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],p=0.5),

    dict(type='ChannelShuffle', p=0.1),


    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussNoise', var_limit=(10.0, 400.0)),
            dict(type='RandomBrightnessContrast'),
            dict(type='HueSaturationValue')
        ],
        p=0.5),
        
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3),
            dict(type='MedianBlur', blur_limit=3)
        ],
        p=0.5)
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True), # script에서 이미지 resize
    dict(type='RandomFlip', flip_ratio=0.8), # 추가
    # dict(type='Normalize', **img_norm_cfg), # 추가
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms, # 여기에서 albu가 들어감
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            # 'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg), # 추가
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'], # , 'gt_masks' 필요 없으므로 삭제
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]
data = dict(train=dict(pipeline=train_pipeline))
optimizer = dict(betas=(0.9, 0.999), lr=0.0001, type='AdamW', weight_decay=0.05)