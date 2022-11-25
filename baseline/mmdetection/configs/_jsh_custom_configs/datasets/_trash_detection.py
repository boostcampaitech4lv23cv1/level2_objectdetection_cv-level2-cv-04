# dataset settings
## make competition dataset

## 수정내용
# 1. dataset root 수정
# 2. test_pipeline에 존재하는 augmentation 삭제
# 3. img_scale 전부 512,512 수정

dataset_type = 'CocoDataset'

root='/opt/ml/dataset/' ## abs path

classes = ("General trash",
           "Paper", 
           "Paper pack", 
           "Metal", 
           "Glass", 
           "Plastic", 
           "Styrofoam", 
           "Plastic bag", 
           "Battery", 
           "Clothing")


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True), # 시간 단축을 위해 512, 512로 이미지 축소
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512), # train과 test의 size를 동일하게 셋팅
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2, # 이거 이상한데...
    #workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        img_prefix=root,
        ann_file=root+'train_2.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes = classes,
        img_prefix=root,
        ann_file=root+'valid_2.json',
        pipeline=test_pipeline), # pipeline 은 test와 같이
    test=dict(
        type=dataset_type,
        classes = classes,
        img_prefix=root,
        ann_file=root+'test.json',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
