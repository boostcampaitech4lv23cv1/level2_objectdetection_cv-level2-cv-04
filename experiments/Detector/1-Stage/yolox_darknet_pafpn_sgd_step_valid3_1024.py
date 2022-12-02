optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=15,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MMDetWandbHook',
            init_kwargs=dict(
                project='1stage',
                entity='boostcamp_aitech4_jdp',
                name='yolox_x_8x8_300e_coco'),
            interval=10,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=100)
    ])
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(type='SyncNormHook', num_last_epochs=15, interval=10, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=64)
img_scale = (1024, 1024)
model = dict(
    type='YOLOX',
    input_size=(1024, 1024),
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='TOODHead',
        num_classes=10,
        in_channels=256,
        feat_channels=320,
        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(
            score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file=
            '/opt/ml/level2_objectdetection_cv-level2-cv-04/folded_anns/train_3.json',
            img_prefix=
            '/opt/ml/level2_objectdetection_cv-level2-cv-04/dataset/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False,
            classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                     'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                     'Clothing')),
        pipeline=[
            dict(type='Mosaic', img_scale=(1024, 1024), pad_val=114.0),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-320, -320)),
            dict(
                type='MixUp',
                img_scale=(1024, 1024),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/opt/ml/level2_objectdetection_cv-level2-cv-04/folded_anns/val_3.json',
        img_prefix='/opt/ml/level2_objectdetection_cv-level2-cv-04/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing')),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/opt/ml/level2_objectdetection_cv-level2-cv-04/dataset/test.json',
        img_prefix='/opt/ml/level2_objectdetection_cv-level2-cv-04/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing')))
max_epochs = 300
num_last_epochs = 15
interval = 10
evaluation = dict(
    save_best='auto', interval=1, dynamic_intervals=[(285, 1)], metric='bbox')
seed = 42
gpu_ids = [0]
work_dir = './workspace/work_dirs/yolox_x_8x8_300e_coco'
device = 'cuda'
