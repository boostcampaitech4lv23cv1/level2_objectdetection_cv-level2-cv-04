_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    './datasets/_trash_detection.py', # 수정해야함
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        arch='b3',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),  # out_indices (Sequence[int]): Output from which stages.
        frozen_stages=0, # faster_rcnn_r50_fpn.py 에서는 1인데 뭐가 맞을까? 일단, layer 0만 얼리자.
        # frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
        # -1 means not freezing any parameters.
        # norm_cfg=dict(
        #     type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False, # faster_rcnn_r50_fpn.py 에서는 True 임
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)
    ),
    neck=dict(
        type='FPN',
        in_channels=[48, 136, 384],
        out_channels=256,
        num_outs=5
    ),
    
    # rpn_head 문제 없고
    
    # roi_head.bbox_roi_extractor.featmap_strides 보자...
    roi_head=dict(
        type='StandardRoIHead', # 이거 구데가 아님?
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            # featmap_strides=[4, 8, 16, 32]), # 이거 고쳐줘야 할 것 같은데
            featmap_strides=[4, 8, 16] # feature map 3개니까 일단 3개로 하자.
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10, # class 개수 수정
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    
    # training and testing settings
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)))


# optimizer
# optimizer_config = dict(grad_clip=None)
# optimizer = dict(
#     type='SGD',
#     lr=0.04,
#     momentum=0.9,
#     weight_decay=0.0001,
#     paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True))
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.1,
#     step=[8, 11])
# # runtime settings
# runner = dict(type='EpochBasedRunner', max_epochs=12)

# # NOTE: `auto_scale_lr` is for automatically scaling LR,
# # USER SHOULD NOT CHANGE ITS VALUES.
# # base_batch_size = (8 GPUs) x (4 samples per GPU)
# auto_scale_lr = dict(base_batch_size=32)