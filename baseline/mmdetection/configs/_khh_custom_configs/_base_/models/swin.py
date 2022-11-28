_base_ = [
    "./faster_rcnn_r50_fpn.py"
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(

    type='FasterRCNN', # 디텍터 타입

    backbone=dict(
        _delete_=True, # 기존 Faster RCNN의 default는 resnet, 따라서 이전 hyper param은 모두 삭제한다
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
        out_indices=(0, 1, 2, 3), # stage index
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),

    neck=dict(in_channels=[96, 192, 384, 768]), # neck 체널 변경
    
    # rpn_head=dict(
    #     type='RPNHead',
    #     in_channels=768, # 256 to 768 변경, neck으로 들어오는 highlevel의 featuremap channel 개수
    #     feat_channels=768, # 256 to 768, rpn 임배딩 layer의 output channel 수
    #     anchor_generator=dict(
    #         type='AnchorGenerator',
    #         scales=[8],
    #         ratios=[0.5, 1.0, 2.0],
    #         strides=[4, 8, 16, 32, 64]),
    #     bbox_coder=dict(
    #         type='DeltaXYWHBBoxCoder',
    #         target_means=[.0, .0, .0, .0],
    #         target_stds=[1.0, 1.0, 1.0, 1.0]),
    #     loss_cls=dict(
    #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    #     loss_bbox=dict(type='L1Loss', loss_weight=1.0)),


    # roi_head=dict(
    #     type='StandardRoIHead',
        
    #     bbox_roi_extractor=dict(
    #         type='SingleRoIExtractor',
    #         roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
    #         out_channels=768, # 뜯어오는 layer 근데 input의 channel의 수와 동일하지 않나?
    #         featmap_strides=[4, 8, 16, 32]),
        
    #     bbox_head=dict(
    #         type='Shared2FCBBoxHead',
    #         in_channels=768, # 256 to 768
    #         fc_out_channels=1024, # fc를 통과한 이후의 channel 수
    #         roi_feat_size=7,
    #         num_classes=10, # 데이터셋에 맞춰서 수정
    #         bbox_coder=dict(
    #             type='DeltaXYWHBBoxCoder',
    #             target_means=[0., 0., 0., 0.],
    #             target_stds=[0.1, 0.1, 0.2, 0.2]),
    #         reg_class_agnostic=False,
    #         loss_cls=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #         loss_bbox=dict(type='L1Loss', loss_weight=1.0))),


    # # model training and testing settings
    # train_cfg=dict(
    #     rpn=dict(
    #         assigner=dict(
    #             type='MaxIoUAssigner',
    #             pos_iou_thr=0.7,
    #             neg_iou_thr=0.3,
    #             min_pos_iou=0.3,
    #             match_low_quality=True,
    #             ignore_iof_thr=-1),
    #         sampler=dict(
    #             type='RandomSampler',
    #             num=256,
    #             pos_fraction=0.5,
    #             neg_pos_ub=-1,
    #             add_gt_as_proposals=False),
    #         allowed_border=-1,
    #         pos_weight=-1,
    #         debug=False),
    #     rpn_proposal=dict(
    #         nms_pre=2000,
    #         max_per_img=1000,
    #         nms=dict(type='nms', iou_threshold=0.7),
    #         min_bbox_size=0),
    #     rcnn=dict(
    #         assigner=dict(
    #             type='MaxIoUAssigner',
    #             pos_iou_thr=0.5,
    #             neg_iou_thr=0.5,
    #             min_pos_iou=0.5,
    #             match_low_quality=False,
    #             ignore_iof_thr=-1),
    #         sampler=dict(
    #             type='RandomSampler',
    #             num=512,
    #             pos_fraction=0.25,
    #             neg_pos_ub=-1,
    #             add_gt_as_proposals=True),
    #         pos_weight=-1,
    #         debug=False)),


    # test_cfg=dict(
    #     rpn=dict(
    #         nms_pre=1000,
    #         max_per_img=1000,
    #         nms=dict(type='nms', iou_threshold=0.7),
    #         min_bbox_size=0),
    #     rcnn=dict(
    #         score_thr=0.05,
    #         nms=dict(type='nms', iou_threshold=0.5),
    #         max_per_img=100)
    #         )
    )
