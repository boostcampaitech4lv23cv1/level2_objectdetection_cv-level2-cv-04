_base_ = [
    "/opt/ml/level2_objectdetection_cv-level2-cv-04/baseline/mmdetection/configs/_chj_custom_configs/models/faster_rcnn_r50_fpn.py"
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'  # noqa

# __all__ = [
#     'FPN', 'BFP', 'ChannelMapper', 'HRFPN', 'NASFPN', 'FPN_CARAFE', 'PAFPN',
#     'NASFCOS_FPN', 'RFP', 'YOLOV3Neck', 'FPG', 'DilatedEncoder',
#     'CTResNetNeck', 'SSDNeck', 'YOLOXPAFPN', 'DyHead'
# ]


model = dict(

    type='FasterRCNN', # 디텍터 타입

    backbone=dict(
        _delete_=True, # 기존 Faster RCNN의 default는 resnet, 따라서 이전 hyper param은 모두 삭제한다
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
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

    # neck=dict(in_channels=[96, 192, 384, 768]), # neck 체널 변경
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536]))