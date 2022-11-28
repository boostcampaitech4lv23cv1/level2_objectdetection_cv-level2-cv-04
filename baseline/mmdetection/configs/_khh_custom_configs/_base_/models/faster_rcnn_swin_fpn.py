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
    )
