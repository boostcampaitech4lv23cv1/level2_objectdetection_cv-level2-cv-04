_base_ = [
    "/opt/ml/level2_objectdetection_cv-level2-cv-04/baseline/mmdetection/configs/_chj_custom_configs/models/faster_rcnn_r50_fpn.py"
]

# __all__ = [
#     'FPN', 'BFP', 'ChannelMapper', 'HRFPN', 'NASFPN', 'FPN_CARAFE', 'PAFPN',
#     'NASFCOS_FPN', 'RFP', 'YOLOV3Neck', 'FPG', 'DilatedEncoder',
#     'CTResNetNeck', 'SSDNeck', 'YOLOXPAFPN', 'DyHead'
# ]


model = dict(

    type='FasterRCNN', # 디텍터 타입

    # neck=dict(in_channels=[96, 192, 384, 768]), # neck 체널 변경
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048]))
    