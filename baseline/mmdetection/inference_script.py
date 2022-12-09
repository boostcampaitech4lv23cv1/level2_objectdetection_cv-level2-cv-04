import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import argparse

def main():
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # config file 들고오기
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config0_baseline.py") # config file name
    parser.add_argument('--work_dir', type=str, default= None)
    parser.add_argument('--weight_file', type=str, default = 'latest.pth')
    parser.add_argument('--infer_threshold', type=float, default=0.05)

    args = parser.parse_args()

    cfg = Config.fromfile("/opt/bro/baseline/mmdetection/configs/_khh_custom_configs/" + args.config)

    root = "/opt/bro/dataset/"

    # dataset config 수정
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = 4

    cfg.seed=2021
    cfg.gpu_ids = [1]
    cfg.work_dir = "/opt/bro/workspace/work_dirs/" + args.work_dir

    cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    # build dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{args.weight_file}')

    # model build
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
    
    # attach model pretrain weight
    load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

    model = MMDataParallel(model.cuda(), device_ids=[0])

    # inference
    output = single_gpu_test(model, data_loader, show_score_thr=args.infer_threshold) # output 계산

    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []

    # load annot file
    coco = COCO(cfg.data.test.ann_file)

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f'for_submission.csv'), index=None)

main()