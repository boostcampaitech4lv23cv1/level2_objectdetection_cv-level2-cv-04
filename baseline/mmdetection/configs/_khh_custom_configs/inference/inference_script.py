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
    parser.add_argument('--config_file_name', '-cfn',type=str, default="config4_baseline_dy.py") # config file name
    parser.add_argument('--work_dir_name', '-wdn', type=str, default=None)
    parser.add_argument('--weight_file_name', '-wfn', type=str, default = 'latest.pth')

    args = parser.parse_args()

    cfg = Config.fromfile("/opt/level2_objectdetection_cv-level2-cv-04/baseline/mmdetection/configs/_khh_custom_configs/"+ \
        args.config_file_name)

    root='/opt/level2_objectdetection_cv-level2-cv-04/dataset/'

    # dataset config 수정
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'
    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = 4

    cfg.seed=2021
    cfg.gpu_ids = [1]
    cfg.work_dir = "/opt/level2_objectdetection_cv-level2-cv-04/baseline/mmdetection/work_dirs/" + \
                    args.work_dir_name

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
    checkpoint_path = os.path.join(cfg.work_dir, f'{args.weight_file_name}')

    # model build
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
    
    # attach model pretrain weight
    load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

    model = MMDataParallel(model.cuda(), device_ids=[0])

    # inference
    output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산

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