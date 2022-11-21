import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmdet.apis import train_detector
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
from mmdet.utils import get_device
import argparse



## 여기서 config를 불러와서 트레인한다.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_num', '-f', type=str, default='0')
    parser.add_argument('--config_file_name', '-cfn', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()


    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")


    ## abs path files ↓
    file_path = '/opt/level2_objectdetection_cv-level2-cv-04/baseline/mmdetection/configs/_khh_custom_configs/'+ args.config_file_name
    cfg = Config.fromfile(file_path)
    root='/opt/level2_objectdetection_cv-level2-cv-04/dataset/'
    epoch = 'latest'

    

    # modify dataset configs
    cfg.data.train.ann_file = cfg.data.train.img_prefix + 'train_fold'+ f'{args.fold_num}' + '.json'
    cfg.data.val.ann_file = cfg.data.val.img_prefix + 'val_fold' + f'{args.fold_num}' + '.json'
    cfg.data.test.ann_file = cfg.data.test.img_prefix + 'test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (1024,1024) # Resize
    cfg.data.test.test_mode = True
    cfg.data.samples_per_gpu = 4
    
    # modify epochs
    cfg.runner.max_epochs = args.epochs

    # set seed
    cfg.seed=42
    cfg.gpu_ids = [0]

    # root for logging
    work_dir_folder_name = f'{args.config_file_name.split(".")[0]}' + '__' + f'fold{args.fold_num}' # log folder name
    cfg.work_dir = '/opt/level2_objectdetection_cv-level2-cv-04/baseline/mmdetection/work_dirs/'+ \
        work_dir_folder_name

    # modify model config
    cfg.model.test_cfg.nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)

    # modify gradient clip
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

    # setup device
    cfg.device = get_device()

    # model build
    model = build_detector(cfg.model)
    # dataset build
    datasets = [build_dataset(cfg.data.train)]

    ## print your setting
    print("config file name:", args.config_file_name)
    print("save_dir", cfg.work_dir)
    print("<<settings>>")
    print(cfg.pretty_text)

    train_detector(model, datasets[0], cfg, distributed=False, validate=False)

main()