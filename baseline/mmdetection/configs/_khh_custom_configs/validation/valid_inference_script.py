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


### validation을 위해 test_config를 이용합니다.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_name', '-cfn',type=str, default="config4_baseline_dy.py") # config file name
    parser.add_argument('--work_dir_name', '-wdn', type=str, default=None)
    parser.add_argument('--weight_file_name', '-wfn', type=str, default = 'latest.pth')
    parser.add_argument('--annot_name', '-an', type=str, default='val_fold0.json')
    args = parser.parse_args()

    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # config file 들고오기
    file_path = '/opt/level2_objectdetection_cv-level2-cv-04/baseline/mmdetection/configs/_khh_custom_configs/'+ args.config_file_name
    cfg = Config.fromfile(file_path)

    # dataset root
    root='/opt/level2_objectdetection_cv-level2-cv-04/dataset/'

    # modify valid dataset config
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + args.annot_name
    cfg.data.test.test_mode = True

    # valid batch size
    cfg.data.samples_per_gpu = 4

    # set default
    cfg.seed=42
    cfg.gpu_ids = [1]

    # set default path
    cfg.work_dir = '/opt/level2_objectdetection_cv-level2-cv-04/baseline/mmdetection/work_dirs/' + args.work_dir_name

    # modify model
    # cfg.model.roi_head.bbox_head.num_classes = 10 # already done

    # modify model test config
    cfg.model.test_cfg.nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)


    # build dataset
    dataset = build_dataset(cfg.data.test) # valid data set
    
    # bulid dataloader
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    # load latest.path file
    checkpoint_path = os.path.join(cfg.work_dir, args.weight_file_name)

    # build detector
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) 
    
    # attach pretrain weight
    load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

    # pretrain detector load
    model = MMDataParallel(model.cuda(), device_ids=[0])

    # predict
    output = single_gpu_test(model, data_loader, show_score_thr=0.00) # output 계산

    # submission 양식에 맞게 output 후처리
    file_names = []
    prediction_strings = []
    
    coco = COCO(cfg.data.test.ann_file)

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                #  (label, score, xmin, ymin, xmax, ymax)
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
            
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    # inference weight 이름으로 infernce weight가 저장된 폴더에 저장함.
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_valid_{args.weight_file_name.split(".")[0]}.csv'), index=None)
    print(submission.head())

main()