import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from map_boxes import mean_average_precision_for_boxes
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import argparse
import json
import tqdm


### validation을 위해 test_config를 이용합니다.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_name', '-cfn',type=str, default="config4_baseline_dy.py") # config file name
    parser.add_argument('--work_dir_name', '-wdn', type=str, default=None)
    parser.add_argument('--weight_file_name', '-wfn', type=str, default = 'latest.pth')
    parser.add_argument('--annot_file_name', '-afn', type=str, default='val_0.json')
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
    cfg.data.test.ann_file = "/opt/level2_objectdetection_cv-level2-cv-04/folded_anns/" + args.annot_file_name
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
    print(dataset) # print dataset info

    
    # bulid dataloader by dataset
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


    # inference
    output = single_gpu_test(model, data_loader, show_score_thr=0.00)
    
    # valid annot load
    file_names = []
    prediction_strings = []
    coco = COCO(cfg.data.test.ann_file) # don't confuse naming(test) "cfg.data.test.ann_file" is validation root

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=coco.getImgIds()[i]))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
            
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    submission.to_csv(os.path.join(cfg.work_dir,"for_validation.csv"), index=False)

    
    ### ↓↓↓↓↓ get mAP

    LABEL_NAME = ["General trash", "Paper", "Paper pack", "Metal", 
                  "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

    # load prediction
    pred_df = pd.read_csv(os.path.join(cfg.work_dir,"for_validation.csv"))

    ## not use check
    # print("len(full_coco.getImgIds()):", len(coco.getImgIds()))
    # print("len(pred_df['image_id'].unique()):", len(pred_df['image_id'].unique()) )

    file_names = pred_df['image_id'].values.tolist()
    bboxes = pred_df['PredictionString'].values.tolist()
    new_pred = []
    gt = []

    # pred format change
    for i, bbox in enumerate(bboxes):
        if isinstance(bbox, float):
            print(f'{file_names[i]} empty box')

    for file_name, bbox in zip(file_names, bboxes):
        boxes = np.array(str(bbox).split(' '))
        if len(boxes) % 6 == 1:
            boxes = boxes[:-1].reshape(-1, 6)
        elif len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        else:
            raise Exception('error', 'invalid box count')
        for box in boxes:
            new_pred.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
    
    # gt format change
    for image_id in coco.getImgIds():
        image_info = coco.loadImgs(image_id)[0]
        annotation_id = coco.getAnnIds(imgIds=image_info['id'])
        annotation_info_list = coco.loadAnns(annotation_id)
        file_name = image_info['file_name']
        for annotation in annotation_info_list:
            gt.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])

    mean_ap, average_precisions = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=0.5)

    print(mean_ap)



main()