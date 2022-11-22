from map_boxes import mean_average_precision_for_boxes
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO

def main():
    ## modify argparser down ↓
    GT_FULL_JSON = "/opt/level2_objectdetection_cv-level2-cv-04/dataset/train.json"
    TRAIN_JSON = "/opt/level2_objectdetection_cv-level2-cv-04/dataset/train_fold3.json"
    GT__VALID_JSON = "/opt/level2_objectdetection_cv-level2-cv-04/dataset/val_fold3.json"
    PRED_CSV = "/opt/level2_objectdetection_cv-level2-cv-04/baseline/mmdetection/work_dirs/config4_baseline_dy__fold3/submission_valid_latest.csv"

    LABEL_NAME = ["General trash", "Paper", "Paper pack", "Metal", 
                  "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

    # load prediction
    full_coco = COCO(GT_FULL_JSON)
    train_coco = COCO(TRAIN_JSON)
    valid_coco = COCO(GT__VALID_JSON)
    pred_df = pd.read_csv(PRED_CSV)

    print("len(full_coco.getImgIds()):", len(full_coco.getImgIds()))
    print("len(train_coco.getImgIds()):", len(train_coco.getImgIds()))
    print("len(valid_coco.getImgIds()):", len(valid_coco.getImgIds()))
    print("len(pred_df['image_id'].unique()):", len(pred_df['image_id'].unique()) ) # 예측한 것의 고유 이미지 개수



    # file_names = pred_df['image_id'].values.tolist()
    # bboxes = pred_df['PredictionString'].values.tolist()
    # new_pred = []
    # gt = []

    # # pred format change
    # for i, bbox in enumerate(bboxes):
    #     if isinstance(bbox, float):
    #         print(f'{file_names[i]} empty box')

    # for file_name, bbox in tqdm(zip(file_names, bboxes)):
    #     boxes = np.array(str(bbox).split(' '))
    #     if len(boxes) % 6 == 1:
    #         boxes = boxes[:-1].reshape(-1, 6)
    #     elif len(boxes) % 6 == 0:
    #         boxes = boxes.reshape(-1, 6)
    #     else:
    #         raise Exception('error', 'invalid box count')
    #     for box in boxes:
    #         new_pred.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
    
    # # gt format change
    # coco = COCO(GT_JSON)
    # for image_id in coco.getImgIds():
    #     image_info = coco.loadImgs(image_id)[0]
    #     annotation_id = coco.getAnnIds(imgIds=image_info['id'])
    #     annotation_info_list = coco.loadAnns(annotation_id)
    #     file_name = image_info['file_name']
    #     for annotation in annotation_info_list:
    #         gt.append([file_name, annotation['category_id'],
    #                 float(annotation['bbox'][0]),
    #                 float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
    #                 float(annotation['bbox'][1]),
    #                 (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])
    # print(len(gt))
    # # mean_ap, average_precisions = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=0.5)

    # # print(mean_ap)

main()