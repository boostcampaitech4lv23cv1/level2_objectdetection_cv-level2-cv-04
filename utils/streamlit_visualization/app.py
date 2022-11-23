import streamlit as st

import os
import json

import pandas as pd
import cv2 as cv
import numpy as np
from copy import deepcopy

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


###
submission_file_dir = '/opt/ml/JDP/baseline/mmdetection/workspace/workspace/work_dirs/submission_latest.csv'
gt_json_file_dir = '/opt/ml/dataset/train.json'
dataset_path = '/opt/ml/dataset/'
###



colors = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7", "#7c1158"]
colors = list(map(lambda x : x.lstrip('#'), colors))
colors = list(map(lambda x : tuple(int(x[i:i+2], 16) for i in (0, 2, 4)), colors))

LABEL_NAME = ["General trash", "Paper", "Paper pack", "Metal", 
              "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

def draw_bbox(img, bbox, boxcolor, label, boxsize):
    if boxsize < 100:
        boxsize = 100
    if boxsize > 500:
        boxsize = 500
    
    cv.rectangle(img = img, pt1=bbox[:2], pt2=bbox[2:], color = boxcolor, thickness=5)
    if bbox[2] > 980:
        bbox[2] = 980
    if bbox[3] < 40:
        bbox[3] = 40
    
    cv.putText(img=img, text=label, org=(bbox[0], bbox[1]-10), fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=boxsize/250, color=(255,255,255), thickness=boxsize//150)
    return img

def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def main():
    st.title("Compare inferenced file with ground truth")
    
    
    data = pd.read_csv(submission_file_dir, index_col=False)

    with open(gt_json_file_dir) as f:
        gt_data = json.load(f)
        
    
    #image index 설정
    image_index = int(st.number_input('보고싶은 이미지의 인덱스를 입력해주세요', value=0))
    target_image_id = gt_data['images'][image_index]['id']
    
    target_data = data.iloc[image_index, 0]
    preds = target_data.split(' ')[:-1]
    
    preds = np.array(list(map(float, preds)))
    preds = np.reshape(preds, (-1, 6))
    order = np.argsort(preds[:, 1])[::-1]
    preds = preds[order]
    
    #confidence threshold 설정
    conf_threshold = st.slider('다음 값 이상의 Confidence score 를 가지는 박스만 표시:', 0.0, 100.0, 50.0)
    th_preds = preds[preds[:, 1] >= conf_threshold/100.0]
    # print(th_preds)
    
    img_path = os.path.join(dataset_path, gt_data['images'][image_index]['file_name'])
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    infered_img = deepcopy(img)
    print(target_image_id)
    ##gt 이미지
    gt_bboxes = []
    for anns in gt_data['annotations']:
        if anns['image_id'] == target_image_id:
            boxsize = int(anns['bbox'][2])
            label = LABEL_NAME[anns['category_id']]
            bbox = list(map(int, [anns['bbox'][0], anns['bbox'][1], anns['bbox'][0]+anns['bbox'][2], anns['bbox'][1]+anns['bbox'][3]]))
            gt_bboxes.append(bbox)
            boxcolor = colors[anns['category_id']]
            img = draw_bbox(img, bbox, boxcolor, label, boxsize)
            
    ##infered 이미지
    for pred in th_preds:
        correct = False
        bbox = list(map(int, pred[2:].tolist()))
        for gt_bbox in gt_bboxes:
            if IoU(gt_bbox, bbox) > 0.5:
                correct = True
                break
        
        if correct:
            label = 'TP'
            boxcolor = (0,255,0)
        else:
            label = 'FP'
            boxcolor = (255,0,0)
        boxsize = int(bbox[2] - bbox[0])
        infered_img = draw_bbox(infered_img, bbox, boxcolor, label, boxsize)
        


    col1, col2 = st.columns(2)
    col1.image(img)
    col2.image(infered_img)
    


main()