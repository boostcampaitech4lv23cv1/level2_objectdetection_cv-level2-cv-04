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
import wandb
from datetime import datetime
from pytz import timezone
import random, os
import numpy as np
import torch

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    boolean_parse = lambda x: True if x in (None, 'True', 'true', 'T', 't', '1') else False
    now = datetime.now(timezone('Asia/Seoul'))
    parser = argparse.ArgumentParser(description='basic Argparse')
    parser.add_argument('--seed', type=int, default=42, help='시드')
    parser.add_argument('--annot', type=str, default='3', help='사용할 fold num (0~4)')
    parser.add_argument('--config_path', type=str, help='config 파일 경로')  #
    parser.add_argument('--resize', type=int, default=512, help='이미지 resize 크기')  
    parser.add_argument('--epochs', type=int, default=10, help='epochs')  
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')  
    parser.add_argument('--grad_clip', type=boolean_parse, default=True, help='gradient clip 사용 여부')  
    parser.add_argument('--val', type=boolean_parse, default=True, help='val 수행 여부')  
    parser.add_argument('--root_dir', type=str, default="/opt/ml/dataset/", help='dataset 폴더 경로')  
    parser.add_argument('--ann_dir', type=str, default="/opt/ml/folded_anns/", help='train/val json 파일이 존재하는 위치')  
    parser.add_argument('--work_dir', type=str, default='my_experiment', help='model.pth, log 등이 저장될 폴더 이름')  
    parser.add_argument('--wdb_project', type=str, help='wandb 프로젝트 이름')  
    parser.add_argument('--wdb_exp', type=str, default=f'{now.year}-{now.month}-{now.day}, {now.hour}:{now.minute}')  #wandb 실험 이름
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_path)

    cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='MMDetWandbHook',
         init_kwargs={"project": args.wdb_project, # 저장할 프로젝트이름
                      "entity" : "boostcamp_aitech4_jdp", # 현재 팀 공통으로 쓰고있는 entity
                      "name": f"{args.wdb_exp}"}, # 실험 이름
         interval=10,
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=100)]

    root = args.root_dir
    
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = os.path.join(args.ann_dir, ("train_" + args.annot + ".json"))
    cfg.data.train.pipeline[2]['img_scale'] = (args.resize,args.resize)

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = os.path.join(args.ann_dir, ("val_" + args.annot + ".json"))
    cfg.data.val.pipeline[1]['img_scale'] = (args.resize,args.resize)

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = os.path.join(root, 'test.json')
    cfg.data.test.pipeline[1]['img_scale'] = (args.resize,args.resize)
    
    cfg.data.samples_per_gpu = args.batch_size # 배치 사이즈를 의미
    
    cfg.seed = args.seed # config에 시드 고정
    seed_everything(args.seed) # 재생산을 위해서 모든 시드를 고정

    cfg.gpu_ids = [0]
    cfg.work_dir = './workspace/work_dirs/' + args.work_dir

    cfg.model.roi_head.bbox_head.num_classes = 10 # 모델의 헤드 개수변경
    
    if args.grad_clip:
        cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2) # gradient clipping(트레이닝 기법)
    else:
        cfg.optimizer_config.grad_clip = None
            
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1) # 저장주기를 1로 하되 가장 높은 3개만 남긴다
    cfg.device = get_device()

    datasets = [build_dataset(cfg.data.train)]

    print(datasets[0])

    model = build_detector(cfg.model)
    model.init_weights()

    # ⭐️ Set the evaluation interval.
    cfg.evaluation.interval = 1 # validation 인터벌을 의미함.
    
    # ⭐️ Set meta data
    meta = dict()
    meta['exp_name'] = os.path.basename(args.config_path)
    # ⭐️ Save config file in wandb and local
    os.makedirs(cfg.work_dir, exist_ok=True)
    # 현재 config 파일을 work_dir에 저장함
    cfg.dump(os.path.join(cfg.work_dir, meta['exp_name'])) 

    # 실험의 epoch 결정
    cfg.runner.max_epochs = args.epochs # epoch argparser로 받아 오버라이팅

    # validate=True, wandb에 valid 기록 저장됨.
    # meta=meta, wandb에 실험의 전체적인 config 저장됨.
    train_detector(model, datasets[0], cfg, distributed=False, validate=args.val, meta=meta)
    

main()