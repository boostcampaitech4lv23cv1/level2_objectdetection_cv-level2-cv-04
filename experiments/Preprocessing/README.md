# Preprocessing

## Basic Augmentation

#### 가설

모델 실험 중 모든 모델에서 아래 표와 같이 모든 모델에서 overfitting 현상을 확인할 수 있었음.
(충분한 epoch로 train loss가 안정된 상태임)

우리는 모든 모델에서 공통적으로 나타나는 overfitting을 해결하기 위해 
유효한 augmentation을 찾는 실험을 진행 하기로 하기로 하였음.

![[표] 서로다른 모델에서 공통적으로 나타나는 overfitting 현상](./images/Untitled.png)

[표] 서로다른 모델에서 공통적으로 나타나는 overfitting 현상

우리는 많은 augmentation 중에 data와 가장 잘 맞는 augmentation을 빠르게 찾기 위해, 
원본 image을 크게 변경시키지 않는 soft augmentation을 만들고 크게 3가지 계열로 묶어 실험을 진행하였음. 

유형은 아래와 같음.
aug1 : 색상을 변경시키는 augmentation
aug2 : 경계값을 흐리게 만드는 augmentation
aug3 : bounding box의 geometry 정보를 변경시키는 augmentation

① aug1 : 색상 관련 augmentation

`RandomBrightnessContrast`, `RGBShift`, `HueSaturationValue`, `ChannelShuffle`

![[이미지] 색상관련 augmentation(aug1) 적용 전/후](./images/Untitled%201.png)

[이미지] 색상관련 augmentation(aug1) 적용 전/후

② aug2 : 경계값을 흐리게하는 노이즈와 블러 관련 augmentation

`GaussNoise`, `RandomBrightnessContrast`, `Blur`, `MedianBlur`

![[이미지] 경계값과 관련된 augmentation(aug2) 적용 전/후](./images/Untitled%202.png)

[이미지] 경계값과 관련된 augmentation(aug2) 적용 전/후

③ aug3 : bbox의 이동 회전 관련

`RandomRotate90`, `HorizontalFlip`, `ShiftScaleRotate`

![[이미지] bbox의 geometry 변경과 관련된 augmentation(aug3) 적용 전/후](./images/Untitled%203.png)

[이미지] bbox의 geometry 변경과 관련된 augmentation(aug3) 적용 전/후

#### 실험 결과

![[표] baseline과 aug1, aug2, aug3의 train, validation score 성능](./images/Untitled%204.png)

[표] baseline과 aug1, aug2, aug3의 train, validation score 성능

![[이미지] baseline과 aug1, aug2, aug3의 validation score 그래프(x축 epoch)](./images/Untitled%205.png)

[이미지] baseline과 aug1, aug2, aug3의 validation score 그래프(x축 epoch)

- 동일한 epoch에서 train mAP는 크게 감소하였으며, 
valid mAP는 유의미한 차이(mAP 0.02 이상)를 보이지 않았음.

#### 결과 분석

당시, validation에 크게 유효한 성능을 보이는 않는 것으로 확인되어 추가 실험을 진행 하지 않았음.

하지만, 대회가 끝난 후 실험 결과를 다시 살펴 보니 baseline과 동일한 epoch으로 실험을 진행하였기 때문에
augmentation된 데이터를 충분히 학습하지 못했을 수도 있었다는 생각이 듬(train mAP가 굉장히 낮음).
augmentation을 고려하여 더 긴 epoch으로 충분히 학습한 이후에 성능을 비교했으면 지금과 다른 결론이 나왔을수도 있다고 생각됨.
* * *
## Advanced Augmentation

### Mosaic augmentation

#### 가설

주어진 데이터셋의 이미지들은 다양한 background의 환경에서 수집된 이미지들이었음.
이러한 특징 때문에, 모델이 background와 object를 잘 구분하지 못한다고 가정하였음.

또 주어지는 이미지가 5,000장 이하로 데이터의 개수가 매우 부족하다는 생각이 들었음.

우리는 이러한 단점을 mosaic augmentation의 특징(여러가지 다양한 이미지를 합성하여 학습에 이용)
이 다양한 background를 경험하게 하고, 다양한 합성 이미지를 학습에 활용하므로, 
데이터 부족의 문제를 일부 해줄 것이라는 가설을 세웠음.

![[이미지] 서로다른 background를 가지고 있는 이미지들](./images/Untitled%206.png)

[이미지] 서로다른 background를 가지고 있는 이미지들

#### 실험 결과

![[표] mosaic augmentation 전후의 model의 성능차이](./images/Untitled%207.png)

[표] mosaic augmentation 전후의 model의 성능차이

![[이미지] baseline과 mosaic augmentation이 적용된 모델의 RPN classification loss](./images/Untitled%208.png)

[이미지] baseline과 mosaic augmentation이 적용된 모델의 RPN classification loss

![[이미지] bounding box 사이즈별 baseline과 mosaic augmentation의 성능차이 (x축 epochs)](./images/Untitled%209.png)

[이미지] bounding box 사이즈별 baseline과 mosaic augmentation의 성능차이 (x축 epochs)

![[이미지] yolov7-e6e 모델에서 mosaic 파라미터 0.3 vs 1.0 성능 차이](./images/Untitled%2010.png)

[이미지] yolov7-e6e 모델에서 mosaic 파라미터 0.3 vs 1.0 성능 차이

![Untitled](./images/Untitled%2011.png)

- 동일한 epoch에서는 mosaic augmentation을 적용한 model의 train mAP가 감소하였음.
- 동일한 epoch에서는 mosaic augmentation을 적용한 model의 train, validation 성능의 gap이 적었음.
(generalization의 성능이 향상됨)
- middle, large size bbox에서의 성능이 mosaic augmentation에서 상승하였음.
- rpn classification loss는 baseline과 mosaic간 차이가 나지 않았음.
- yolov7에서는 mosaic augmentation 을 쎄게 줄 수록 학습 속도가 더 빨랐고 제출 성능도 더 좋았다. 
아직 학습이 더 될 기미가 보여 10 epoch 정도를 더 늘려서 다시 실험해보고 싶었지만, 
시간이 부족한 관계로 성능 개선이 확실하다고 판단한 pseudo 실험으로 넘어갔다.

#### 결과 분석

Mosaic augmentation을 적용 후 모델의 성능은 아래와 같이 큰폭으로 증가하였다.

가설과는 다르게 rpn의 classification loss는 크게 감소하지 않았기 때문에(baseline과 mosaic aug의 rpn cls loss 차이 이미지)
background를 잘 구분하는 효과는 없었다고 생각됨.

하지만, mosaic이 된 데이터를 학습에 이용한 model의 경우 bounding box의 사이즈별로 성능의 차이가 큰것을 확인할 수 있었으며,
이는 mosaic augmentation의 경우 다양한 bounding box의 사이즈를 예측하는데 영향을 미친것으로 보임.

하지만, 실험과 같이 조금 더 긴 epoch에서의 결과를 비교한다면 조금 더 정확한 비교를 할 수 있지 않았을까 생각됨.
또한 위 가설을 결론을 검증하기 위해 RandomResizedCrop을 이용해 추가 실험을 진행하면 더 정확한 결론을 얻을 수 있을것이라 생각됨.
* * *
### Mixup Augmentation

#### 가설

앞서 mosaic augmentation의 유효한 효과를 확인 후, bbox의 사이즈를 자유롭게 조정하는 형태의 augmentation,
합성(synthesized) 이미지 유형의 augmentation이 효과가 있을것이라는 가설을 세웠음.

![[이미지] mixup augmentation이 적용된 이미지들](./images/Untitled%2012.png)

[이미지] mixup augmentation이 적용된 이미지들

#### 실험 결과

![[표] mixup 적용 전후의 성능 차이](./images/Untitled%2013.png)

[표] mixup 적용 전후의 성능 차이

![[이미지] baseline, mixup, mosaic이 적용된 모델들의 validation score](./images/Untitled%2014.png)

[이미지] baseline, mixup, mosaic이 적용된 모델들의 validation score

- mixup augmentation이 적용된 후 model의 train mAP와 validation mAP는 하락하였음.

#### 결과 분석

mixup 적용 후 train과 validation mAP의 성능은 크게 하락하였음.
조금더 epoch가 긴 상태에서 분석을 진행하였으면 더 정확한 비교가 되었을것이라는 아쉬움이 있었으며,

transparent한 이미지가 test set에 존재하지 않아 악영향을 미쳤을 수도 있다는 생각이 듬.
추가적으로 찾아본 Cutmix 논문([https://arxiv.org/pdf/1905.04899.pdf](https://arxiv.org/pdf/1905.04899.pdf))에서도 우리의 실험과 유사하게
mixup을 적용할 경우 성능이 소폭 하락하는 것을 확인할 수 있었음.

![Untitled](./images/Untitled%2015.png)