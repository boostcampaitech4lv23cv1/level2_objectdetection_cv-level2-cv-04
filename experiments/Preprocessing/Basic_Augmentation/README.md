#Preprocessing
##Basic Augmentation
####가설

모델 실험 중 모든 모델에서 아래 표와 같이 모든 모델에서 overfitting 현상을 확인할 수 있었음.
(충분한 epoch로 train loss가 안정된 상태임)

우리는 모든 모델에서 공통적으로 나타나는 overfitting을 해결하기 위해 
유효한 augmentation을 찾는 실험을 진행 하기로 하기로 하였음.

![[표] 서로다른 모델에서 공통적으로 나타나는 overfitting 현상](Object%20Detection%20wrap-up%20report%2031b3d22efe7244abb1d8ed08c1ba393d/Untitled%2016.png)

[표] 서로다른 모델에서 공통적으로 나타나는 overfitting 현상

우리는 많은 augmentation 중에 data와 가장 잘 맞는 augmentation을 빠르게 찾기 위해, 
원본 image을 크게 변경시키지 않는 soft augmentation을 만들고 크게 3가지 계열로 묶어 실험을 진행하였음. 

유형은 아래와 같음.
aug1 : 색상을 변경시키는 augmentation
aug2 : 경계값을 흐리게 만드는 augmentation
aug3 : bounding box의 geometry 정보를 변경시키는 augmentation

① aug1 : 색상 관련 augmentation

`RandomBrightnessContrast`, `RGBShift`, `HueSaturationValue`, `ChannelShuffle`

![[이미지] 색상관련 augmentation(aug1) 적용 전/후](images/Untitled%2017.png)

[이미지] 색상관련 augmentation(aug1) 적용 전/후

② aug2 : 경계값을 흐리게하는 노이즈와 블러 관련 augmentation

`GaussNoise`, `RandomBrightnessContrast`, `Blur`, `MedianBlur`

![[이미지] 경계값과 관련된 augmentation(aug2) 적용 전/후](images/Untitled%2018.png)

[이미지] 경계값과 관련된 augmentation(aug2) 적용 전/후

③ aug3 : bbox의 이동 회전 관련

`RandomRotate90`, `HorizontalFlip`, `ShiftScaleRotate`

![[이미지] bbox의 geometry 변경과 관련된 augmentation(aug3) 적용 전/후](images/Untitled%2019.png)

[이미지] bbox의 geometry 변경과 관련된 augmentation(aug3) 적용 전/후

####실험 결과

![[표] baseline과 aug1, aug2, aug3의 train, validation score 성능](images/Untitled%2020.png)

[표] baseline과 aug1, aug2, aug3의 train, validation score 성능

![[이미지] baseline과 aug1, aug2, aug3의 validation score 그래프(x축 epoch)](images/Untitled%2021.png)

[이미지] baseline과 aug1, aug2, aug3의 validation score 그래프(x축 epoch)

- 동일한 epoch에서 train mAP는 크게 감소하였으며, 
valid mAP는 유의미한 차이(mAP 0.02 이상)를 보이지 않았음.
####결과 분석

당시, validation에 크게 유효한 성능을 보이는 않는 것으로 확인되어 추가 실험을 진행 하지 않았음.

하지만, 대회가 끝난 후 실험 결과를 다시 살펴 보니 baseline과 동일한 epoch으로 실험을 진행하였기 때문에
augmentation된 데이터를 충분히 학습하지 못했을 수도 있었다는 생각이 듬(train mAP가 굉장히 낮음).
augmentation을 고려하여 더 긴 epoch으로 충분히 학습한 이후에 성능을 비교했으면 지금과 다른 결론이 나왔을수도 있다고 생각됨.