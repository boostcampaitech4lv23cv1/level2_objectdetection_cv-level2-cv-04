# Pseudo Labeling

#### 가설

우리는 단일 모델로 underfitting을 개선하기 위해서는 dataset을 추가적으로 확보해야 할 필요성을 느낌.
pseudo labeling을 통해 train 데이터셋의 숫자를 4,882개에서 9,753개로 증가시킨다면(이미지 기준)
현재 겪고 있는 underfitting이 개선되어 더 높은 test score가 가능할 것이라는 가설을 세움.

우리는 앙상블하여 test mAP 0.6473를 달성한 test submission을 이용하여 pseudo-labeling을 제작하였음.

submission.csv의 object detection 결과 중 confidence score를  0.5 threshold로 필터링하여 사용하였음.
(confidence 0.5 수준으로 필터링했을 경우가 육안으로 가장 acceptable한 최소값이었음.)

필터링한 결과 만들어진 pseudo label의 시각화 결과는 다음과 같음.

![[이미지] 쉐도 라벨링된 test dataset의 라벨링 시각화 결과](./images/Untitled.png)

[이미지] 쉐도 라벨링된 test dataset의 라벨링 시각화 결과

#### 실험 결과

pseudo label dataset이 추가된 실험결과는 다음과 같음.
(pseudo label을 추가한 실험의 경우 validation score가 없어 test score만 사용함.
또한  대조군과 다르게 이미지 사이즈를 800,800으로 실험 진행함)

![[표] trainset과 pseudo label set을 training한 모델의 성능차이](./images/Untitled%201.png)

[표] trainset과 pseudo label set을 training한 모델의 성능차이

![Untitled](./images/Untitled%202.png)

![Untitled](./images/Untitled%203.png)

| Model | Pseudo labeling | Test mAP |
| --- | --- | --- |
| YOLOv7-E6E | ⭕️ | 0.6480 |
| YOLOv7-E6E | ❌ | 0.6188 |

#### 결과 분석

pseudo label 적용 후 대조군 비교하여 - 0.037의 성능감소를 보임.

대회 마지막에 적용한 실험으로 시간부족으로 통제 변인을 지키지 않았고 
따라서 정량적으로 결과 분석을 진행할 수 없었음.

반면 yolov7-e6e 모델에서 pseudo label 적용하여 더 큰 train dataset 을 사용한 결과 train loss가 더 빨리 줄어드는 것을 확인할 수 있었다.

pseudo label 을 적용한 data를 추가한 train dataset을 사용해 YOLOv7-E6E 모델을 학습시킨 결과 Test mAP 성능이 0.03이나 증가했다.

pseudo label을 만들때 confidence threshold를 더욱 올려서 실험했으면  어떤 결과가 나왔을까?
또 pseudo labeling을 적용한 모델은 inference시 test config의 IoU threshold 를 더 크게 높여 사용했으면
어떤 결과가 나올지 추가 실험이 필요했으나 시간이 부족하여 실험하지 못하여 아쉬움이 따른다.