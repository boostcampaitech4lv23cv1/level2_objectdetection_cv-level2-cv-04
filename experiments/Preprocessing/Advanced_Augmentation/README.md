#Preprocessing
##Advanced Augmentation
###Mosaic Augmentation
####가설

주어진 데이터셋의 이미지들은 다양한 background의 환경에서 수집된 이미지들이었음.
이러한 특징 때문에, 모델이 background와 object를 잘 구분하지 못한다고 가정하였음.

또 주어지는 이미지가 5,000장 이하로 데이터의 개수가 매우 부족하다는 생각이 들었음.

우리는 이러한 단점을 mosaic augmentation의 특징(여러가지 다양한 이미지를 합성하여 학습에 이용)
이 다양한 background를 경험하게 하고, 다양한 합성 이미지를 학습에 활용하므로, 
데이터 부족의 문제를 일부 해줄 것이라는 가설을 세웠음.

![[이미지] 서로다른 background를 가지고 있는 이미지들](images/Untitled%2022.png)

[이미지] 서로다른 background를 가지고 있는 이미지들

####실험 결과

![[표] mosaic augmentation 전후의 model의 성능차이](images/Untitled%2023.png)

[표] mosaic augmentation 전후의 model의 성능차이

![[이미지] baseline과 mosaic augmentation이 적용된 모델의 RPN classification loss](images/Untitled%2024.png)

[이미지] baseline과 mosaic augmentation이 적용된 모델의 RPN classification loss

![[이미지] bounding box 사이즈별 baseline과 mosaic augmentation의 성능차이 (x축 epochs)](images/Untitled%2025.png)

[이미지] bounding box 사이즈별 baseline과 mosaic augmentation의 성능차이 (x축 epochs)

![[이미지] yolov7-e6e 모델에서 mosaic 파라미터 0.3 vs 1.0 성능 차이](images/Untitled%2026.png)

[이미지] yolov7-e6e 모델에서 mosaic 파라미터 0.3 vs 1.0 성능 차이

![Untitled](images/Untitled%2027.png)

- 동일한 epoch에서는 mosaic augmentation을 적용한 model의 train mAP가 감소하였음.
- 동일한 epoch에서는 mosaic augmentation을 적용한 model의 train, validation 성능의 gap이 적었음.
(generalization의 성능이 향상됨)
- middle, large size bbox에서의 성능이 mosaic augmentation에서 상승하였음.
- rpn classification loss는 baseline과 mosaic간 차이가 나지 않았음.
- yolov7에서는 mosaic augmentation 을 쎄게 줄 수록 학습 속도가 더 빨랐고 제출 성능도 더 좋았다. 
아직 학습이 더 될 기미가 보여 10 epoch 정도를 더 늘려서 다시 실험해보고 싶었지만, 
시간이 부족한 관계로 성능 개선이 확실하다고 판단한 pseudo 실험으로 넘어갔다.

####결과 분석

Mosaic augmentation을 적용 후 모델의 성능은 아래와 같이 큰폭으로 증가하였다.

가설과는 다르게 rpn의 classification loss는 크게 감소하지 않았기 때문에(baseline과 mosaic aug의 rpn cls loss 차이 이미지)
background를 잘 구분하는 효과는 없었다고 생각됨.

하지만, mosaic이 된 데이터를 학습에 이용한 model의 경우 bounding box의 사이즈별로 성능의 차이가 큰것을 확인할 수 있었으며,
이는 mosaic augmentation의 경우 다양한 bounding box의 사이즈를 예측하는데 영향을 미친것으로 보임.

하지만, 실험과 같이 조금 더 긴 epoch에서의 결과를 비교한다면 조금 더 정확한 비교를 할 수 있지 않았을까 생각됨.
또한 위 가설을 결론을 검증하기 위해 RandomResizedCrop을 이용해 추가 실험을 진행하면 더 정확한 결론을 얻을 수 있을것이라 생각됨.


###Mixup Augmentation
####가설

앞서 mosaic augmentation의 유효한 효과를 확인 후, bbox의 사이즈를 자유롭게 조정하는 형태의 augmentation,
합성(synthesized) 이미지 유형의 augmentation이 효과가 있을것이라는 가설을 세웠음.

![[이미지] mixup augmentation이 적용된 이미지들](images/Untitled%2028.png)

[이미지] mixup augmentation이 적용된 이미지들


####실험 결과

![[표] mixup 적용 전후의 성능 차이](images/Untitled%2029.png)

[표] mixup 적용 전후의 성능 차이

![[이미지] baseline, mixup, mosaic이 적용된 모델들의 validation score](images/Untitled%2030.png)

[이미지] baseline, mixup, mosaic이 적용된 모델들의 validation score

- mixup augmentation이 적용된 후 model의 train mAP와 validation mAP는 하락하였음.

####결과 분석

mixup 적용 후 train과 validation mAP의 성능은 크게 하락하였음.
조금더 epoch가 긴 상태에서 분석을 진행하였으면 더 정확한 비교가 되었을것이라는 아쉬움이 있었으며,

transparent한 이미지가 test set에 존재하지 않아 악영향을 미쳤을 수도 있다는 생각이 듬.
추가적으로 찾아본 Cutmix 논문([https://arxiv.org/pdf/1905.04899.pdf](https://arxiv.org/pdf/1905.04899.pdf))에서도 우리의 실험과 유사하게
mixup을 적용할 경우 성능이 소폭 하락하는 것을 확인할 수 있었음.

![Untitled](images/Untitled%2031.png)