# Detector
## Anchor Generator
### Anchor Scale Experiment
#### baseline model

- Detector : Faster R-CNN
- Backbone : ResNeXt-101
- Neck : FPN
- Image size : 512

변인 : Anchor Generator의 Scale


#### 가설
작은 물체들을 더 잘 예측하기 위해 다양한 Scale을 사용하면 성능이 더 올라가지 않을까?

General Trash에 대한 성능이 제일 낮았고, 특히 사람도 알아보기 힘든 작은 크기의 General Trash 가 존재했다. 작은 크기의 General Trash를 틀리는 경우를 보완하고자 Scale 파라미터에 더 작은 값들을 추가해 성능 변화를 실험해보고자 했다. 기존 Scale [8] 에서 Scale [2, 4, 8], Scale [4, 8] 를 추가적으로 실험했다.


#### 실험 결과

![Untitled](images/Untitled%2011.png)


#### 결과 분석

- 작은 물체를 더 잘맞추게 하기위해 Scale 을 조절하였지만, 의도와는 반대로 mAP_s 지표는 오히려 baseline model이 가장 성능이 좋았다.
- mAP_s로 분류하는 기준 넓이는 1024로 이보다 작은면 s로 분류하는데, 1024 보다 작은 bbox는 매우 적어 mAP_s의 값도 0.008 로 매우 낮은 수치로 나오는 것 같다. 해당 지표는 무시하기로 판단했다.
- small: 0, 1024, medium: 1024, 9216, large: 9216, 10000000000
- mAP_m 수치를 보면 scale [4, 8] 의 성능이 가장 좋고, scale [2, 4, 8] 은 오히려 떨어진 것을 볼 수 있다. → scale 2 는 오히려 성능을 하락시킨다고 해석했다.

scale [4, 8] 이 적당한 파라미터인 것 같다.