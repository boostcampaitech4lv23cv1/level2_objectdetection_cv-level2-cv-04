# Detector
## 1-stage
### YOLOv7 성능 확인 실험
#### 가설

YOLOv7-E6E 모델의 성능이 좋지 않을까?

앞선 실험에서 이미지의 사이즈는 성능에 중요한 영향을 끼친다는 사실을 실험을 통해서 검증했다. 더 큰 이미지 사이즈를 권장하는 모델을 사용하는 것이 성능이 더 좋을 것이다.

추가적으로 output layer로 P3, P4, P5 를 사용하는 기본 모델보다 P6를 추가적으로 사용하는 모델들 중에서도 가장 큰 모델인 YOLOv7-E6E 모델을 사용하는 것이 성능이 더 잘 나올 것이다.

![Untitled](images/Untitled%203.png)


#### 실험 결과

|  | YOLOv7 | YOLOv7-E6E |
| --- | --- | --- |
| Test mAP | 0.4063 | 0.5996 |

#### 결과 분석
현재까지 단일 모델 기준 SOTA

- image size가 달라서 모델끼리의 비교에 있어 정확한 변인 통제가 일어나지 않았지만, YOLOv7-E6E의 성능이 다른 단일 모델과 비교해도 좋은 성능을 보였다.
- confusion matrix

![Untitled](images/Untitled%204.png)

- General trash
    - Background로 간주하는 경우가 다른 class에 비해 많다.
- Battery
    - Battery 를 Metal로 인지하는 경우가 많았다.
- Prediction vs Ground Truth
- Ground Truth 에서 Bbox가 많이 뭉쳐있어도 신기하게도 잘 골라낸다.

**Prediction**

![Untitled](images/Untitled%205.png)

![Untitled](images/Untitled%206.png)

**Ground Truth**

![Untitled](images/Untitled%207.png)

![Untitled](images/Untitled%208.png)