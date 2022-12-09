#Validation Dataset Search
하루 10회의 제출 제한이 있었기 때문에,
우리는 test data의 점수와 유사한 validation data set을 찾는 활동을 수행하였음.

우리는 sklearn의 `Stratified Group KFold`를 이용하여 5개의 train, validation dataset을 만들고,
제출하여 test mAP의 차이를 확인하였음.

![[표] 제출한 5개의 validation set의 validation mAP와 test mAP의 차이](images/Untitled%2032.png)

[표] 제출한 5개의 validation set의 validation mAP와 test mAP의 차이

실험결과 case 3의 valid set이 가장 test set의 성능과 유사했으며.
우리는 case 3의 validation set을 기반으로 프로젝트를 진행하였음.

추가로 validation 탐색 이후 실제 제출한 valid와 test mAP의 차이는 아래 그림과 같이,
신뢰도가 매우 높았으며, 때문에 우리는 모든 dataset을 이용한 경우를 제외하고
그 외 실험들의 결과를 효과적으로 예측 가능했음.

![[표] 추가로 진행한 실험들에서 valid mAP와 test mAP의 차이](images/Untitled%2033.png)

[표] 추가로 진행한 실험들에서 valid mAP와 test mAP의 차이