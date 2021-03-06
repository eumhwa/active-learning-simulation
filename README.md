# Active learning simulation
## CNN feature based active learning

### 1. Sampling Method
#### 1.1 entropy sampling
uncertainty-based approach로 모형의 예측확률을 이용하여 entropy를 계산하고,
entropy가 큰(uncertainty가 큰) 데이터를 선택한다.
단, 분류 문제에서 주로 사용하는 cross-entropy loss를 이용하여 DL모형을 학습시키면, 
수식에서 알 수 있듯이 정답을 맞추는 것을 넘어 예측 확률값을 극단으로 보내게 되고
이는 over confidence문제를 야기한다. 이는 OOD 데이터에 대해서도 과잉확신을 하는 등 
모형의 신뢰성을 떨어뜨리는 요인이다.
향후 label smoothing, calibration등의 방법을 통한 실험도 진행이 필요하다.

#### 1.2 core-set selection (k-center greedy)
diversity-based approach로 CNN layer를 통해 임베딩된 feature 공간 상에서
data point들을 아우르는 k개의 core-set을 선택하는 방법으로
각 core-set은 radius 내 인접한 data point를 대표한다.
k-center greedy 알고리즘을 구현하여 실험을 진행한다.

### 2. Simulation Process
#### 2.1 아래 절차를 N회 반복
    1) train/val/test 분할
    2) train 이용하여 baseline CNN 학습 및 test 성능 확인
    3) val에 대해 active learning 과 random sampling 각각 적용
    4) 위에서 추출한 2가지 sampled data를 train에 각각 추가하여 2개 모형 학습
    5) 2개 모형 test 성능 비교

#### 2.2 class imbalanced setting하에서도 test 수행

### 3. Datasets
kaggle의 flower dataset을 이용한다. 102개의 클래스 중 일부 클래스만 이용한다.