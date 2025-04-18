# ADP 실기 시험 문제 유형 분류

## 기계학습 영역

### 1. 데이터 전처리
- **결측값 처리**: 데이터셋의 결측치를 적절한 방법으로 처리하는 문제
- **이상값 탐지 및 수정**: 데이터셋에서 이상값을 찾고 처리하는 문제
- **데이터 정규화/표준화**: 데이터의 스케일을 조정하는 문제
- **파생 변수 생성**: 기존 변수를 활용하여 새로운 변수를 생성하는 문제
- **변수 선택**: 다중공선성(VIF) 등을 고려한 변수 선택 문제
- **데이터 분할**: 훈련/테스트 데이터셋 분할 문제

### 2. 탐색적 데이터 분석(EDA)
- **기초 통계량 분석**: 데이터의 평균, 분산, 중앙값 등 기초 통계량 계산 문제
- **데이터 시각화**: 히스토그램, 산점도, 상자그림 등을 활용한 시각화 문제
- **데이터 분포 확인**: 데이터의 분포 특성을 파악하는 문제

### 3. 기계학습 알고리즘을 활용한 모델 구축
- **회귀 분석**: 선형 회귀, 로지스틱 회귀 등을 활용한 모델 구축 문제
- **분류 분석**: SVM, 의사결정나무, 랜덤 포레스트 등을 활용한 분류 모델 구축 문제
- **군집화 분석**: K-means 등을 활용한 군집화 문제
- **차원 축소**: PCA 등을 활용한 차원 축소 문제

### 4. 모델 평가 및 최적 모델 선정
- **평가 지표 활용**: 정확도, 정밀도, 재현율, F1-score 등을 활용한 모델 평가 문제
- **교차 검증**: K-fold 교차 검증 등을 활용한 모델 평가 문제
- **하이퍼파라미터 튜닝**: 그리드 서치, 랜덤 서치 등을 활용한 하이퍼파라미터 최적화 문제
- **앙상블 기법**: 배깅, 부스팅 등을 활용한 모델 성능 향상 문제

## 통계 영역

### 1. 기초 통계 분석
- **기술통계량 계산**: 평균, 분산, 표준편차, 중앙값 등 계산 문제
- **히스토그램 및 상자그림 시각화**: 데이터 분포 시각화 문제
- **확률 분포**: 정규분포, 이항분포, 포아송분포 등 확률분포 관련 문제

### 2. 통계적 추론 및 통계 모형 구축
- **가설 검정**: 귀무가설과 대립가설 설정, 검정통계량 계산, 유의확률 해석 문제
- **신뢰구간 계산**: 모수에 대한 신뢰구간 계산 문제
- **정규 모집단에서의 추론**: 정규분포 가정 하에서의 통계적 추론 문제

### 3. 상관 분석 및 회귀 분석
- **상관 분석**: 변수 간 상관관계 분석 문제
- **단순 회귀 분석**: 한 개의 독립변수를 사용한 회귀분석 문제
- **다중 회귀 분석**: 여러 개의 독립변수를 사용한 회귀분석 문제

### 4. 다변량 분석 및 시계열 분석
- **분산분석(ANOVA)**: 집단 간 평균 차이 검정 문제
- **범주형 자료분석**: 카이제곱 검정, 교차분석 등 범주형 자료 분석 문제
- **시계열 분석**: 시간에 따른 데이터 패턴 분석 문제
- **관리도 분석**: 품질관리를 위한 관리도 작성 및 해석 문제

## ADP 15회 실기 문제 분류 예시

### 기계학습 영역
- 데이터 전처리: 변수 선택(VIF), 파생변수 생성, 데이터 분할(train/test)
- 탐색적 데이터 분석: EDA와 시각화 및 통계량 제시
- 기계학습 알고리즘 모델 구축: 로지스틱 회귀 분석, SVM 등 3가지 알고리즘 평가
- 모델 평가: confusionMatrix 확인, F1-score 비교
- 군집화 분석: 최적의 군집수와 군집 레이블 구하기

### 통계 영역
- 데이터 전처리 및 집계: 총사용량을 연월별 총합으로 계산
- 시각화: 요일별 평균 전력사용량 그래프 작성
- 통계적 추론: 요일별 각 유형의 평균 전력 사용량 간 연관성 검정
- 상관 분석: 전력사용량과 온도와의 상관계수 계산
