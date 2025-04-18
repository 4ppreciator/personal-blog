# ADP 15회 실기 문제 정리

## 1번 문제 (기계학습 영역)

### 데이터 확인 및 전처리
- 철강데이터 종속변수: target
- 데이터 출처: https://www.kaggle.com/uciml/faulty-steel-plates

### 문제 유형 분류
1. **데이터 전처리 및 EDA**
   - 1.1 EDA와 시각화 및 통계량 제시
   - 1.2 변수 선택(VIF), 파생변수 생성, 데이터 분할(train/test(20%)), 시각화와 통계량 제시

2. **기계학습 알고리즘을 활용한 모델 구축**
   - 1.3 종속변수들 중 "1"인지 아닌지 판단하는 로지스틱 회귀 분석 실시
   - 1.4 종속변수(y)를 다항(7 class)인 상태에서 SVM을 포함한 3가지 알고리즘으로 평가

3. **모델 평가 및 최적 모델 선정**
   - 1.3 confusionMatrix 확인 및 cut off value 결정
   - 1.5 군집 레이블을 추가한 데이터를 이전 모델에 다시 학습하여 F1-score 비교

4. **군집화 분석**
   - 1.5 종속변수를 제외한 나머지 데이터를 바탕으로 군집분석 실시, 최적의 군집수와 군집 레이블 구하기

## 2번 문제 (통계 영역)

### 데이터 설명
- 데이터 출처: 직접제작
- 데이터 설명: 2050년 1년동안의 5유형(A,B,C,D,E)의 전력사용량
- 각 유형의 전력사용량은 1분마다 갱신되며 누적됨
- 6시간마다(00:00, 06:00, 12:00, 18:00시에) 전력사용량은 0으로 초기화됨
- 제공 파일:
  - problem2_usage.csv: 6시간 간격의 총 전력사용량 데이터
  - problem2_usage_history.csv: 1분간격의 A,B,C,D,E 유형의 소비 누적 전력 데이터
  - problem2_avg_tem.csv: 2050년 1년동안 일자별 평균 온도 데이터

### 문제 유형 분류
1. **데이터 전처리 및 집계**
   - 2-1번: usage의 총사용량을 연월별 총합으로 계산하여 CSV 파일로 작성

2. **시각화**
   - 2-2번: 요일별 평균 전력사용량을 나타내는 그래프 작성 (각 유형별로 색을 다르게 표현)

3. **통계적 추론 및 통계 모형 구축**
   - 2-3번: 요일별 각 유형의 평균 전력 사용량 간에 연관성이 있는지 검정

4. **상관 분석**
   - 2-4번: 일자마다 각 유형의 전력사용량의 합을 구하고 온도와의 상관계수 계산
