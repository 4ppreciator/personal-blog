# 기계학습 영역 학습 자료

## 1. 데이터 전처리

### 개념
데이터 전처리는 원시 데이터를 분석에 적합한 형태로 변환하는 과정입니다. 이 과정은 데이터 분석 및 모델링의 성능에 큰 영향을 미치므로 매우 중요합니다. 주요 전처리 작업으로는 결측값 처리, 이상값 탐지 및 수정, 데이터 정규화/표준화, 파생 변수 생성, 변수 선택, 데이터 분할 등이 있습니다.

### Python 코드 예시

#### 결측값 처리
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('sample_data.csv')

# 결측값 확인
print("결측값 개수:")
print(df.isnull().sum())

# 결측값 처리 방법 1: 삭제
df_dropped = df.dropna()  # 결측값이 있는 행 삭제

# 결측값 처리 방법 2: 대체
df_filled_mean = df.fillna(df.mean())  # 평균값으로 대체
df_filled_median = df.fillna(df.median())  # 중앙값으로 대체
df_filled_mode = df.fillna(df.mode().iloc[0])  # 최빈값으로 대체

# 결측값 처리 방법 3: 보간
df_interpolated = df.interpolate(method='linear')  # 선형 보간
```

#### 이상값 탐지 및 수정
```python
# 이상값 탐지 방법 1: Z-score
from scipy import stats

z_scores = stats.zscore(df['numeric_column'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)  # Z-score가 3 이상인 값을 이상값으로 간주
df_no_outliers = df[filtered_entries]

# 이상값 탐지 방법 2: IQR(Interquartile Range)
Q1 = df['numeric_column'].quantile(0.25)
Q3 = df['numeric_column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_no_outliers_iqr = df[(df['numeric_column'] >= lower_bound) & (df['numeric_column'] <= upper_bound)]

# 이상값 시각화
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['numeric_column'])
plt.title('Boxplot for Outlier Detection')
plt.show()
```

#### 데이터 정규화/표준화
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 정규화 (Min-Max Scaling): 데이터를 0~1 범위로 변환
min_max_scaler = MinMaxScaler()
df_normalized = min_max_scaler.fit_transform(df[['numeric_column1', 'numeric_column2']])

# 표준화 (Z-score Standardization): 평균 0, 표준편차 1로 변환
standard_scaler = StandardScaler()
df_standardized = standard_scaler.fit_transform(df[['numeric_column1', 'numeric_column2']])
```

#### 파생 변수 생성
```python
# 날짜 데이터에서 파생 변수 생성
df['date'] = pd.to_datetime(df['date_column'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek

# 수치형 데이터에서 파생 변수 생성
df['ratio'] = df['numeric_column1'] / df['numeric_column2']
df['log_transform'] = np.log1p(df['numeric_column'])  # log(1+x) 변환
df['squared'] = df['numeric_column'] ** 2
```

#### 변수 선택 (VIF를 이용한 다중공선성 확인)
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF 계산 함수
def calculate_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif_data

# 변수 선택
features = ['feature1', 'feature2', 'feature3', 'feature4']
vif_result = calculate_vif(df, features)
print(vif_result)

# VIF가 10 이상인 변수 제거 (다중공선성이 높은 변수)
high_vif_features = vif_result[vif_result["VIF"] > 10]["Variable"].tolist()
selected_features = [f for f in features if f not in high_vif_features]
```

#### 데이터 분할 (훈련/테스트 세트)
```python
from sklearn.model_selection import train_test_split

# 독립변수(X)와 종속변수(y) 분리
X = df[selected_features]
y = df['target_variable']

# 훈련/테스트 세트 분할 (테스트 세트 비율 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 출력 결과 해석 예시

#### 결측값 확인 결과 해석
```
결측값 개수:
feature1    10
feature2     5
feature3     0
feature4    15
```
위 결과는 feature1에 10개, feature2에 5개, feature4에 15개의 결측값이 있음을 보여줍니다. feature3에는 결측값이 없습니다. 이를 통해 어떤 변수에 결측값이 많은지 파악하고, 적절한 처리 방법을 선택할 수 있습니다.

#### VIF 계산 결과 해석
```
   Variable        VIF
0  feature1   2.345678
1  feature2  15.678901
2  feature3   3.456789
3  feature4   8.901234
```
VIF(Variance Inflation Factor)는 다중공선성을 측정하는 지표로, 일반적으로 10 이상이면 다중공선성이 높다고 판단합니다. 위 결과에서 feature2의 VIF가 15.68로 높게 나타났으므로, 이 변수는 다른 변수들과 높은 상관관계가 있어 모델에서 제외하는 것이 좋을 수 있습니다.

## 2. 탐색적 데이터 분석(EDA)

### 개념
탐색적 데이터 분석(Exploratory Data Analysis, EDA)은 데이터의 주요 특성을 파악하기 위해 다양한 기법을 사용하여 데이터를 요약하고 시각화하는 과정입니다. EDA를 통해 데이터의 패턴, 이상점, 관계 등을 발견하고, 이를 바탕으로 가설을 설정하거나 적절한 분석 방법을 선택할 수 있습니다.

### Python 코드 예시

#### 기초 통계량 분석
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('sample_data.csv')

# 데이터 기본 정보 확인
print("데이터 크기:", df.shape)
print("\n데이터 타입:")
print(df.dtypes)

# 기술 통계량 확인
print("\n기술 통계량:")
print(df.describe())

# 범주형 변수의 빈도 확인
print("\n범주형 변수 빈도:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col} 빈도:")
    print(df[col].value_counts())
```

#### 데이터 시각화
```python
# 히스토그램
plt.figure(figsize=(12, 6))
for i, col in enumerate(df.select_dtypes(include=['int64', 'float64']).columns[:4]):
    plt.subplot(2, 2, i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()

# 상자그림(Boxplot)
plt.figure(figsize=(12, 6))
for i, col in enumerate(df.select_dtypes(include=['int64', 'float64']).columns[:4]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# 산점도(Scatter plot)
plt.figure(figsize=(10, 8))
sns.scatterplot(x='feature1', y='feature2', hue='target_variable', data=df)
plt.title('Scatter plot of feature1 vs feature2')
plt.show()

# 상관관계 히트맵
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 쌍별 산점도(Pairplot)
sns.pairplot(df, hue='target_variable')
plt.show()
```

### 출력 결과 해석 예시

#### 기초 통계량 해석
```
데이터 크기: (1000, 10)

데이터 타입:
feature1       float64
feature2       float64
feature3       float64
feature4       float64
category1      object
category2      object
target_variable  int64

기술 통계량:
         feature1     feature2     feature3     feature4  target_variable
count  990.000000  995.000000  1000.000000  985.000000      1000.000000
mean     0.123456    2.345678     3.456789    4.567890         0.600000
std      1.234567    2.345678     3.456789    4.567890         0.490000
min     -5.678901   -3.456789    -2.345678   -1.234567         0.000000
25%     -0.987654    0.987654     1.987654    2.987654         0.000000
50%      0.123456    2.345678     3.456789    4.567890         1.000000
75%      1.234567    3.456789     4.567890    5.678901         1.000000
max      6.789012    8.901234    10.123456   12.345678         1.000000
```
위 결과는 데이터의 기본 정보와 기술 통계량을 보여줍니다. 데이터는 1000개의 행과 10개의 열로 구성되어 있으며, 수치형 변수 4개, 범주형 변수 2개, 그리고 타겟 변수 1개가 있습니다. 기술 통계량을 통해 각 변수의 평균, 표준편차, 최소값, 최대값 등을 확인할 수 있습니다. 또한 결측값이 있는 변수도 확인할 수 있습니다(count가 1000보다 작은 경우).

#### 상관관계 히트맵 해석
상관관계 히트맵은 변수 간의 상관계수를 시각적으로 보여줍니다. 상관계수가 1에 가까울수록 강한 양의 상관관계, -1에 가까울수록 강한 음의 상관관계, 0에 가까울수록 상관관계가 없음을 의미합니다. 히트맵을 통해 어떤 변수들이 서로 강한 상관관계를 가지는지 파악하고, 이를 바탕으로 변수 선택이나 다중공선성 문제를 해결할 수 있습니다.

## 3. 기계학습 알고리즘을 활용한 모델 구축

### 개념
기계학습 알고리즘은 데이터로부터 패턴을 학습하여 예측이나 분류를 수행하는 알고리즘입니다. 주요 기계학습 알고리즘으로는 회귀 분석, 분류 분석, 군집화 분석, 차원 축소 등이 있습니다. 각 알고리즘은 특정 문제 유형에 적합하며, 데이터의 특성과 목적에 따라 적절한 알고리즘을 선택해야 합니다.

### Python 코드 예시

#### 회귀 분석
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

# 데이터 로드 및 분할
df = pd.read_csv('sample_data.csv')
X = df[['feature1', 'feature2', 'feature3', 'feature4']]
y = df['target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
print(f'R-squared: {r2:.4f}')

# 회귀 계수 확인
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': linear_model.coef_
})
print("\n회귀 계수:")
print(coefficients)
```

#### 분류 분석
```python
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 로지스틱 회귀
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
y_pred_proba_logistic = logistic_model.predict_proba(X_test)[:, 1]

# SVM
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_pred_proba_svm = svm_model.predict_proba(X_test)[:, 1]

# 랜덤 포레스트
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# 모델 평가
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    
    print(f'\n{model_name} 평가 결과:')
    print(f'정확도: {accuracy:.4f}')
    print(f'\n혼동 행렬:')
    print(conf_matrix)
    print(f'\n분류 보고서:')
    print(class_report)

evaluate_model(y_test, y_pred_logistic, '로지스틱 회귀')
evaluate_model(y_test, y_pred_svm, 'SVM')
evaluate_model(y_test, y_pred_rf, '랜덤 포레스트')
```

#### 군집화 분석
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 최적의 군집 수 찾기
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f'k={k}: 실루엣 점수 = {silhouette_avg:.4f}')

# 실루엣 점수 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('군집 수 (k)')
plt.ylabel('실루엣 점수')
plt.title('군집 수에 따른 실루엣 점수')
plt.show()

# 최적의 군집 수로 K-means 수행
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# 군집 결과 시각화 (2차원으로 축소하여 시각화)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.colorbar(scatter)
plt.xlabel('주성분 1')
plt.ylabel('주성분 2')
plt.title(f'K-means 군집화 결과 (k={optimal_k})')
plt.show()
```

### 출력 결과 해석 예시

#### 선형 회귀 모델 평가 해석
```
Mean Squared Error: 0.2345
R-squared: 0.7890

회귀 계수:
    Feature  Coefficient
0  feature1      0.5678
1  feature2      0.1234
2  feature3     -0.3456
3  feature4      0.7890
```
위 결과는 선형 회귀 모델의 성능과 회귀 계수를 보여줍니다. MSE(Mean Squared Error)는 0.2345로 예측값과 실제값의 차이가 작음을 의미합니다. R-squared는 0.7890으로 모델이 데이터의 약 78.9%를 설명할 수 있음을 의미합니다. 회귀 계수를 통해 각 변수가 타겟 변수에 미치는 영향을 확인할 수 있습니다. 예를 들어, feature1이 1단위 증가할 때 타겟 변수는 0.5678단위 증가하며, feature3이 1단위 증가할 때 타겟 변수는 0.3456단위 감소합니다.

#### 분류 모델 평가 해석
```
로지스틱 회귀 평가 결과:
정확도: 0.8500

혼동 행렬:
[[85 15]
 [10 90]]

분류 보고서:
              precision    recall  f1-score   support

           0       0.89      0.85      0.87       100
           1       0.86      0.90      0.88       100

    accuracy                           0.88       200
   macro avg       0.88      0.88      0.88       200
weighted avg       0.88      0.88      0.88       200
```
위 결과는 로지스틱 회귀 모델의 성능을 보여줍니다. 정확도는 0.85로 모델이 테스트 데이터의 85%를 정확히 분류했음을 의미합니다. 혼동 행렬은 실제 클래스와 예측 클래스의 관계를 보여줍니다. 예를 들어, 실제 클래스가 0인 100개의 샘플 중 85개는 정확히 0으로 예측되었고, 15개는 잘못 예측되었습니다. 분류 보고서는 각 클래스별 정밀도(precision), 재현율(recall), F1-score를 보여줍니다. 이를 통해 모델이 각 클래스를 얼마나 잘 분류하는지 확인할 수 있습니다.

## 4. 모델 평가 및 최적 모델 선정

### 개념
모델 평가는 구축한 모델의 성능을 측정하고, 여러 모델 중 최적의 모델을 선정하는 과정입니다. 회귀 모델의 경우 MSE, RMSE, MAE, R-squared 등의 지표를, 분류 모델의 경우 정확도, 정밀도, 재현율, F1-score, AUC 등의 지표를 사용합니다. 또한 교차 검증을 통해 모델의 일반화 성능을 평가하고, 하이퍼파라미터 튜닝을 통해 모델의 성능을 최적화할 수 있습니다.

### Python 코드 예시

#### 교차 검증
```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 데이터 로드
df = pd.read_csv('sample_data.csv')
X = df[['feature1', 'feature2', 'feature3', 'feature4']]
y = df['target_variable']

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 여러 모델에 대한 교차 검증
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    print(f'{name} 교차 검증 정확도: {scores.mean():.4f} (±{scores.std():.4f})')
```

#### 하이퍼파라미터 튜닝
```python
from sklearn.model_selection import GridSearchCV

# 로지스틱 회귀 하이퍼파라미터 튜닝
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X, y)

print("로지스틱 회귀 최적 하이퍼파라미터:", grid_search_lr.best_params_)
print("로지스틱 회귀 최고 정확도:", grid_search_lr.best_score_)

# SVM 하이퍼파라미터 튜닝
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

grid_search_svm = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X, y)

print("\nSVM 최적 하이퍼파라미터:", grid_search_svm.best_params_)
print("SVM 최고 정확도:", grid_search_svm.best_score_)

# 랜덤 포레스트 하이퍼파라미터 튜닝
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X, y)

print("\n랜덤 포레스트 최적 하이퍼파라미터:", grid_search_rf.best_params_)
print("랜덤 포레스트 최고 정확도:", grid_search_rf.best_score_)
```

#### ROC 곡선 및 AUC
```python
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 최적 모델로 예측 확률 계산
best_model_lr = grid_search_lr.best_estimator_
best_model_svm = grid_search_svm.best_estimator_
best_model_rf = grid_search_rf.best_estimator_

best_model_lr.fit(X_train, y_train)
best_model_svm.fit(X_train, y_train)
best_model_rf.fit(X_train, y_train)

y_proba_lr = best_model_lr.predict_proba(X_test)[:, 1]
y_proba_svm = best_model_svm.predict_proba(X_test)[:, 1]
y_proba_rf = best_model_rf.predict_proba(X_test)[:, 1]

# ROC 곡선 그리기
plt.figure(figsize=(10, 8))

# 로지스틱 회귀
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.4f})')

# SVM
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
auc_svm = auc(fpr_svm, tpr_svm)
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.4f})')

# 랜덤 포레스트
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})')

# 기준선
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# 최적 모델 선정
auc_scores = {
    'Logistic Regression': auc_lr,
    'SVM': auc_svm,
    'Random Forest': auc_rf
}

best_model_name = max(auc_scores, key=auc_scores.get)
print(f'최적 모델: {best_model_name} (AUC = {auc_scores[best_model_name]:.4f})')
```

### 출력 결과 해석 예시

#### 교차 검증 결과 해석
```
Logistic Regression 교차 검증 정확도: 0.8450 (±0.0234)
SVM 교차 검증 정확도: 0.8750 (±0.0187)
Random Forest 교차 검증 정확도: 0.8900 (±0.0141)
```
위 결과는 세 가지 모델의 5-fold 교차 검증 정확도를 보여줍니다. 랜덤 포레스트 모델이 평균 정확도 0.89로 가장 높은 성능을 보이며, 표준편차도 0.0141로 가장 낮아 안정적인 성능을 보입니다. 따라서 세 모델 중 랜덤 포레스트가 가장 좋은 성능을 보인다고 할 수 있습니다.

#### 하이퍼파라미터 튜닝 결과 해석
```
로지스틱 회귀 최적 하이퍼파라미터: {'C': 1, 'penalty': 'l2', 'solver': 'liblinear'}
로지스틱 회귀 최고 정확도: 0.8550

SVM 최적 하이퍼파라미터: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
SVM 최고 정확도: 0.8850

랜덤 포레스트 최적 하이퍼파라미터: {'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
랜덤 포레스트 최고 정확도: 0.9050
```
위 결과는 각 모델의 최적 하이퍼파라미터와 그때의 정확도를 보여줍니다. 랜덤 포레스트 모델이 최적 하이퍼파라미터 설정에서 0.905의 정확도로 가장 높은 성능을 보입니다. 이를 통해 최종 모델로 랜덤 포레스트를 선택하고, 해당 하이퍼파라미터 설정을 사용하는 것이 좋다고 판단할 수 있습니다.

#### ROC 곡선 및 AUC 해석
ROC(Receiver Operating Characteristic) 곡선은 다양한 임계값에서의 True Positive Rate(TPR)과 False Positive Rate(FPR)의 관계를 보여주는 그래프입니다. AUC(Area Under the Curve)는 ROC 곡선 아래 영역의 넓이로, 1에 가까울수록 좋은 성능을 의미합니다. 위 예시에서 랜덤 포레스트 모델의 AUC가 가장 높다면, 이 모델이 가장 좋은 성능을 보인다고 할 수 있습니다. 또한 ROC 곡선이 왼쪽 위 모서리에 가까울수록 좋은 성능을 의미하므로, 곡선의 형태도 함께 고려해야 합니다.
