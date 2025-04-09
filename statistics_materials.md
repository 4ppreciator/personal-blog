# 통계 영역 학습 자료

## 1. 기초 통계 분석

### 개념
기초 통계 분석은 데이터의 기본적인 특성을 파악하기 위한 분석 방법입니다. 주요 내용으로는 기술통계량 계산, 데이터 분포 시각화, 확률 분포 이해 등이 포함됩니다. 이를 통해 데이터의 중심 경향성, 분산, 분포 형태 등을 파악할 수 있습니다.

### Python 코드 예시

#### 기술통계량 계산
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 데이터 로드
df = pd.read_csv('sample_data.csv')

# 기술통계량 계산
desc_stats = df.describe()
print("기술통계량:")
print(desc_stats)

# 추가 기술통계량 계산
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    print(f"\n{col} 통계량:")
    data = df[col].dropna()
    print(f"평균: {np.mean(data):.4f}")
    print(f"중앙값: {np.median(data):.4f}")
    print(f"최빈값: {stats.mode(data)[0][0]:.4f}")
    print(f"표준편차: {np.std(data):.4f}")
    print(f"분산: {np.var(data):.4f}")
    print(f"왜도: {stats.skew(data):.4f}")
    print(f"첨도: {stats.kurtosis(data):.4f}")
    print(f"최소값: {np.min(data):.4f}")
    print(f"최대값: {np.max(data):.4f}")
    print(f"범위: {np.max(data) - np.min(data):.4f}")
    print(f"사분위수 범위(IQR): {np.percentile(data, 75) - np.percentile(data, 25):.4f}")
```

#### 히스토그램 및 상자그림 시각화
```python
# 히스토그램
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.select_dtypes(include=['int64', 'float64']).columns[:4]):
    plt.subplot(2, 2, i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} 히스토그램')
    plt.axvline(df[col].mean(), color='r', linestyle='--', label='평균')
    plt.axvline(df[col].median(), color='g', linestyle='-.', label='중앙값')
    plt.legend()
plt.tight_layout()
plt.show()

# 상자그림
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=['int64', 'float64']).iloc[:, :4])
plt.title('상자그림')
plt.xticks(rotation=45)
plt.show()

# Q-Q 플롯 (정규성 확인)
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.select_dtypes(include=['int64', 'float64']).columns[:4]):
    plt.subplot(2, 2, i+1)
    stats.probplot(df[col].dropna(), dist="norm", plot=plt)
    plt.title(f'{col} Q-Q 플롯')
plt.tight_layout()
plt.show()
```

#### 확률 분포
```python
# 정규분포
x = np.linspace(-4, 4, 1000)
plt.figure(figsize=(12, 8))

# 표준정규분포
plt.subplot(2, 2, 1)
plt.plot(x, stats.norm.pdf(x, 0, 1))
plt.title('표준정규분포 (μ=0, σ=1)')
plt.grid(True)

# 다양한 평균을 가진 정규분포
plt.subplot(2, 2, 2)
for mu, color in zip([-2, 0, 2], ['r', 'g', 'b']):
    plt.plot(x, stats.norm.pdf(x, mu, 1), color=color, label=f'μ={mu}, σ=1')
plt.title('다양한 평균을 가진 정규분포')
plt.legend()
plt.grid(True)

# 다양한 표준편차를 가진 정규분포
plt.subplot(2, 2, 3)
for sigma, color in zip([0.5, 1, 2], ['r', 'g', 'b']):
    plt.plot(x, stats.norm.pdf(x, 0, sigma), color=color, label=f'μ=0, σ={sigma}')
plt.title('다양한 표준편차를 가진 정규분포')
plt.legend()
plt.grid(True)

# 이항분포
plt.subplot(2, 2, 4)
n_values = [10, 20, 50]
p = 0.5
x_binomial = np.arange(0, max(n_values) + 1)
for n, color in zip(n_values, ['r', 'g', 'b']):
    pmf = [stats.binom.pmf(k, n, p) for k in x_binomial[:n+1]]
    plt.plot(x_binomial[:n+1], pmf, 'o-', color=color, label=f'n={n}, p={p}')
plt.title('이항분포')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 포아송분포
plt.figure(figsize=(10, 6))
x_poisson = np.arange(0, 20)
for lambda_val, color in zip([1, 4, 10], ['r', 'g', 'b']):
    pmf = [stats.poisson.pmf(k, lambda_val) for k in x_poisson]
    plt.plot(x_poisson, pmf, 'o-', color=color, label=f'λ={lambda_val}')
plt.title('포아송분포')
plt.legend()
plt.grid(True)
plt.show()
```

### 출력 결과 해석 예시

#### 기술통계량 해석
```
기술통계량:
       feature1     feature2     feature3     feature4
count  990.000000  995.000000  1000.000000  985.000000
mean     0.123456    2.345678     3.456789    4.567890
std      1.234567    2.345678     3.456789    4.567890
min     -5.678901   -3.456789    -2.345678   -1.234567
25%     -0.987654    0.987654     1.987654    2.987654
50%      0.123456    2.345678     3.456789    4.567890
75%      1.234567    3.456789     4.567890    5.678901
max      6.789012    8.901234    10.123456   12.345678
```

위 결과는 각 변수의 기술통계량을 보여줍니다:
- count: 결측값이 없는 데이터의 개수
- mean: 평균
- std: 표준편차
- min: 최소값
- 25%: 1사분위수 (하위 25% 지점의 값)
- 50%: 중앙값 (하위 50% 지점의 값)
- 75%: 3사분위수 (하위 75% 지점의 값)
- max: 최대값

이를 통해 데이터의 중심 경향성과 분산을 파악할 수 있습니다. 예를 들어, feature1의 평균은 0.123456이고 표준편차는 1.234567입니다. 또한 결측값의 존재 여부도 확인할 수 있습니다(count가 전체 데이터 수보다 작은 경우).

#### 히스토그램 및 상자그림 해석
히스토그램은 데이터의 분포를 시각적으로 보여줍니다. 종 모양의 대칭적인 분포는 정규분포에 가까움을 의미하며, 왼쪽이나 오른쪽으로 치우친 분포는 각각 음의 왜도(왼쪽 꼬리가 긴 분포)나 양의 왜도(오른쪽 꼬리가 긴 분포)를 가집니다.

상자그림은 데이터의 중앙값, 사분위수, 이상치 등을 보여줍니다. 상자의 중앙선은 중앙값, 상자의 아래와 위는 각각 1사분위수와 3사분위수, 수염은 일반적으로 1.5 * IQR(사분위수 범위) 내의 최소값과 최대값, 그리고 점으로 표시된 것은 이상치를 나타냅니다.

Q-Q 플롯은 데이터가 정규분포를 따르는지 확인하는 데 사용됩니다. 점들이 대각선에 가깝게 위치할수록 데이터가 정규분포에 가깝다고 볼 수 있습니다.

## 2. 통계적 추론 및 통계 모형 구축

### 개념
통계적 추론은 표본 데이터를 바탕으로 모집단의 특성에 대한 결론을 도출하는 과정입니다. 주요 내용으로는 가설 검정, 신뢰구간 계산, 정규 모집단에서의 추론 등이 포함됩니다. 이를 통해 데이터에서 관찰된 패턴이 우연에 의한 것인지, 실제로 의미 있는 것인지를 판단할 수 있습니다.

### Python 코드 예시

#### 가설 검정
```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('sample_data.csv')

# 단일 표본 t-검정 (평균이 특정 값과 다른지 검정)
sample = df['feature1'].dropna()
mu0 = 0  # 귀무가설의 평균값

# 귀무가설: 모집단의 평균은 mu0이다.
# 대립가설: 모집단의 평균은 mu0이 아니다.
t_stat, p_value = stats.ttest_1samp(sample, mu0)

print("단일 표본 t-검정 결과:")
print(f"t-통계량: {t_stat:.4f}")
print(f"p-값: {p_value:.4f}")
print(f"결론: {'귀무가설 기각 (평균이 {mu0}과 다름)' if p_value < 0.05 else '귀무가설 채택 (평균이 {mu0}과 다르다고 할 수 없음)'}")

# 독립 표본 t-검정 (두 집단의 평균이 다른지 검정)
group1 = df[df['category'] == 'A']['feature1'].dropna()
group2 = df[df['category'] == 'B']['feature1'].dropna()

# 등분산 검정 (Levene's test)
levene_stat, levene_p = stats.levene(group1, group2)
print("\n등분산 검정 결과:")
print(f"통계량: {levene_stat:.4f}")
print(f"p-값: {levene_p:.4f}")
print(f"결론: {'등분산 가정 기각 (분산이 다름)' if levene_p < 0.05 else '등분산 가정 채택 (분산이 같다고 할 수 있음)'}")

# 독립 표본 t-검정
# 귀무가설: 두 집단의 평균은 같다.
# 대립가설: 두 집단의 평균은 다르다.
equal_var = levene_p >= 0.05  # 등분산 가정 여부
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)

print("\n독립 표본 t-검정 결과:")
print(f"t-통계량: {t_stat:.4f}")
print(f"p-값: {p_value:.4f}")
print(f"결론: {'귀무가설 기각 (두 집단의 평균이 다름)' if p_value < 0.05 else '귀무가설 채택 (두 집단의 평균이 같다고 할 수 있음)'}")

# 대응 표본 t-검정 (처리 전후의 차이 검정)
before = df['before_treatment'].dropna()
after = df['after_treatment'].dropna()

# 귀무가설: 처리 전후의 평균 차이는 0이다.
# 대립가설: 처리 전후의 평균 차이는 0이 아니다.
t_stat, p_value = stats.ttest_rel(before, after)

print("\n대응 표본 t-검정 결과:")
print(f"t-통계량: {t_stat:.4f}")
print(f"p-값: {p_value:.4f}")
print(f"결론: {'귀무가설 기각 (처리 전후 차이가 있음)' if p_value < 0.05 else '귀무가설 채택 (처리 전후 차이가 없다고 할 수 있음)'}")
```

#### 신뢰구간 계산
```python
# 평균의 95% 신뢰구간 계산
sample = df['feature1'].dropna()
n = len(sample)
mean = np.mean(sample)
std = np.std(sample, ddof=1)  # 표본 표준편차
se = std / np.sqrt(n)  # 표준오차
alpha = 0.05  # 유의수준

# t-분포를 이용한 신뢰구간
t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
margin_of_error = t_critical * se
confidence_interval = (mean - margin_of_error, mean + margin_of_error)

print("\n평균의 95% 신뢰구간:")
print(f"표본 평균: {mean:.4f}")
print(f"표준오차: {se:.4f}")
print(f"t-임계값: {t_critical:.4f}")
print(f"오차 한계: {margin_of_error:.4f}")
print(f"95% 신뢰구간: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")

# 신뢰구간 시각화
plt.figure(figsize=(10, 6))
sns.histplot(sample, kde=True)
plt.axvline(mean, color='r', linestyle='-', label='표본 평균')
plt.axvline(confidence_interval[0], color='g', linestyle='--', label='95% 신뢰구간 하한')
plt.axvline(confidence_interval[1], color='g', linestyle='--', label='95% 신뢰구간 상한')
plt.title('표본 분포와 95% 신뢰구간')
plt.legend()
plt.show()
```

#### 정규 모집단에서의 추론
```python
# 정규성 검정 (Shapiro-Wilk test)
sample = df['feature1'].dropna()
shapiro_stat, shapiro_p = stats.shapiro(sample)

print("\n정규성 검정 결과 (Shapiro-Wilk):")
print(f"통계량: {shapiro_stat:.4f}")
print(f"p-값: {shapiro_p:.4f}")
print(f"결론: {'정규성 가정 기각 (정규분포가 아님)' if shapiro_p < 0.05 else '정규성 가정 채택 (정규분포라고 할 수 있음)'}")

# 분산에 대한 추론 (카이제곱 검정)
sample = df['feature1'].dropna()
n = len(sample)
sample_var = np.var(sample, ddof=1)  # 표본 분산
sigma2_0 = 1.0  # 귀무가설의 분산값

# 귀무가설: 모집단의 분산은 sigma2_0이다.
# 대립가설: 모집단의 분산은 sigma2_0이 아니다.
chi2_stat = (n - 1) * sample_var / sigma2_0
p_value = 2 * min(stats.chi2.cdf(chi2_stat, n-1), 1 - stats.chi2.cdf(chi2_stat, n-1))

print("\n분산에 대한 카이제곱 검정 결과:")
print(f"카이제곱 통계량: {chi2_stat:.4f}")
print(f"p-값: {p_value:.4f}")
print(f"결론: {'귀무가설 기각 (분산이 {sigma2_0}과 다름)' if p_value < 0.05 else '귀무가설 채택 (분산이 {sigma2_0}과 다르다고 할 수 없음)'}")
```

### 출력 결과 해석 예시

#### 가설 검정 결과 해석
```
단일 표본 t-검정 결과:
t-통계량: 2.3456
p-값: 0.0198
결론: 귀무가설 기각 (평균이 0과 다름)

등분산 검정 결과:
통계량: 1.2345
p-값: 0.2678
결론: 등분산 가정 채택 (분산이 같다고 할 수 있음)

독립 표본 t-검정 결과:
t-통계량: -3.4567
p-값: 0.0006
결론: 귀무가설 기각 (두 집단의 평균이 다름)

대응 표본 t-검정 결과:
t-통계량: 4.5678
p-값: 0.0000
결론: 귀무가설 기각 (처리 전후 차이가 있음)
```

위 결과는 다양한 가설 검정의 결과를 보여줍니다:

1. 단일 표본 t-검정: p-값이 0.0198로 유의수준 0.05보다 작으므로 귀무가설을 기각합니다. 즉, 모집단의 평균은 0과 다르다고 할 수 있습니다.

2. 등분산 검정: p-값이 0.2678로 유의수준 0.05보다 크므로 귀무가설을 채택합니다. 즉, 두 집단의 분산은 같다고 할 수 있습니다.

3. 독립 표본 t-검정: p-값이 0.0006으로 유의수준 0.05보다 작으므로 귀무가설을 기각합니다. 즉, 두 집단의 평균은 다르다고 할 수 있습니다.

4. 대응 표본 t-검정: p-값이 0.0000으로 유의수준 0.05보다 작으므로 귀무가설을 기각합니다. 즉, 처리 전후의 평균 차이는 0이 아니라고 할 수 있습니다.

#### 신뢰구간 해석
```
평균의 95% 신뢰구간:
표본 평균: 0.1235
표준오차: 0.0391
t-임계값: 1.9623
오차 한계: 0.0767
95% 신뢰구간: (0.0468, 0.2002)
```

위 결과는 모집단 평균의 95% 신뢰구간을 보여줍니다. 표본 평균은 0.1235이고, 95% 신뢰구간은 (0.0468, 0.2002)입니다. 이는 모집단의 평균이 95%의 확률로 이 구간 내에 있다는 것을 의미합니다. 신뢰구간이 0을 포함하지 않으므로, 모집단의 평균은 0과 다르다고 할 수 있습니다(이는 단일 표본 t-검정의 결과와 일치합니다).

## 3. 상관 분석 및 회귀 분석

### 개념
상관 분석은 두 변수 간의 관계의 강도와 방향을 측정하는 분석 방법입니다. 회귀 분석은 독립변수와 종속변수 간의 관계를 모델링하는 방법으로, 독립변수의 값을 바탕으로 종속변수의 값을 예측할 수 있습니다. 단순 회귀 분석은 하나의 독립변수를 사용하고, 다중 회귀 분석은 여러 개의 독립변수를 사용합니다.

### Python 코드 예시

#### 상관 분석
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 데이터 로드
df = pd.read_csv('sample_data.csv')

# 피어슨 상관계수 계산
correlation_matrix = df.corr(method='pearson')
print("피어슨 상관계수 행렬:")
print(correlation_matrix)

# 스피어만 상관계수 계산 (순위 상관계수)
spearman_corr = df.corr(method='spearman')
print("\n스피어만 상관계수 행렬:")
print(spearman_corr)

# 켄달 타우 상관계수 계산
kendall_corr = df.corr(method='kendall')
print("\n켄달 타우 상관계수 행렬:")
print(kendall_corr)

# 상관계수 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('피어슨 상관계수 히트맵')
plt.show()

# 특정 두 변수 간의 상관관계 시각화
x = 'feature1'
y = 'feature2'
plt.figure(figsize=(10, 6))
sns.scatterplot(x=x, y=y, data=df)
plt.title(f'{x}와 {y}의 산점도')
plt.show()

# 상관계수 검정
r, p_value = stats.pearsonr(df[x].dropna(), df[y].dropna())
print(f"\n{x}와 {y}의 피어슨 상관계수 검정 결과:")
print(f"상관계수(r): {r:.4f}")
print(f"p-값: {p_value:.4f}")
print(f"결론: {'유의한 상관관계가 있음' if p_value < 0.05 else '유의한 상관관계가 없음'}")
```

#### 단순 회귀 분석
```python
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 준비
X = df[['feature1']]
y = df['target']

# statsmodels를 이용한 단순 회귀 분석
X_sm = sm.add_constant(X)  # 상수항 추가
model = sm.OLS(y, X_sm).fit()
print(model.summary())

# 회귀선 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(x='feature1', y='target', data=df)
plt.plot(X, model.predict(X_sm), color='red', linewidth=2)
plt.title('단순 회귀 분석')
plt.xlabel('feature1')
plt.ylabel('target')
plt.show()

# sklearn을 이용한 단순 회귀 분석
model_sk = LinearRegression()
model_sk.fit(X, y)
y_pred = model_sk.predict(X)

# 모델 평가
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("\nsklearn 단순 회귀 분석 결과:")
print(f"계수: {model_sk.coef_[0]:.4f}")
print(f"절편: {model_sk.intercept_:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R-squared: {r2:.4f}")
```

#### 다중 회귀 분석
```python
# 데이터 준비
X_multi = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# statsmodels를 이용한 다중 회귀 분석
X_multi_sm = sm.add_constant(X_multi)  # 상수항 추가
model_multi = sm.OLS(y, X_multi_sm).fit()
print(model_multi.summary())

# sklearn을 이용한 다중 회귀 분석
model_multi_sk = LinearRegression()
model_multi_sk.fit(X_multi, y)
y_pred_multi = model_multi_sk.predict(X_multi)

# 모델 평가
mse_multi = mean_squared_error(y, y_pred_multi)
r2_multi = r2_score(y, y_pred_multi)
print("\nsklearn 다중 회귀 분석 결과:")
print(f"계수: {model_multi_sk.coef_}")
print(f"절편: {model_multi_sk.intercept_:.4f}")
print(f"MSE: {mse_multi:.4f}")
print(f"R-squared: {r2_multi:.4f}")

# 다중공선성 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Variable"] = X_multi.columns
vif_data["VIF"] = [variance_inflation_factor(X_multi.values, i) for i in range(X_multi.shape[1])]
print("\n다중공선성 확인 (VIF):")
print(vif_data)
```

### 출력 결과 해석 예시

#### 상관 분석 결과 해석
```
피어슨 상관계수 행렬:
          feature1  feature2  feature3  target
feature1   1.0000    0.7500    0.2500   0.8000
feature2   0.7500    1.0000    0.3000   0.6500
feature3   0.2500    0.3000    1.0000   0.3500
target     0.8000    0.6500    0.3500   1.0000

feature1와 feature2의 피어슨 상관계수 검정 결과:
상관계수(r): 0.7500
p-값: 0.0000
결론: 유의한 상관관계가 있음
```

위 결과는 변수 간의 상관관계를 보여줍니다:

1. feature1과 target 간의 상관계수는 0.8로, 강한 양의 상관관계가 있습니다.
2. feature2와 target 간의 상관계수는 0.65로, 중간 정도의 양의 상관관계가 있습니다.
3. feature3와 target 간의 상관계수는 0.35로, 약한 양의 상관관계가 있습니다.
4. feature1과 feature2 간의 상관계수는 0.75로, 강한 양의 상관관계가 있으며, 이는 통계적으로 유의합니다(p-값 < 0.05).

#### 단순 회귀 분석 결과 해석
```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 target   R-squared:                       0.640
Model:                            OLS   Adj. R-squared:                  0.639
Method:                 Least Squares   F-statistic:                     354.9
Date:                Wed, 09 Apr 2025   Prob (F-statistic):           1.23e-56
Time:                        02:45:00   Log-Likelihood:                -223.43
No. Observations:                 200   AIC:                             450.9
Df Residuals:                     198   BIC:                             457.7
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.2345      0.123     10.037      0.000       0.992       1.477
feature1       0.8000      0.042     18.839      0.000       0.716       0.884
==============================================================================
Omnibus:                        2.388   Durbin-Watson:                   1.975
Prob(Omnibus):                  0.303   Jarque-Bera (JB):                2.410
Skew:                           0.234   Prob(JB):                        0.300
Kurtosis:                       2.854   Cond. No.                         3.42
==============================================================================

sklearn 단순 회귀 분석 결과:
계수: 0.8000
절편: 1.2345
MSE: 0.7890
R-squared: 0.6400
```

위 결과는 단순 회귀 분석의 결과를 보여줍니다:

1. 회귀 방정식: target = 1.2345 + 0.8000 * feature1
2. R-squared: 0.64로, 모델이 종속변수 분산의 64%를 설명할 수 있음을 의미합니다.
3. F-statistic과 Prob (F-statistic): 모델이 통계적으로 유의함을 나타냅니다(p-값 < 0.05).
4. feature1의 계수는 0.8000으로, feature1이 1단위 증가할 때 target은 평균적으로 0.8000단위 증가합니다.
5. 계수의 p-값(P>|t|)이 0.000으로, feature1이 target에 유의한 영향을 미침을 나타냅니다.
6. MSE(Mean Squared Error): 0.7890으로, 예측값과 실제값 간의 평균 제곱 오차를 나타냅니다.

## 4. 다변량 분석 및 시계열 분석

### 개념
다변량 분석은 여러 변수를 동시에 분석하는 방법으로, 분산분석(ANOVA), 범주형 자료분석 등이 포함됩니다. 시계열 분석은 시간에 따라 수집된 데이터를 분석하는 방법으로, 시간에 따른 패턴, 추세, 계절성 등을 파악할 수 있습니다.

### Python 코드 예시

#### 분산분석(ANOVA)
```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 데이터 로드
df = pd.read_csv('sample_data.csv')

# 일원배치 분산분석 (One-way ANOVA)
# 귀무가설: 모든 그룹의 평균이 같다.
# 대립가설: 적어도 하나의 그룹의 평균이 다르다.
groups = [df[df['category'] == cat]['value'].dropna() for cat in df['category'].unique()]
f_stat, p_value = stats.f_oneway(*groups)

print("일원배치 분산분석 결과:")
print(f"F-통계량: {f_stat:.4f}")
print(f"p-값: {p_value:.4f}")
print(f"결론: {'귀무가설 기각 (그룹 간 평균이 다름)' if p_value < 0.05 else '귀무가설 채택 (그룹 간 평균이 같다고 할 수 있음)'}")

# 그룹별 상자그림
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='value', data=df)
plt.title('카테고리별 값의 분포')
plt.show()

# statsmodels를 이용한 일원배치 분산분석
model = ols('value ~ C(category)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nstatsmodels를 이용한 일원배치 분산분석 결과:")
print(anova_table)

# 사후 검정 (Tukey's HSD)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(endog=df['value'], groups=df['category'], alpha=0.05)
print("\nTukey's HSD 사후 검정 결과:")
print(tukey)

# 이원배치 분산분석 (Two-way ANOVA)
model2 = ols('value ~ C(category) + C(region) + C(category):C(region)', data=df).fit()
anova_table2 = sm.stats.anova_lm(model2, typ=2)
print("\n이원배치 분산분석 결과:")
print(anova_table2)

# 상호작용 효과 시각화
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='value', hue='region', data=df)
plt.title('카테고리와 지역별 값의 분포')
plt.show()
```

#### 범주형 자료분석
```python
# 교차표(Contingency Table) 생성
contingency_table = pd.crosstab(df['category'], df['result'])
print("교차표:")
print(contingency_table)

# 카이제곱 검정
# 귀무가설: 두 변수는 독립적이다.
# 대립가설: 두 변수는 독립적이지 않다.
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("\n카이제곱 검정 결과:")
print(f"카이제곱 통계량: {chi2:.4f}")
print(f"p-값: {p:.4f}")
print(f"자유도: {dof}")
print(f"결론: {'귀무가설 기각 (변수 간 연관성이 있음)' if p < 0.05 else '귀무가설 채택 (변수 간 연관성이 없다고 할 수 있음)'}")

# 기대 빈도
print("\n기대 빈도:")
expected_table = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
print(expected_table)

# 모자이크 플롯
plt.figure(figsize=(10, 6))
from statsmodels.graphics.mosaicplot import mosaic
mosaic(df, ['category', 'result'])
plt.title('카테고리와 결과의 모자이크 플롯')
plt.show()
```

#### 시계열 분석
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# 시계열 데이터 로드
df_time = pd.read_csv('time_series_data.csv')
df_time['date'] = pd.to_datetime(df_time['date'])
df_time.set_index('date', inplace=True)

# 시계열 데이터 시각화
plt.figure(figsize=(12, 6))
plt.plot(df_time.index, df_time['value'])
plt.title('시계열 데이터')
plt.xlabel('날짜')
plt.ylabel('값')
plt.grid(True)
plt.show()

# 시계열 분해 (추세, 계절성, 잔차)
decomposition = seasonal_decompose(df_time['value'], model='additive', period=12)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.show()

# 정상성 검정 (Augmented Dickey-Fuller test)
# 귀무가설: 시계열이 단위근을 가진다 (비정상적이다).
# 대립가설: 시계열이 정상적이다.
result = adfuller(df_time['value'].dropna())
print('ADF 통계량: %f' % result[0])
print('p-값: %f' % result[1])
print('임계값:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
print(f"결론: {'시계열이 정상적임' if result[1] < 0.05 else '시계열이 비정상적임'}")

# 자기상관함수(ACF)와 부분자기상관함수(PACF) 플롯
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df_time['value'].dropna(), ax=ax1)
plot_pacf(df_time['value'].dropna(), ax=ax2)
plt.tight_layout()
plt.show()

# ARIMA 모델 적합
# 비정상 시계열인 경우 차분 적용
if result[1] >= 0.05:
    df_time['value_diff'] = df_time['value'].diff().dropna()
    series = df_time['value_diff']
else:
    series = df_time['value']

# ARIMA 모델 적합 (p, d, q 값은 ACF, PACF 플롯을 통해 결정)
p, d, q = 1, 1, 1  # 예시 값
model = ARIMA(df_time['value'], order=(p, d, q))
model_fit = model.fit()
print("\nARIMA 모델 요약:")
print(model_fit.summary())

# 예측
forecast = model_fit.forecast(steps=12)
forecast_index = pd.date_range(start=df_time.index[-1], periods=13, freq='M')[1:]
forecast_series = pd.Series(forecast, index=forecast_index)

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(df_time.index, df_time['value'], label='관측값')
plt.plot(forecast_series.index, forecast_series, color='red', label='예측값')
plt.title('ARIMA 모델 예측')
plt.xlabel('날짜')
plt.ylabel('값')
plt.legend()
plt.grid(True)
plt.show()
```

### 출력 결과 해석 예시

#### 분산분석 결과 해석
```
일원배치 분산분석 결과:
F-통계량: 15.6789
p-값: 0.0000
결론: 귀무가설 기각 (그룹 간 평균이 다름)

statsmodels를 이용한 일원배치 분산분석 결과:
                 sum_sq    df         F    PR(>F)
C(category)  234.567890   3.0  15.678901  0.000000
Residual     987.654321  196.0       NaN       NaN

Tukey's HSD 사후 검정 결과:
   Multiple Comparison of Means - Tukey HSD, FWER=0.05   
=======================================================
group1 group2  meandiff  lower   upper  reject
-------------------------------------------------------
A      B       -1.2345  -2.3456 -0.1234   True
A      C       -2.3456  -3.4567 -1.2345   True
A      D       -3.4567  -4.5678 -2.3456   True
B      C       -1.1111  -2.2222 -0.0000   True
B      D       -2.2222  -3.3333 -1.1111   True
C      D       -1.1111  -2.2222 -0.0000   True
-------------------------------------------------------
```

위 결과는 일원배치 분산분석과 사후 검정의 결과를 보여줍니다:

1. 일원배치 분산분석: F-통계량은 15.6789이고 p-값은 0.0000으로, 유의수준 0.05보다 작습니다. 따라서 귀무가설을 기각하고, 적어도 하나의 그룹의 평균이 다르다고 할 수 있습니다.

2. Tukey's HSD 사후 검정: 모든 그룹 쌍의 평균 차이가 통계적으로 유의합니다(reject = True). 예를 들어, 그룹 A와 B의 평균 차이는 -1.2345이고, 95% 신뢰구간은 (-2.3456, -0.1234)입니다. 이 구간이 0을 포함하지 않으므로, 두 그룹의 평균은 통계적으로 유의하게 다릅니다.

#### 범주형 자료분석 결과 해석
```
교차표:
result    성공  실패
category          
A         30   10
B         25   15
C         20   20
D         15   25

카이제곱 검정 결과:
카이제곱 통계량: 12.3456
p-값: 0.0063
자유도: 3
결론: 귀무가설 기각 (변수 간 연관성이 있음)

기대 빈도:
result        성공        실패
category                    
A       22.500000  17.500000
B       22.500000  17.500000
C       22.500000  17.500000
D       22.500000  17.500000
```

위 결과는 카이제곱 검정의 결과를 보여줍니다:

1. 교차표: 각 카테고리별 성공과 실패의 빈도를 보여줍니다. 예를 들어, 카테고리 A는 성공 30건, 실패 10건입니다.

2. 카이제곱 검정: 카이제곱 통계량은 12.3456이고 p-값은 0.0063으로, 유의수준 0.05보다 작습니다. 따라서 귀무가설을 기각하고, 카테고리와 결과 간에 연관성이 있다고 할 수 있습니다.

3. 기대 빈도: 두 변수가 독립적이라면 기대되는 빈도를 보여줍니다. 실제 빈도와 기대 빈도의 차이가 클수록 두 변수 간의 연관성이 강합니다.

#### 시계열 분석 결과 해석
```
ADF 통계량: -3.456789
p-값: 0.009876
임계값:
	1%: -3.456
	5%: -2.873
	10%: -2.573
결론: 시계열이 정상적임

ARIMA 모델 요약:
                               ARIMA Model Results                              
==============================================================================
Dep. Variable:                  value   No. Observations:                  120
Model:                 ARIMA(1, 1, 1)   Log Likelihood                -234.567
Date:                Wed, 09 Apr 2025   AIC                            475.134
Time:                        02:45:00   BIC                            483.456
Sample:                    01-01-2015   HQIC                           478.789
                         - 12-01-2024                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.7500      0.123      6.098      0.000       0.509       0.991
ma.L1          0.5000      0.150      3.333      0.001       0.206       0.794
sigma2         1.2345      0.234      5.275      0.000       0.776       1.693
==============================================================================
```

위 결과는 시계열 분석의 결과를 보여줍니다:

1. ADF 검정: ADF 통계량은 -3.456789이고 p-값은 0.009876으로, 유의수준 0.05보다 작습니다. 따라서 귀무가설을 기각하고, 시계열이 정상적이라고 할 수 있습니다.

2. ARIMA 모델: ARIMA(1,1,1) 모델이 적합되었습니다. AR(1) 계수는 0.75, MA(1) 계수는 0.5이며, 둘 다 통계적으로 유의합니다(p-값 < 0.05). AIC(Akaike Information Criterion)는 475.134로, 모델의 적합도를 나타냅니다. 일반적으로 AIC가 낮을수록 더 좋은 모델입니다.
