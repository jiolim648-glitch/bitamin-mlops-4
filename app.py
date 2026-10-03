import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터 로드
df = pd.read_csv("WA_FnUseC_TelcoCustomerChurn.csv")

# 2. 학습에 쓰지 않을 컬럼 제거
df = df.drop(columns=["customerID"])

# 3. TotalCharges 숫자형 변환 및 결측치 처리
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# 4. 타깃 변수 Churn을 0/1로 변환
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 5. 범주형 입력 변수 One-Hot Encoding
X = df.drop(columns=["Churn"])
y = df["Churn"]

X = pd.get_dummies(X, drop_first=True)

# 6. 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. 수치형 변수 스케일링
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 8. 모델 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Random Forest 모델 추가
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 9. 평가 (두 모델 비교)
for name, m in [("Logistic Regression", model), ("Random Forest", rf_model)]:
    pred = m.predict(X_test)
    print(f"{name} accuracy: {accuracy_score(y_test, pred):.4f}")