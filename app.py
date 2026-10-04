import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1. 데이터 로드
df = pd.read_csv("WA_FnUseC_TelcoCustomerChurn.csv")

# 2. 학습에 쓰지 않을 컬럼 제거
df = df.drop(columns=["customerID"])

# 3. TotalCharges 숫자형 변환
# 숫자로 변환할 수 없는 값은 NaN으로 처리
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 4. 타깃 변수 Churn을 0/1로 변환
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 5. 입력 변수와 타깃 변수 분리
X = df.drop(columns=["Churn"])
y = df["Churn"]

# 6. 수치형 / 범주형 컬럼 분리
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# 7. 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. 수치형 전처리 Pipeline
# 결측치는 중앙값으로 대체하고 StandardScaler 적용
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 9. 범주형 전처리 Pipeline
# 결측치는 최빈값으로 대체하고 One-Hot Encoding 적용
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

# 10. 수치형 / 범주형 전처리 통합
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# 11. Logistic Regression Pipeline
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# 12. Random Forest Pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# 13. 모델 학습
logistic_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# 14. 평가
for name, model in [
    ("Logistic Regression", logistic_pipeline),
    ("Random Forest", rf_pipeline)
]:
    pred = model.predict(X_test)
    print(f"{name} accuracy: {accuracy_score(y_test, pred):.4f}")