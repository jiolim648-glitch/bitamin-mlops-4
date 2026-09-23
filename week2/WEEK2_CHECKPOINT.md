
## 1. 조별 레포 개설 및 1주차 결과물 push

<img width="911" height="605" alt="스크린샷 2026-09-23 오후 8 55 23" src="https://github.com/user-attachments/assets/85c6d064-be2a-4fad-9a2c-7bafe9c4df90" />


## 2. 역할별 코드 수정

각자의 브랜치에서 동일한 `app.py`를 수정했다. 브랜치는 서로 독립적이므로 이 단계에서는 다른 조원의 변경사항이 내 코드에 보이지 않는다.

내가 수정한 내용은 다음과 같다.

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest 모델 추가
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 평가 (두 모델 비교)
for name, m in [("Logistic Regression", model), ("Random Forest", rf_model)]:
    pred = m.predict(X_test)
    print(f"{name} accuracy: {accuracy_score(y_test, pred):.4f}")
```

---

## 3. 변경사항 Commit & Push

`git status`로 `app.py`가 modified 상태인지, `git diff`로 수정한 코드가 의도한 내용인지 확인한 뒤 commit하고 push했다.

```bash
git status
git diff
git add app.py
git commit -m "feat: add random forest model"
git log --oneline -3
git push
```

---

## 4. Pull Request 생성

`feature/random-forest` → `main`으로 PR #1을 생성했다. PR 본문에 변경사항(Random Forest 모델 추가, Logistic Regression과 비교 가능하도록 구성)을 작성했다.

![PR #1 생성](images/02_pr_open.png)

---

## 5. Merge Conflict 해결

내 PR보다 A 담당의 `feature/preprocessing`(PR #2)이 먼저 `main`에 merge되었다. 두 브랜치 모두 `app.py`의 모델 학습/평가 부분을 수정했기 때문에 `origin/main`을 내 브랜치에 합치는 과정에서 conflict가 발생했다.

| 구분 | 변경 내용 |
|---|---|
| `HEAD` (내 브랜치) | Random Forest 모델 추가, 두 모델 accuracy 비교 |
| `origin/main` (A 담당) | 수치형 변수(`tenure`, `MonthlyCharges`, `TotalCharges`) StandardScaler 스케일링 |

두 변경은 서로 다른 기능이므로 한쪽을 버리지 않고 **스케일링 → 모델 학습(LR, RF) → 비교 평가** 순서로 합쳐 conflict marker(`<<<<<<<`, `=======`, `>>>>>>>`)를 제거했다.

```bash
git add app.py
git commit -m "fix: resolve merge conflict"
git push
```

---

## 6. Pull Request Merge

A 담당의 PR #2와 내 PR #1이 모두 `main`에 merge되었다.

![Closed PR 목록](images/03_pr_list_closed.png)

**PR #2 — `feature/preprocessing` → `main` (A 담당)**

![PR #2 merged](images/04_pr2_preprocessing_merged.png)

**PR #1 — `feature/random-forest` → `main` (내 작업)**

![PR #1 merged](images/05_pr1_random_forest_merged.png)

---

## 7. main 반영 확인

`main` 브랜치의 최신 커밋이 `Merge pull request #1`로 바뀌었고, `app.py`에 내 브랜치의 변경사항이 반영된 것을 확인했다.

![main 반영 확인](images/06_main_after_merge.png)

---

## Week2 심화 체크포인트 (선택)

| 번호 | 심화 체크포인트 | 통과 확인 | 진행 여부 |
|:---:|---|---|:---:|
| 1 | Issue 생성 및 Commit 연결 | Issue Timeline에 관련 Commit 표시 | ☐ |
| 2 | PR Template 작성 | 새 PR 생성 시 Template 자동 표시 | ☐ |
| 3 | Commit Message Convention 적용 | Commit History + CONTRIBUTING.md 확인 | ☐ |
| 4 | Branch Protection 설정 | main Ruleset 활성화 및 승인 전 Merge 제한 확인 | ☐ |
