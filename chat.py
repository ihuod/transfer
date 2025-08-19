# pip install lightgbm
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1) Категориальные признаки (если есть)
cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) in ("category", "object")]
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_test[c]  = X_test[c].astype("category")

# 2) Модель
reg = lgb.LGBMRegressor(
    objective="regression",     # варианты: "regression_l1", "huber", "fair", "poisson" (для неотрицательных счётчиков)
    n_estimators=10000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.2,
    random_state=42,
    n_jobs=-1
)

# 3) Обучение с ранней остановкой
reg.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    early_stopping_rounds=200,
    categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
    verbose=200
)

# 4) Оценка
y_pred = reg.predict(X_test, num_iteration=reg.best_iteration_)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  best_iter={reg.best_iteration_}")

# 5) Топ-важности (gain)
import matplotlib.pyplot as plt
importances = pd.Series(
    reg.booster_.feature_importance(importance_type="gain", iteration=reg.best_iteration_),
    index=reg.booster_.feature_name()
).sort_values(ascending=False).head(30)
importances.plot(kind="barh", figsize=(8, 10)); plt.gca().invert_yaxis(); plt.title("Top-30 Feature Importance (gain)")
plt.show()
