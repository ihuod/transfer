# pip install catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1) Категориальные признаки (по pandas-дататипам)
cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) in ("object", "category")]
cat_features = [X_train.columns.get_loc(c) for c in cat_cols]

# 2) Пулы данных
train_pool = Pool(X_train, y_train, cat_features=cat_features)
valid_pool = Pool(X_test,  y_test,  cat_features=cat_features)

# 3) Модель
reg = CatBoostRegressor(
    loss_function="RMSE",      # варианты: "MAE", "Quantile", "Poisson" (для неотрицательных счётчиков)
    iterations=20000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    subsample=0.8,
    rsm=0.8,
    random_seed=42,
    eval_metric="RMSE",
    verbose=200
)

# 4) Обучение с early stopping
reg.fit(
    train_pool,
    eval_set=valid_pool,
    early_stopping_rounds=500,
    use_best_model=True
)

# 5) Оценка
y_pred = reg.predict(valid_pool)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  best_iter={reg.get_best_iteration()}")

# 6) Важность признаков
imp = pd.Series(
    reg.get_feature_importance(train_pool, type="PredictionValuesChange"),
    index=X_train.columns
).sort_values(ascending=False)
print(imp.head(30))
