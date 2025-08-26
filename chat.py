import xgboost as xgb
from sklearn.metrics import mean_squared_error

xgb_reg = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=4000,
    learning_rate=0.05,
    max_depth=8,            # keep moderate; high depth + sparse often overfits
    min_child_weight=1.0,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_lambda=1.0,
    reg_alpha=0.0,
    tree_method="hist",     # use "gpu_hist" if you have a GPU
    n_jobs=-1,
    random_state=42
)

xgb_reg.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    early_stopping_rounds=200,
    verbose=100
)

y_pred = xgb_reg.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print({"rmse": rmse})
