# time-based split
n_val = int(0.1 * len(y_train))
X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]
X_val, y_val = X_train[-n_val:], y_train[-n_val:]

params = {
    "objective": "reg:pseudohubererror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 100,     # large data → higher leaf min weight
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "gamma": 1.0,                # require gain to split
    "reg_lambda": 10.0,
    "reg_alpha": 0.0,
    "nthread": -1,
    "random_state": 42
}

dtr = xgb.DMatrix(X_tr, label=y_tr)
dva = xgb.DMatrix(X_val, label=y_val)
dte = xgb.DMatrix(X_test)

bst = xgb.train(
    params, dtr,
    num_boost_round=20000,
    evals=[(dva, "val")],
    early_stopping_rounds=300,
    verbose_eval=200
)

pred = bst.predict(dte, ntree_limit=bst.best_ntree_limit)
print({"val_mae": float(mean_absolute_error(y_val, bst.predict(dva))),
       "test_mae": float(mean_absolute_error(y_test, pred))})
