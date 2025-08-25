import numpy as np
import pandas as pd
import optuna

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

# X_train, y_train, X_test, y_test уже заданы
is_df = isinstance(X_train, pd.DataFrame)
feat_names = list(X_train.columns) if is_df else [f'x{i}' for i in range(X_train.shape[1])]

def get_fold(X, idx):
    return (X.iloc[idx] if is_df else X[idx])

tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    # гиперпараметры
    l1_ratio = trial.suggest_float('l1_ratio', 0.05, 0.95)
    alpha    = trial.suggest_float('alpha', 1e-4, 1e2, log=True)
    tol      = trial.suggest_float('tol', 1e-4, 1e-2, log=True)
    scaler_name = trial.suggest_categorical('scaler', ['robust', 'standard', 'maxabs'])
    scaler = {'robust': RobustScaler(with_centering=True, with_scaling=True),
              'standard': StandardScaler(with_mean=True, with_std=True),
              'maxabs': MaxAbsScaler()}[scaler_name]

    pipe = make_pipeline(
        scaler,
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, tol=tol,
                   random_state=42, fit_intercept=True)
    )

    fold_losses = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
        Xtr, Xva = get_fold(X_train, tr_idx), get_fold(X_train, va_idx)
        ytr, yva = y_train[tr_idx], y_train[va_idx]
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xva)
        loss = mean_absolute_error(yva, pred)  # цель: минимизировать MAE
        fold_losses.append(loss)
        trial.report(loss, step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_losses))

study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=50, n_jobs=1)  # увеличьте n_trials при необходимости
best = study.best_params

# Финальная модель на всём train
scaler = {'robust': RobustScaler(with_centering=True, with_scaling=True),
          'standard': StandardScaler(with_mean=True, with_std=True),
          'maxabs': MaxAbsScaler()}[best['scaler']]

final_model = make_pipeline(
    scaler,
    ElasticNet(alpha=best['alpha'], l1_ratio=best['l1_ratio'],
               max_iter=20000, tol=best['tol'], random_state=42, fit_intercept=True)
).fit(X_train, y_train)

y_pred = final_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
spearman = spearmanr(y_test, y_pred).correlation
print({'mae': mae, 'r2': r2, 'spearman': spearman, **best})

# Извлечение “сырых” коэффициентов (обратное к скейлингу)
enet = final_model.named_steps['elasticnet']
if best['scaler'] == 'standard':
    sc = final_model.named_steps['standardscaler']
    center = pd.Series(sc.mean_, index=feat_names)
    scale  = pd.Series(sc.scale_, index=feat_names).replace(0, 1.0)
    betas_scaled = pd.Series(enet.coef_, index=feat_names)
    betas_raw = betas_scaled / scale
    intercept_raw = float(enet.intercept_ - (betas_scaled * (center / scale)).sum())
elif best['scaler'] == 'robust':
    sc = final_model.named_steps['robustscaler']
    center = pd.Series(getattr(sc, 'center_', np.zeros(len(feat_names))), index=feat_names)
    scale  = pd.Series(getattr(sc, 'scale_',  np.ones(len(feat_names))), index=feat_names).replace(0, 1.0)
    betas_scaled = pd.Series(enet.coef_, index=feat_names)
    betas_raw = betas_scaled / scale
    intercept_raw = float(enet.intercept_ - (betas_scaled * (center / scale)).sum())
else:  # maxabs
    sc = final_model.named_steps['maxabsscaler']
    scale  = pd.Series(sc.max_abs_, index=feat_names).replace(0, 1.0)
    betas_scaled = pd.Series(enet.coef_, index=feat_names)
    betas_raw = betas_scaled / scale
    intercept_raw = float(enet.intercept_)

coef_df = pd.DataFrame({'feature': feat_names, 'beta_raw': betas_raw}).sort_values('beta_raw', ascending=False)
