import numpy as np
import pandas as pd
from scipy.stats import spearmanr

df = df_test_pred.copy()
y = df['PROF_TRANSF_COMM'].astype(float)
y_pred = df['PROF_TRANSF_COMM_PRED'].astype(float)

n = len(df)
k_top = int(np.ceil(0.05 * n))
k_mid = int(np.ceil(0.15 * n))

# 1) Группы по предсказанию (через ранжирование)
order = np.argsort(-y_pred.values)
pred_group = pd.Series('low', index=df.index)
pred_group.iloc[order[:k_top]] = 'high'
pred_group.iloc[order[k_top:k_top + k_mid]] = 'mid'
df['pred_group'] = pred_group

# 2) Группы по факту (терцили 80/15/5)
df['actual_group'] = pd.qcut(y, q=[0, 0.80, 0.95, 1.0], labels=['low','mid','high'])

# 3) Глобальная ранговая корреляция
spearman = spearmanr(y, y_pred).correlation

# 4) Сводка по группам предсказания
grp = (df.groupby('pred_group')
         .agg(n=('PROF_TRANSF_COMM','size'),
              y_mean=('PROF_TRANSF_COMM','mean'),
              y_median=('PROF_TRANSF_COMM','median'),
              y_sum=('PROF_TRANSF_COMM','sum'),
              ypred_mean=('PROF_TRANSF_COMM_PRED','mean'))
         .reindex(['low','mid','high']))

# 5) Матрица соответствия (как распределились фактические группы внутри предсказанных)
conf = pd.crosstab(df['pred_group'], df['actual_group'], normalize='index').reindex(index=['low','mid','high'], columns=['low','mid','high'])

# 6) Метрики для топ-5%
is_actual_high = (df['actual_group'] == 'high').astype(int)
is_pred_high = (df['pred_group'] == 'high').astype(int)

precision_top5 = (is_actual_high & is_pred_high).sum() / is_pred_high.sum()
recall_top5 = (is_actual_high & is_pred_high).sum() / is_actual_high.sum()
baseline_rate = is_actual_high.mean()  # ≈ 0.05
lift_top5 = precision_top5 / baseline_rate if baseline_rate > 0 else np.nan

# 7) Capture: доля суммы факта в топ-5% по предсказанию
capture_sum_top5 = y[is_pred_high == 1].sum() / y.sum() if y.sum() != 0 else np.nan

report = {
    'spearman': spearman,
    'precision@5%': precision_top5,
    'recall@5%': recall_top5,
    'lift@5%': lift_top5,
    'capture_sum@5%': capture_sum_top5
}

# Для прод-использования: пороги по предсказанию из трейна
# cuts = np.quantile(df_train_pred['PROF_TRANSF_COMM_PRED'], [0.80, 0.95])
# df['pred_group_prod'] = pd.cut(y_pred, bins=[-np.inf, cuts[0], cuts[1], np.inf], labels=['low','mid','high'], right=False)

grp, conf, report
