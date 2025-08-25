feat_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else [f'x{i}' for i in range(X_train.shape[1])]
scaler = model.named_steps['robustscaler']
enet   = model.named_steps['elasticnetcv']

center = pd.Series(getattr(scaler, 'center_', np.zeros(len(feat_names))), index=feat_names)
scale  = pd.Series(getattr(scaler, 'scale_',  np.ones(len(feat_names))),  index=feat_names).replace(0, 1.0)

betas_scaled = pd.Series(enet.coef_, index=feat_names)
betas_raw = betas_scaled / scale
intercept_raw = float(enet.intercept_ - (betas_scaled * (center / scale)).sum())

coef_df = pd.DataFrame({'feature': feat_names, 'beta_raw': betas_raw}).sort_values('beta_raw', ascending=False)
