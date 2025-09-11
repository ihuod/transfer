# Предсказания
y_pred = catboost_model.predict(X_test)
y_pred_proba = catboost_model.predict_proba(X_test)[:, 1]

print("📊 МЕТРИКИ CatBoost:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Feature importance
feature_importance = catboost_model.get_feature_importance()
feature_names = X_train.columns

print("\n📊 ТОП-20 ВАЖНЫХ ПРИЗНАКОВ:")
for i in np.argsort(feature_importance)[-20:][::-1]:
    print(f"{feature_names[i]}: {feature_importance[i]:.4f}")
