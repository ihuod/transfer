from sklearn.metrics import ConfusionMatrixDisplay

# Предсказанные вероятности
y_pred_proba = catboost_model.predict_proba(X_test)[:, 1]

# Confusion matrix для разных threshold
thresholds = [0.3, 0.5, 0.7]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, threshold in enumerate(thresholds):
    y_pred_threshold = (y_pred_proba > threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_threshold)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                 display_labels=['Не активен', 'Активен'])
    disp.plot(ax=axes[i], cmap='Blues')
    axes[i].set_title(f'Threshold = {threshold}')
    
plt.tight_layout()
plt.show()
