import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, average_precision_score,
                             log_loss)

def calculate_binary_metrics(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Рассчитывает все основные метрики бинарной классификации
    
    Parameters:
    y_true : array-like, true labels
    y_pred : array-like, predicted labels
    y_pred_proba : array-like, predicted probabilities (optional)
    model_name : str, name of the model for display
    
    Returns:
    dict: Dictionary with all metrics
    """
    
    # Базовые метрики
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Дополнительные метрики
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    
    # Метрики, требующие вероятности
    metrics_with_proba = {}
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            pr_auc = average_precision_score(y_true, y_pred_proba)
            logloss = log_loss(y_true, y_pred_proba)
            
            metrics_with_proba = {
                'ROC-AUC': roc_auc,
                'PR-AUC': pr_auc,
                'Log-Loss': logloss
            }
        except:
            print("Предупреждение: Не удалось рассчитать метрики, требующие вероятностей")
    
    # Рассчитываем rates
    total = len(y_true)
    positive_rate = np.mean(y_true)
    predicted_positive_rate = np.mean(y_pred)
    
    # Собираем все метрики в словарь
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'F1-Score': f1,
        'False Positive Rate': fpr,
        'False Negative Rate': fnr,
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'Total Samples': total,
        'Positive Rate (Actual)': positive_rate,
        'Positive Rate (Predicted)': predicted_positive_rate
    }
    
    # Добавляем метрики с вероятностями если они есть
    metrics.update(metrics_with_proba)
    
    # Красиво выводим результаты
    print(f"📊 МЕТРИКИ БИНАРНОЙ КЛАССИФИКАЦИИ - {model_name}")
    print("=" * 60)
    
    # Основные метрики
    print("\n🎯 ОСНОВНЫЕ МЕТРИКИ:")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1-Score:          {f1:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    
    if y_pred_proba is not None and 'ROC-AUC' in metrics:
        print(f"ROC-AUC:           {metrics['ROC-AUC']:.4f}")
        print(f"PR-AUC:            {metrics['PR-AUC']:.4f}")
        print(f"Log-Loss:          {metrics['Log-Loss']:.4f}")
    
    # Confusion Matrix
    print(f"\n📋 CONFUSION MATRIX:")
    print(f"                Predicted 0   Predicted 1")
    print(f"Actual 0         {tn:8d}       {fp:8d}")
    print(f"Actual 1         {fn:8d}       {tp:8d}")
    
    # Rates
    print(f"\n📈 СТАТИСТИКА:")
    print(f"True Positives:  {tp}")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp} (Type I Error)")
    print(f"False Negatives: {fn} (Type II Error)")
    print(f"Total Samples:   {total}")
    print(f"Actual Positive Rate:   {positive_rate:.3f}")
    print(f"Predicted Positive Rate: {predicted_positive_rate:.3f}")
    
    # Дополнительные расчеты
    print(f"\n🔍 ДОПОЛНИТЕЛЬНО:")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")
    
    return metrics

# Дополнительная функция для сравнения нескольких моделей
def compare_models_metrics(models_metrics, metric_names=None):
    """
    Сравнивает метрики нескольких моделей
    """
    if metric_names is None:
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    comparison_data = {}
    for model_name, metrics in models_metrics.items():
        comparison_data[model_name] = {metric: metrics.get(metric, np.nan) for metric in metric_names}
    
    comparison_df = pd.DataFrame(comparison_data).T
    print("📈 СРАВНЕНИЕ МОДЕЛЕЙ:")
    print(comparison_df.round(4))
    
    return comparison_df
