def manual_baseline_auth_times(df, auth_thr, auth_times_col='AUTH_TIMES_x', target_col='TARGET_NEXT_MONTH'):
    """
    Мануальный бейзлайн: предсказываем 1 если AUTH_TIMES_x > auth_thr, иначе 0
    """
    # Создаем предсказания
    y_pred_manual = (df[auth_times_col] > auth_thr).astype(int).values
    y_true = df[target_col].values
    
    # Рассчитываем метрики
    metrics = calculate_binary_metrics(
        y_true, y_pred_manual, 
        model_name=f"Manual Baseline (AUTH_TIMES > {auth_thr})"
    )
    
    return y_pred_manual, metrics

# Подбираем оптимальный порог
def find_optimal_auth_threshold(df, auth_times_col='AUTH_TIMES_x', target_col='TARGET_NEXT_MONTH'):
    """
    Подбирает оптимальный порог для AUTH_TIMES_x
    """
    results = []
    
    # Перебираем разные пороги
    for threshold in range(0, int(df[auth_times_col].max()) + 1):
        y_pred = (df[auth_times_col] > threshold).astype(int)
        
        # Базовые метрики
        accuracy = accuracy_score(df[target_col], y_pred)
        precision = precision_score(df[target_col], y_pred, zero_division=0)
        recall = recall_score(df[target_col], y_pred, zero_division=0)
        f1 = f1_score(df[target_col], y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
    
    # Создаем DataFrame
    results_df = pd.DataFrame(results)
    
    # Находим лучший порог по F1-score
    best_idx = results_df['f1_score'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_f1 = results_df.loc[best_idx, 'f1_score']
    
    return results_df, best_threshold, best_f1

# Визуализация подбора порога
def plot_threshold_analysis(results_df, best_threshold):
    """
    Визуализирует метрики для разных порогов
    """
    plt.figure(figsize=(12, 8))
    
    plt.plot(results_df['threshold'], results_df['accuracy'], label='Accuracy', linewidth=2)
    plt.plot(results_df['threshold'], results_df['precision'], label='Precision', linewidth=2)
    plt.plot(results_df['threshold'], results_df['recall'], label='Recall', linewidth=2)
    plt.plot(results_df['threshold'], results_df['f1_score'], label='F1-Score', linewidth=3)
    
    # Отмечаем лучший порог
    plt.axvline(x=best_threshold, color='red', linestyle='--', 
                label=f'Best threshold: {best_threshold}')
    
    plt.xlabel('Порог AUTH_TIMES_x')
    plt.ylabel('Метрика')
    plt.title('Подбор оптимального порога для мануального бейзлайна')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
