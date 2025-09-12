# Подбираем оптимальный порог
results_df, best_threshold, best_f1 = find_optimal_auth_threshold(df_train)

print(f"🎯 Оптимальный порог: {best_threshold}")
print(f"🏆 Лучший F1-Score: {best_f1:.4f}")

# Визуализируем
plot_threshold_analysis(results_df, best_threshold)

# Посмотрим топ-5 порогов
print("\n📊 Топ-5 порогов по F1-Score:")
print(results_df.nlargest(5, 'f1_score').round(4))



# На тренировочных данных
y_pred_train, metrics_train = manual_baseline_auth_times(
    df_train, best_threshold, 'AUTH_TIMES_x', 'TARGET_NEXT_MONTH'
)

# На тестовых данных (OOT)
y_pred_test, metrics_test = manual_baseline_auth_times(
    df_test, best_threshold, 'AUTH_TIMES_x', 'TARGET_NEXT_MONTH'
)

print("=" * 60)
print("📈 СРАВНЕНИЕ ТРЕНИРОВОЧНЫХ И ТЕСТОВЫХ ДАННЫХ:")
print(f"Train F1: {metrics_train['F1-Score']:.4f}")
print(f"Test F1:  {metrics_test['F1-Score']:.4f}")
