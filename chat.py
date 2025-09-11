from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

# CatBoost сам умеет работать с категориальными features!
cat_features = ['ACTIVITY_GROUP']  # указываем категориальные колонки

catboost_model = CatBoostClassifier(
    random_state=42,
    cat_features=cat_features,
    verbose=100,  # выводим прогресс обучения каждые 100 итераций
    early_stopping_rounds=50,
    class_weights=[1, np.sum(y_train == 0) / np.sum(y_train == 1)]  # баланс классов
)

# Обучаем модель
catboost_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)

print("✅ CatBoost модель обучена")
