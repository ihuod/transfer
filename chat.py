from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Определяем категориальные и числовые колонки
categorical_features = ['ACTIVITY_GROUP']
numerical_features = [col for col in X_train.columns if col != 'ACTIVITY_GROUP']

# Создаем ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# Создаем pipeline
baseline_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    ))
])

# Обучаем модель
baseline_model.fit(X_train, y_train)
print("✅ Модель с One-Hot Encoding обучена")
