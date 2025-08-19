import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# Пример данных (замените на ваш DataFrame)
df = pd.DataFrame({
    'col1': [1.2, 2.5, 3.1, 4.0, 5.2, 6.7, 7.1, 8.0, 9.5, 10.1],
    'col2': [1.0, 2.1, 3.0, 3.9, 5.0, 6.5, 7.0, 8.5, 9.0, 11.2]
})

# 1. Визуализация распределений
plt.figure(figsize=(10, 6))

# Гистограммы + KDE
sns.histplot(df['col1'], kde=True, color="blue", label="Col1", alpha=0.5, bins=5)
sns.histplot(df['col2'], kde=True, color="red", label="Col2", alpha=0.5, bins=5)
plt.title("Гистограммы и KDE распределений")
plt.xlabel("Значения")
plt.ylabel("Частота")
plt.legend()
plt.show()

# Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[['col1', 'col2']], palette=["blue", "red"])
plt.title("Boxplot столбцов")
plt.ylabel("Значения")
plt.show()

# 2. K-S тест
stat, p_value = ks_2samp(df['col1'], df['col2'])
print(f"K-S test: p-value = {p_value:.4f}")
if p_value < 0.05:
    print("Распределения различаются (отвергаем H₀)")
else:
    print("Нет доказательств различия распределений (не отвергаем H₀)")
