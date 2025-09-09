def create_target_variable(df, activity_col='IS_ACTIVE_SBOL', date_col='date', client_col='client_id'):
    """
    Создает целевую переменную - активность в следующем месяце
    """
    df = df.copy()
    
    # Убеждаемся, что дата в правильном формате
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Сортируем по клиенту и дате (очень важно!)
    df = df.sort_values([client_col, date_col])
    
    # Создаем целевую переменную - активность в следующем месяце
    df['TARGET_NEXT_MONTH'] = df.groupby(client_col)[activity_col].shift(-1)
    
    # Для последнего месяца каждого клиента будет NaN - это нормально
    print(f"Создана целевая переменная. Пропусков: {df['TARGET_NEXT_MONTH'].isna().sum()}")
    print(f"Всего записей: {len(df)}")
    
    return df

# Применяем функцию
df_with_target = create_target_variable(df)
