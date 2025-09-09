def check_missing_months_crosstab(df, date_col='date', client_col='client_id'):
    """
    Самая быстрая версия с использованием crosstab
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Создаем кросс-таблицу наличия данных
    presence_table = pd.crosstab(
        index=df[client_col],
        columns=df[date_col],
        dropna=False
    )
    
    # Все возможные даты
    all_dates = pd.date_range(df[date_col].min(), df[date_col].max(), freq='MS')
    
    # Добавляем отсутствующие колонки
    for date in all_dates:
        if date not in presence_table.columns:
            presence_table[date] = 0
    
    # Сортируем колонки по дате
    presence_table = presence_table.reindex(columns=sorted(presence_table.columns))
    
    # Анализируем пропуски
    missing_info = {}
    
    for client in presence_table.index:
        missing_mask = presence_table.loc[client] == 0
        missing_dates = missing_mask[missing_mask].index.tolist()
        
        if missing_dates:
            missing_info[client] = {
                'total_months': len(all_dates),
                'present_months': len(presence_table.columns) - len(missing_dates),
                'missing_months': len(missing_dates),
                'missing_dates_list': missing_dates
            }
    
    return missing_info, presence_table
