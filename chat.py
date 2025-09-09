def check_missing_months(df, date_col='date', client_col='client_id'):
    """
    Проверяет наличие пропущенных месяцев для каждого клиента
    """
    # Убедимся, что дата в datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Получаем уникальные клиенты и даты
    clients = df[client_col].unique()
    all_dates = pd.date_range(df[date_col].min(), df[date_col].max(), freq='MS')
    
    missing_info = {}
    
    for client in clients:
        client_data = df[df[client_col] == client]
        client_dates = client_data[date_col].unique()
        
        # Находим пропущенные даты
        missing_dates = set(all_dates) - set(client_dates)
        
        if missing_dates:
            missing_info[client] = {
                'total_months': len(all_dates),
                'present_months': len(client_dates),
                'missing_months': len(missing_dates),
                'missing_dates_list': sorted(missing_dates)
            }
    
    return missing_info

# Проверяем пропуски
missing_months = check_missing_months(df)
print(f"Клиенты с пропущенными месяцами: {len(missing_months)}")
for client, info in missing_months.items():
    print(f"Клиент {client}: пропущено {info['missing_months']} месяцев")
    print(f"  Пропущенные даты: {info['missing_dates_list']}")
