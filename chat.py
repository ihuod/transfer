def move_columns_to_end(df, columns_to_move):
    """
    Перемещает указанные колонки в конец датафрейма
    """
    other_columns = [col for col in df.columns if col not in columns_to_move]
    new_order = other_columns + columns_to_move
    return df[new_order]

# Используем функцию
merged_df = move_columns_to_end(merged_df, ['IS_ACTIVE_SBOL', 'TARGET_NEXT_MONTH'])
