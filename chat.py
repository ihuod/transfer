import pandas as pd
import numpy as np

def select_many_small_spenders(
    df: pd.DataFrame,
    min_nonzero_merchants: int = 10,
    small_tx_threshold: float = 100.0,
    small_tx_share: float = 0.80,
    merchant_cols: list[str] | None = None,
    id_col: str = "id",
    date_col: str = "date",
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Отбирает строки (клиент-месяц), где:
      1) число мерчей с ненулевыми тратами >= min_nonzero_merchants
      2) доля "малых" трат (<= small_tx_threshold) по сумме >= small_tx_share

    Параметры:
      df: входной DataFrame
      min_nonzero_merchants: минимум мерчей с ненулевыми тратами
      small_tx_threshold: порог "малой" траты
      small_tx_share: минимальная доля суммы "малых" трат (0..1)
      merchant_cols: явный список колонок мерчей; если None — определить автоматически
      id_col: имя столбца с id клиента
      date_col: имя столбца с датой
      exclude_cols: дополнительные столбцы, исключаемые из набора мерчей при автоопределении

    Возвращает:
      Отфильтрованный DataFrame (те же столбцы, что и во входном df).
    """
    if df.empty:
        return df.copy()

    # Определяем столбцы мерчей, если не заданы
    if merchant_cols is None:
        default_exclude = {id_col, date_col, "profit", "profit_pred"}
        if exclude_cols:
            default_exclude |= set(exclude_cols)
        # Берём числовые колонки, исключая служебные
        num_cols = df.select_dtypes(include=[np.number]).columns
        merchant_cols = [c for c in num_cols if c not in default_exclude]
        if len(merchant_cols) == 0:
            raise ValueError("Не удалось определить колонки мерчей. Укажите merchant_cols явно.")

    # Приводим к float (на случай целочисленных/строковых артефактов)
    merch = df[merchant_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 1) число мерчей с ненулевыми тратами
    nonzero_count = (merch > 0).sum(axis=1)
    cond_nonzero = nonzero_count >= min_nonzero_merchants

    # 2) доля "малых" трат по сумме
    total_spend = merch.sum(axis=1)
    small_spend = merch.where(merch <= small_tx_threshold, 0).sum(axis=1)

    # Избежать деления на ноль: если total_spend == 0, считаем долю 0
    small_share = np.where(total_spend > 0, small_spend / total_spend, 0.0)
    cond_small_share = small_share >= small_tx_share

    mask = cond_nonzero & cond_small_share
    result = df.loc[mask].copy()

    # При желании можно добавить диагностические поля:
    result["_nonzero_merchants"] = nonzero_count[mask].values
    result["_small_spend"] = small_spend[mask].values
    result["_total_spend"] = total_spend[mask].values
    result["_small_share"] = small_share[mask]

    return result
