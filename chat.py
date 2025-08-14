import re
import numpy as np
import pandas as pd
import plotly.express as px

# --- 0) Загрузка данных ---
# Замените путь на ваш
# df = pd.read_csv("/absolute/path/to/erip_2024-07-01_2025-07-31.csv")

# Если датафрейм уже есть в памяти, пропустите строку чтения csv
# и начинайте с блока "Приведение типов"

# --- 1) Приведение типов и очистка ---
def to_numeric_amount(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    s = s.replace("\xa0", "").replace(" ", "")
    s = re.sub(r"[^0-9,.\-]", "", s)
    # Если и запятая, и точка — выбираем последний разделитель как десятичный,
    # остальные удаляем (как разделители тысяч)
    if "," in s and "." in s:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        # Только запятая — считаем её десятичным разделителем
        if "," in s and "." not in s:
            s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

# Обязательные преобразования
df["EV_SYS_RGSTRN_DTTM"] = pd.to_datetime(df["EV_SYS_RGSTRN_DTTM"], errors="coerce")
df["EV_TSACTN_AMT_NUM"] = df["EV_TSACTN_AMT"].apply(to_numeric_amount)

# Объём операции (для сумм используем абсолютное значение, чтобы отразить оборот)
df["AMOUNT_ABS"] = df["EV_TSACTN_AMT_NUM"].abs()

# Валюты (ISO numeric): BYN=933, USD=840, EUR=978, RUB=643
currency_map = {933: "BYN", 840: "USD", 978: "EUR", 643: "RUB"}
df["CURRENCY"] = df["SYS_CURY_NUM"].map(currency_map).fillna(df["SYS_CURY_NUM"].astype(str))

# Канал: нормализуем к SOU/ERIP
def normalize_channel(x):
    s = str(x).strip().upper()
    if "ERIP" in s:
        return "ERIP"
    if "SOU" in s:
        return "SOU"
    return "OTHER"

df["CHANNEL"] = df["DSCPTV_FEAT_TYPE_CD"].apply(normalize_channel)

# --- 2) Агрегации по валютам ---
by_currency = (
    df.groupby("CURRENCY", as_index=False)
      .agg(
          NUM_TXN=("PRTY_ID", "count"),
          SUM_AMOUNT=("AMOUNT_ABS", "sum")
      )
)

# Упорядочим валюты
desired_order = ["BYN", "USD", "EUR", "RUB"]
by_currency["CURRENCY"] = pd.Categorical(by_currency["CURRENCY"], categories=desired_order, ordered=True)
by_currency = by_currency.sort_values("CURRENCY")

# --- 3) Агрегации по каналам (SOU vs ERIP) ---
mask_se = df["CHANNEL"].isin(["SOU", "ERIP"])
counts_se = (
    df.loc[mask_se, "CHANNEL"]
      .value_counts()
      .rename_axis("CHANNEL")
      .reset_index(name="NUM_TXN")
)

sums_se = (
    df.loc[mask_se]
      .groupby("CHANNEL", as_index=False)["AMOUNT_ABS"]
      .sum()
)

# Цвета
currency_palette = px.colors.qualitative.Set2
channel_colors = {"ERIP": "#636EFA", "SOU": "#EF553B"}

# --- 4) Графики по валютам ---
fig_count = px.bar(
    by_currency,
    x="CURRENCY",
    y="NUM_TXN",
    color="CURRENCY",
    color_discrete_sequence=currency_palette,
    title="Количество транзакций по валютам",
    text="NUM_TXN"
)
fig_count.update_layout(
    template="plotly_white",
    xaxis_title=None,
    yaxis_title=None,
    legend_title=None,
    height=450
)
fig_count.update_traces(textposition="outside")

fig_sum = px.bar(
    by_currency,
    x="CURRENCY",
    y="SUM_AMOUNT",
    color="CURRENCY",
    color_discrete_sequence=currency_palette,
    title="Сумма транзакций по валютам (оборот, |сумма|)",
    text_auto=".2s"
)
fig_sum.update_layout(
    template="plotly_white",
    xaxis_title=None,
    yaxis_title=None,
    legend_title=None,
    height=450
)

# --- 5) Круговые диаграммы SOU vs ERIP ---
fig_pie_count = px.pie(
    counts_se,
    names="CHANNEL",
    values="NUM_TXN",
    color="CHANNEL",
    color_discrete_map=channel_colors,
    title="Доля числа операций: SOU vs ERIP",
    hole=0.35
)
fig_pie_count.update_traces(textposition="inside", textinfo="percent+label")
fig_pie_count.update_layout(template="plotly_white", height=450, legend_title=None)

fig_pie_sum = px.pie(
    sums_se,
    names="CHANNEL",
    values="AMOUNT_ABS",
    color="CHANNEL",
    color_discrete_map=channel_colors,
    title="Доля суммы транзакций (оборот): SOU vs ERIP",
    hole=0.35
)
fig_pie_sum.update_traces(textposition="inside", textinfo="percent+label")
fig_pie_sum.update_layout(template="plotly_white", height=450, legend_title=None)

# --- 6) Показать графики ---
fig_count.show()
fig_sum.show()
fig_pie_count.show()
fig_pie_sum.show()
