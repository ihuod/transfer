fig_facet = px.bar(
    avg_df,
    x="CHANNEL",
    y="AVG_AMOUNT",
    color="CHANNEL",
    facet_col="CURRENCY",
    facet_col_spacing=0.07,
    category_orders={"CURRENCY": desired_order, "CHANNEL": ["SOU", "ERIP"]},
    color_discrete_map=channel_colors,
    text_auto=".2s",
    title="Средняя сумма транзакции (|сумма|): SOU vs ERIP по валютам"
)
fig_facet.update_layout(template="plotly_white", height=450, showlegend=False)
fig_facet.for_each_yaxis(lambda a: a.update(title=None))
fig_facet.for_each_xaxis(lambda a: a.update(title=None))
fig_facet.show()

# --- Визуализация 2: сгруппированные столбики по валютам ---
fig_grouped = px.bar(
    avg_df,
    x="CURRENCY",
    y="AVG_AMOUNT",
    color="CHANNEL",
    barmode="group",
    category_orders={"CURRENCY": desired_order, "CHANNEL": ["SOU", "ERIP"]},
    color_discrete_map=channel_colors,
    text_auto=".2s",
    title="Средняя сумма транзакции (|сумма|): по валютам и каналам"
)
fig_grouped.update_layout(template="plotly_white", height=450, legend_title=None, xaxis_title=None, yaxis_title=None)
fig_grouped.show()
