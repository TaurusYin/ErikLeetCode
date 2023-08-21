import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 创建一个48小时的时间序列数据
data = {
    "adj_wrk_date": [datetime.now() - timedelta(hours=i) for i in range(47, -1, -1)],
    "qty": [100 + 10 * i for i in range(48)],
}


df = pd.DataFrame(data)

def ts_pred_simple(x):
    model = ExponentialSmoothing(x, trend='add', seasonal='add', seasonal_periods=24)
    model_fit = model.fit(optimized=True, remove_bias=True)
    y_pred = model_fit.predict(len(x), len(x)+24-1)

    return [max(0, int(i)) for i in y_pred]

ts = df.qty.values
prediction = ts_pred_simple(ts)

# Prepare the date range for prediction
pred_dates = [df.adj_wrk_date.max() + timedelta(hours=i) for i in range(1, 25)]

# Combine original data and prediction data for plotting
df_plot = pd.concat([
    df,
    pd.DataFrame({"adj_wrk_date": pred_dates, "qty": prediction})
])

# Sort df_plot by adj_wrk_date
df_plot.sort_values(by='adj_wrk_date', inplace=True)
df_plot.reset_index(drop=True, inplace=True)

# Plotting
plt.figure(figsize=(10,6))
plt.plot(df_plot.adj_wrk_date, df_plot.qty, marker='o', label="Quantity")
plt.plot(df_plot.adj_wrk_date.iloc[-24:], df_plot.qty.iloc[-24:], color='r', marker='o', label="Predicted Quantity")
plt.title("Original vs Predicted Quantity")
plt.xlabel("Time")
plt.ylabel("Quantity")
plt.legend()
plt.grid(True)
plt.show()
