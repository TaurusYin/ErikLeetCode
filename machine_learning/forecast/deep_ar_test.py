import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

# Define the time period
time_index = pd.date_range(start="01-01-2020", periods=100)

# Generate the sales data for three products
np.random.seed(0)
sales_product_1 = 50 + np.sin(np.linspace(0, 2*np.pi, 100)) * 10 + np.random.normal(scale=5, size=100)
sales_product_2 = 100 + np.sin(np.linspace(0, 4*np.pi, 100)) * 20 + np.random.normal(scale=10, size=100)
sales_product_3 = 200 + np.sin(np.linspace(0, 6*np.pi, 100)) * 30 + np.random.normal(scale=15, size=100)

# Create a DataFrame
df = pd.DataFrame(index=time_index)
df["Product 1"] = sales_product_1
df["Product 2"] = sales_product_2
df["Product 3"] = sales_product_3

# Prepare training data
training_data = ListDataset(
    [{"start": df.index[0], "target": df[column]} for column in df.columns],
    freq = "D"
)

# Define the estimator
estimator = DeepAREstimator(freq="D", prediction_length=30, trainer=Trainer(epochs=10))

# Train the model
predictor = estimator.train(training_data=training_data)

# Prepare testing data
test_data = ListDataset(
    [{"start": df.index[0], "target": df[column]} for column in df.columns],
    freq = "D"
)

# Use the model to make predictions and plot the results
for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
    plt.figure(figsize=(10, 7))
    plt.plot(pd.date_range(start="01-01-2020", periods=130), np.concatenate([test_entry['target'], forecast.samples.reshape(30,).mean(axis=0)]), label="Predicted")
    plt.plot(pd.date_range(start="01-01-2020", periods=100), test_entry['target'], label="Actual")
    plt.legend()
    plt.grid(True)
    plt.show()
