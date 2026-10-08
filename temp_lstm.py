# ============================================================
# TIME-SERIES TEMPERATURE PREDICTION USING LSTM
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 1. LOAD DATASET
# ============================================================

# If you have your own CSV file, change this to:
#
# df = pd.read_csv("temperature.csv")
#
# and make sure the temperature column is named "Temperature".

USE_CSV = False

if USE_CSV:

    df = pd.read_csv("temperature.csv")

    temperature = df["Temperature"].values

else:

    # --------------------------------------------------------
    # Sample temperature dataset
    # --------------------------------------------------------
    #
    # This creates a realistic-looking temperature
    # time series so the program can be run immediately.
    #

    np.random.seed(42)

    time = np.arange(0, 1000)

    temperature = (
        25
        + 5 * np.sin(time * 2 * np.pi / 50)
        + 2 * np.sin(time * 2 * np.pi / 200)
        + np.random.normal(0, 0.5, len(time))
    )


print("=" * 60)
print("LSTM TEMPERATURE PREDICTION")
print("=" * 60)

print("\nNumber of temperature values:", len(temperature))


# ============================================================
# 2. CONVERT TO NUMPY ARRAY
# ============================================================

temperature = np.array(temperature, dtype=np.float32)

# Reshape because MinMaxScaler expects
# a 2-dimensional array

temperature = temperature.reshape(-1, 1)


# ============================================================
# 3. NORMALIZE THE DATA
# ============================================================

scaler = MinMaxScaler(feature_range=(0, 1))

normalized_temperature = scaler.fit_transform(temperature)


print("\nOriginal temperature range:", temperature.min(), "to", temperature.max())

print(
    "Normalized range:",
    normalized_temperature.min(),
    "to",
    normalized_temperature.max(),
)


# ============================================================
# 4. CREATE INPUT SEQUENCES
# ============================================================

sequence_length = 30

X = []
y = []

for i in range(len(normalized_temperature) - sequence_length):

    # Previous 30 temperature values
    X.append(normalized_temperature[i : i + sequence_length])

    # Temperature immediately after them
    y.append(normalized_temperature[i + sequence_length])


X = np.array(X)
y = np.array(y)


print("\nInput sequence shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

# Use 80% for training and 20% for testing

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)

y_test = torch.tensor(y_test, dtype=torch.float32)


# ============================================================
# 7. DEFINE LSTM MODEL
# ============================================================


class TemperatureLSTM(nn.Module):

    def __init__(self, input_size=1, hidden_size=64, num_layers=2):

        super(TemperatureLSTM, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # Pass input through LSTM

        output, _ = self.lstm(x)

        # Take output from final time step

        output = output[:, -1, :]

        # Predict temperature

        output = self.fc(output)

        return output


# ============================================================
# 8. CREATE MODEL
# ============================================================

model = TemperatureLSTM(input_size=1, hidden_size=64, num_layers=2)


print("\nModel:")
print(model)


# ============================================================
# 9. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


# ============================================================
# 10. TRAIN THE MODEL
# ============================================================

epochs = 100

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(epochs):

    # Clear gradients

    optimizer.zero_grad()

    # Forward pass

    predictions = model(X_train)

    # Calculate loss

    loss = criterion(predictions, y_train)

    # Backpropagation

    loss.backward()

    # Update weights

    optimizer.step()

    # Print loss

    if (epoch + 1) % 10 == 0:

        print(f"Epoch [{epoch + 1}/{epochs}] " f"Loss: {loss.item():.6f}")


# ============================================================
# 11. MAKE PREDICTIONS
# ============================================================

model.eval()

with torch.no_grad():

    predictions = model(X_test)


# Convert PyTorch tensors to NumPy

predictions = predictions.numpy()

actual = y_test.numpy()


# ============================================================
# 12. INVERSE TRANSFORM
# ============================================================

# Convert normalized values back
# to actual temperature values

predicted_temperature = scaler.inverse_transform(predictions)

actual_temperature = scaler.inverse_transform(actual)


# ============================================================
# 13. CALCULATE TEST ERROR
# ============================================================

mse = np.mean((actual_temperature - predicted_temperature) ** 2)

rmse = np.sqrt(mse)

mae = np.mean(np.abs(actual_temperature - predicted_temperature))


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")


# ============================================================
# 14. DISPLAY SOME PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for i in range(min(10, len(actual_temperature))):

    print(
        f"Actual: {actual_temperature[i][0]:.2f} °C   "
        f"Predicted: {predicted_temperature[i][0]:.2f} °C"
    )


# ============================================================
# 15. PLOT ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(actual_temperature, label="Actual Temperature")

plt.plot(predicted_temperature, label="Predicted Temperature")

plt.xlabel("Time")
plt.ylabel("Temperature (°C)")

plt.title("Actual vs Predicted Temperature using LSTM")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 16. PLOT TRAINING DATA
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(temperature, label="Temperature")

plt.xlabel("Time")

plt.ylabel("Temperature (°C)")

plt.title("Temperature Time Series")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
