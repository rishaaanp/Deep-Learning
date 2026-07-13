import numpy as np
import time
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------
# Generate Dataset
# -----------------------------------------------------
X, y = make_regression(n_samples=100000, n_features=20, noise=10, random_state=42)

# Standardize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add Bias Term
X = np.c_[np.ones(X.shape[0]), X]

# Reshape Target
y = y.reshape(-1, 1)


# -----------------------------------------------------
# Mean Squared Error Function
# -----------------------------------------------------
def mse(X, y, theta):
    predictions = X @ theta
    return np.mean((predictions - y) ** 2)


# -----------------------------------------------------
# Gradient Descent Function
# -----------------------------------------------------
def gradient_descent(X, y, learning_rate=0.01, epochs=500, batch_size=None):

    m, n = X.shape
    theta = np.zeros((n, 1))
    updates = 0

    start_time = time.time()

    # Batch GD if batch_size is None
    if batch_size is None:
        batch_size = m

    for epoch in range(epochs):

        # Shuffle dataset every epoch
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):

            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]

            predictions = X_batch @ theta

            gradient = (2 / len(X_batch)) * X_batch.T @ (predictions - y_batch)

            theta -= learning_rate * gradient

            updates += 1

    end_time = time.time()

    return {
        "theta": theta,
        "loss": mse(X, y, theta),
        "time": end_time - start_time,
        "updates": updates,
    }


# -----------------------------------------------------
# Train Models
# -----------------------------------------------------

# Batch Gradient Descent
batch = gradient_descent(X, y, batch_size=len(X))

# Stochastic Gradient Descent
sgd = gradient_descent(X, y, batch_size=1)

# Mini-Batch Gradient Descent (Batch Size = 128)
mini = gradient_descent(X, y, batch_size=128)

# -----------------------------------------------------
# Display Results
# -----------------------------------------------------

print("\n" + "=" * 75)
print(
    "{:<20} {:>15} {:>18} {:>15}".format(
        "Optimizer", "Time (s)", "Updates", "Final Loss"
    )
)
print("=" * 75)

print(
    "{:<20} {:>15.4f} {:>18} {:>15.4f}".format(
        "Batch GD", batch["time"], batch["updates"], batch["loss"]
    )
)

print(
    "{:<20} {:>15.4f} {:>18} {:>15.4f}".format(
        "SGD", sgd["time"], sgd["updates"], sgd["loss"]
    )
)

print(
    "{:<20} {:>15.4f} {:>18} {:>15.4f}".format(
        "Mini-Batch GD", mini["time"], mini["updates"], mini["loss"]
    )
)

print("=" * 75)

# -----------------------------------------------------
# Find Best Optimizer
# -----------------------------------------------------

results = {"Batch GD": batch, "SGD": sgd, "Mini-Batch GD": mini}

best_optimizer = min(results.items(), key=lambda x: x[1]["loss"])[0]

print("\nBest Optimizer (Based on Final Loss):", best_optimizer)

print("\nObservations:")
print(f"• Batch GD       : {batch['updates']} updates, Loss = {batch['loss']:.2f}")
print(f"• SGD            : {sgd['updates']} updates, Loss = {sgd['loss']:.2f}")
print(f"• Mini-Batch GD  : {mini['updates']} updates, Loss = {mini['loss']:.2f}")
