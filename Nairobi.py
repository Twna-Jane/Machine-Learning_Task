#150364 Ndungi Tiffany Waithira
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset 
data = pd.read_csv("Nairobi Office Price Ex.csv")


# Extract SIZE as feature (x) and PRICE as target (y)
x = data['SIZE'].values  # Feature: office size
y = data['PRICE'].values  # Target: office price

# Function to compute Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    N = len(y_true)
    mse = (1 / N) * np.sum((y_true - y_pred) ** 2)
    return mse

# Gradient Descent function to update slope (m) and intercept (c)
def gradient_descent(x, y, m, c, learning_rate):
    N = len(y)
    y_pred = m * x + c
    dm = (-2 / N) * np.sum(x * (y - y_pred))
    dc = (-2 / N) * np.sum(y - y_pred)
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

# Initialize parameters
m = np.random.rand()
c = np.random.rand()
learning_rate = 0.0001
epochs = 10

print(f"Initial slope (m): {m:.4f}, Initial intercept (c): {c:.4f}")

# Train the model
for epoch in range(epochs):
    y_pred = m * x + c
    error = mean_squared_error(y, y_pred)
    print(f"Epoch {epoch + 1}: MSE = {error:.4f}")
    m, c = gradient_descent(x, y, m, c, learning_rate)

print(f"Final slope (m): {m:.4f}, Final intercept (c): {c:.4f}")

predicted_size = m * 100 + c
print(f"For office size 100 sq. ft. the price will be: {predicted_size:.4f}")

# Plot the line of best fit
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, m * x + c, color='red', label='Best Fit Line')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.title('Linear Regression: Office Size vs Price')
plt.legend()
plt.show()




