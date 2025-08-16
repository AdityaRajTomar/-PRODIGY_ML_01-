import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Set seed
np.random.seed(42)

# Generate data
n_samples = 50
square_footage = np.random.randint(1000, 4000, size=n_samples)
bedrooms = np.random.randint(2, 6, size=n_samples)
bathrooms = np.random.randint(1, 4, size=n_samples)

# Price = base formula + noise
price = (square_footage * 150) + (bedrooms * 10000) + (bathrooms * 5000) + np.random.randint(-10000, 10000, size=n_samples)

# Create DataFrame
df = pd.DataFrame({
    'square_footage': square_footage,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'price': price
})

# Train/test split
X = df[['square_footage', 'bedrooms', 'bathrooms']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", round(mse, 2))

# Scatter plot of predictions vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', s=70)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()
