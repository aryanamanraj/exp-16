import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'Area': [1000, 1200, 1500, 1800, 2000,
             2200, 2500, 2800, 3000, 3500],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    'Price': [200000, 240000, 300000, 360000, 400000,
              440000, 500000, 560000, 600000, 700000]
}

df = pd.DataFrame(data)

print(df)
X = df[['Area', 'Bedrooms']]
y = df['Price']

print("Input Features:")
print(X)

print("\nTarget Variable:")
print(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))
model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully!")
y_pred = model.predict(X_test)

print("Actual Prices:")
print(y_test.to_numpy())

print("\nPredicted Prices:")
print(np.round(y_pred, 2))
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 4))
new_house = pd.DataFrame({
    'Area': [2400],
    'Bedrooms': [4]
})

predicted_price = model.predict(new_house)

print("Area:", new_house['Area'][0], "sq. ft.")
print("Bedrooms:", new_house['Bedrooms'][0])
print("Predicted House Price:", round(predicted_price[0], 2))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000,
             2200, 2500, 2800, 3000, 3500],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    'Price': [200000, 240000, 300000, 360000, 400000,
              440000, 500000, 560000, 600000, 700000]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# Select features and target
X = df[['Area', 'Bedrooms']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

print("\nActual Prices:", y_test.to_numpy())
print("Predicted Prices:", np.round(y_pred, 2))

print("Mean Squared Error:",
      round(mean_squared_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 4))

# Predict a new house price
new_house = pd.DataFrame({
    'Area': [2400],
    'Bedrooms': [4]
})

price = model.predict(new_house)[0]
print("\nPredicted Price for New House:", round(price, 2))

# Plot results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
