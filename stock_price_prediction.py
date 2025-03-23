import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# Generating synthetic stock price data
def generate_stock_data(n_days=300):
    np.random.seed(42)
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    prices = np.cumsum(np.random.randn(n_days) * 2 + 50)  # Simulated price movement
    volume = np.random.randint(1000, 5000, n_days)  # Random trading volume
    return pd.DataFrame({'Date': dates, 'Stock Price': prices, 'Volume': volume})

# Creating DataFrame
df = generate_stock_data()
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df.index + 1  # Creating numerical day feature
print("Sample Stock Data:")
print(df.head())

# Plotting stock price trend
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Stock Price'], label='Stock Price', color='blue')
plt.xlabel("Date")
plt.ylabel("Stock Price ($)")
plt.title("Stock Price Trend")
plt.legend()
plt.show()

# Splitting data into features and target variable
X = df[['Day', 'Volume']]
y = df['Stock Price']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Plotting actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='red', alpha=0.5)
plt.xlabel("Actual Prices ($)")
plt.ylabel("Predicted Prices ($)")
plt.title("Actual vs Predicted Stock Prices")
plt.show()

# Predicting future stock prices
future_days = np.array([i for i in range(len(df), len(df) + 30)]).reshape(-1, 1)
future_volume = np.random.randint(1000, 5000, 30).reshape(-1, 1)
future_features = np.hstack((future_days, future_volume))
future_predictions = model.predict(future_features)

# Creating a DataFrame for future predictions
future_dates = [df['Date'].max() + datetime.timedelta(days=i) for i in range(1, 31)]
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Stock Price': future_predictions, 'Predicted Volume': future_volume.flatten()})

# Plotting future predictions
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Stock Price'], label='Historical Prices', color='blue')
plt.plot(future_df['Date'], future_df['Predicted Stock Price'], label='Predicted Prices', color='green', linestyle='dashed')
plt.xlabel("Date")
plt.ylabel("Stock Price ($)")
plt.title("Stock Price Prediction")
plt.legend()
plt.show()

# Analyzing trading volume trend
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Volume'], label='Trading Volume', color='purple')
plt.xlabel("Date")
plt.ylabel("Volume")
plt.title("Trading Volume Trend")
plt.legend()
plt.show()

# Saving results to a CSV file
df.to_csv("historical_stock_prices.csv", index=False)
future_df.to_csv("predicted_stock_prices.csv", index=False)
print("\nStock data saved to CSV files.")

# Displaying model coefficients
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Predicting for new sample data
new_data = pd.DataFrame({'Day': [310, 320, 330], 'Volume': [4000, 4200, 4400]})
new_predictions = model.predict(new_data)
print("\nPredicted Prices for New Data:")
for i, price in enumerate(new_predictions):
    print(f"Day {new_data['Day'][i]}: ${price:.2f}")
