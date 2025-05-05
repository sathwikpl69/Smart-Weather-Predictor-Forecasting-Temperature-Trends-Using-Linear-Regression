# Smart Weather Predictor - Linear Regression Model
# -------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # For saving model and features

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load Dataset
df = pd.read_csv("GlobalWeatherRepository.csv")

print("Initial Data Shape:", df.shape)
print("\nSample Data:\n", df.head())

# Step 2: Drop Irrelevant Columns
columns_to_drop = [
    'country', 'location_name', 'timezone', 'last_updated', 
    'sunrise', 'sunset', 'moonrise', 'moonset', 
    'moon_phase', 'moon_illumination', 'temperature_fahrenheit', 
    'condition_text', 'wind_degree', 'wind_direction', 'precip_mm', 
    'precip_in', 'visibility_km', 'visibility_miles', 'gust_mph', 
    'gust_kph', 'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 
    'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide', 
    'air_quality_PM2.5', 'air_quality_PM10', 'air_quality_us-epa-index', 
    'air_quality_gb-defra-index', 'feels_like_fahrenheit', 'pressure_in',
    'uv_index', 'last_updated_epoch'
]
df.drop(columns=columns_to_drop, inplace=True)

# Step 3: Handle Missing Values
df.dropna(inplace=True)
print("\nData Shape After Cleaning:", df.shape)

# Step 4: No categorical encoding needed (all remaining features are numeric)

# Step 5: Define Features and Label
X = df.drop(columns=['temperature_celsius'])
y = df['temperature_celsius']

# Optional: Correlation Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap='coolwarm', center=0, annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 6: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predict and Evaluate
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📈 Model Evaluation Metrics:")
print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Step 9: Visualize Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Actual vs Predicted Temperature")
plt.grid(True)
plt.show()

# ------------------------------------------
# 🔄 Step 10: Save Artifacts for Streamlit UI
# ------------------------------------------

# Save trained model
with open("linear_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature names
with open("features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# Save metrics
metrics = {
    "R2 Score": r2,
    "MAE": mae,
    "RMSE": rmse
}
pd.DataFrame([metrics]).to_csv("metrics.csv", index=False)

# Save cleaned dataset for EDA in GUI
df.to_csv("cleaned_weather_data.csv", index=False)
