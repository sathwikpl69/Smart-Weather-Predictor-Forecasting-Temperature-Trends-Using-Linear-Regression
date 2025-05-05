# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and assets
model = pickle.load(open("linear_model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))
metrics = pd.read_csv("metrics.csv").iloc[0]
df = pd.read_csv("cleaned_weather_data.csv")

st.set_page_config(page_title="Smart Weather Predictor", layout="wide")
st.title("🌤️ Smart Weather Predictor (Linear Regression)")

tab1, tab2, tab3 = st.tabs(["🔍 Predict Temperature", "📊 EDA", "📈 Model Performance"])

# Tab 1 - Prediction
with tab1:
    st.subheader("Enter Weather Conditions:")

    input_data = []
    for feature in features:
        # Use sliders for continuous features to improve user interaction
        if feature in ['humidity', 'pressure_mb', 'cloud', 'wind_mph', 'wind_kph', 'latitude', 'longitude']:
            value = st.slider(f"{feature.replace('_', ' ').title()}", 
                              min_value=float(df[feature].min()), 
                              max_value=float(df[feature].max()), 
                              value=float(df[feature].mean()))
        else:
            value = st.number_input(f"{feature.replace('_', ' ').title()}", value=float(df[feature].mean()))
        input_data.append(value)

    if st.button("Predict Temperature"):
        try:
            # Make prediction and display the result
            prediction = model.predict([input_data])[0]
            st.success(f"🌡️ Predicted Temperature: {prediction:.2f} °C")
        except Exception as e:
            st.error(f"Error: {e}")

# Tab 2 - EDA (Exploratory Data Analysis)
with tab2:
    st.subheader("Explore Weather Data")

    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Correlation Heatmap**")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt.gcf())
        
    with col2:
        st.write("**Pair Plot**")
        sns.pairplot(df)
        st.pyplot(plt.gcf())

    # Adding more interactivity in the plot
    st.write("**Distribution of Temperature**")
    plt.figure(figsize=(8, 5))
    sns.histplot(df['temperature_celsius'], kde=True)
    st.pyplot(plt.gcf())

# Tab 3 - Model Performance and Evaluation
with tab3:
    st.subheader("Model Evaluation Metrics")
    st.metric("R² Score", f"{metrics['R2 Score']:.3f}")
    st.metric("MAE", f"{metrics['MAE']:.2f}")
    st.metric("RMSE", f"{metrics['RMSE']:.2f}")

    st.write("**Feature Importance (Coefficients):**")
    coeff_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
    coeff_df = coeff_df.sort_values(by="Coefficient", key=abs, ascending=False)
    st.dataframe(coeff_df)

    # Plotting feature importance as a bar chart
    plt.figure(figsize=(8, 5))
    sns.barplot(data=coeff_df, x="Coefficient", y="Feature", palette="coolwarm")
    st.pyplot(plt.gcf())

    # Plot actual vs predicted temperature
    st.write("**Actual vs Predicted Temperature**")
    y_pred = model.predict(df[features])
    plt.figure(figsize=(8, 5))
    plt.scatter(df['temperature_celsius'], y_pred, alpha=0.7, color='green')
    plt.plot([df['temperature_celsius'].min(), df['temperature_celsius'].max()], 
             [df['temperature_celsius'].min(), df['temperature_celsius'].max()], 'r--', linewidth=2)
    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.title("Actual vs Predicted Temperature")
    st.pyplot(plt.gcf())

    # Residuals plot
    st.write("**Residuals Plot**")
    residuals = df['temperature_celsius'] - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(df['temperature_celsius'], residuals, alpha=0.7, color='blue')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Actual Temperature")
    st.pyplot(plt.gcf())
