import streamlit as st
import pandas as pd
import requests
import io

st.title("💓 Heart Disease Prediction App")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Data Preview:", df.head())

    if st.button("🔍 Predict"):
        try:
            # Send file to FastAPI backend
            response = requests.post(
                "https://hearts-lpgb.onrender.com/predict",
                files={"file": uploaded_file}
            )

            if response.status_code == 200:
                predictions = response.json()
                df["prediction"] = [item["prediction"] for item in predictions]

                st.success("✅ Prediction successful!")
                st.write("📋 Prediction Results:", df)

                # CSV download
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode()

                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_bytes,
                    file_name="heart_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"❌ API returned status code {response.status_code}")
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")
