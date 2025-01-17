import streamlit as st
import pandas as pd
import joblib

# Load the saved model, label encoders, and scaler
model = joblib.load('model_rf.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Define numerical features and expected columns
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
expected_features = model.feature_names_in_  # Get the features used during training

# Streamlit app
st.title("Churn Prediction App")

st.sidebar.header("User Input Features")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file for predictions", type=["csv"])

if uploaded_file is not None:
    # Load user data
    user_df = pd.read_csv(uploaded_file)

    # Remove the 'Churn' column if it exists
    if 'Churn' in user_df.columns:
        user_df = user_df.drop(columns=['Churn'])

    # Check if all expected features exist in the uploaded file
    missing_features = set(expected_features) - set(user_df.columns)
    if missing_features:
        st.error(f"The uploaded file is missing the following required features: {', '.join(missing_features)}")
    else:
        # Preprocess the data
        for column, encoder in label_encoders.items():
            if column in user_df:
                user_df[column] = encoder.transform(user_df[column])

        # Scale numerical features
        user_df[numerical_features] = scaler.transform(user_df[numerical_features])

        # Ensure the input DataFrame matches the training features
        user_df = user_df[expected_features]

        # Make predictions and get probabilities
        predictions = model.predict(user_df)
        probabilities = model.predict_proba(user_df)[:, 1]  # Probability of class 1 (Churn)

        # Create a DataFrame for the results
        results = pd.DataFrame({
            "Prediction (0 = No Churn, 1 = Churn)": predictions,
            "Probability of Churn": probabilities
        })

        st.write("### Predictions with Probabilities")
        st.write(results)

        # Display summary statistics
        churn_rate = probabilities.mean() * 100
        st.write(f"### Average Probability of Churn: {churn_rate:.2f}%")
else:
    st.write("Please upload a CSV file for predictions.")
