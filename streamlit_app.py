import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Feature ranges and descriptions
FEATURE_INFO = {
    'CRIM': {'min': 0.0, 'max': 100.0, 'help': 'Crime rate per capita'},
    'ZN': {'min': 0.0, 'max': 100.0, 'help': 'Proportion of residential land zoned'},
    'TAX': {'min': 100.0, 'max': 1000.0, 'help': 'Property tax rate'},
    'B': {'min': 0.0, 'max': 400.0, 'help': 'Proportion of Black residents'},
    'PTRATIO': {'min': 10.0, 'max': 25.0, 'help': 'Pupil-teacher ratio'},
    'LSTAT': {'min': 0.0, 'max': 40.0, 'help': 'Percentage lower status population'},
    'RM': {'min': 3.0, 'max': 9.0, 'help': 'Average number of rooms'},
    'AGE': {'min': 0.0, 'max': 100.0, 'help': 'Proportion of units built before 1940'},
    'DIS': {'min': 1.0, 'max': 12.0, 'help': 'Distance to employment centers'},
    'RAD': {'min': 1.0, 'max': 24.0, 'help': 'Accessibility to highways'}
}

# Load model and preprocessors
@st.cache_resource
def load_model():
    try:
        model = joblib.load('linear_regression_model.pkl')
        scaler = joblib.load('scaler.pkl')
        selected_features = joblib.load('selected_features.pkl')
        return model, scaler, selected_features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def main():
    # Page title and description
    st.title("Boston House Price Predictor 🏠")
    st.write("""
    ### Predict house prices in Boston
    Enter the house features below to get a price prediction.
    """)

    # Load model and features
    model, scaler, selected_features = load_model()
    
    if model is None:
        st.error("Please ensure model files exist in the directory")
        return

    # Create input form
    with st.form("prediction_form"):
        st.subheader("House Features")
        
        # Create two columns for inputs
        feature_values = {}
        col1, col2 = st.columns(2)
        
        # Distribute features between columns with appropriate ranges
        for i, feature in enumerate(selected_features):
            with col1 if i % 2 == 0 else col2:
                info = FEATURE_INFO.get(feature, {'min': 0.0, 'max': 100.0, 'help': feature})
                feature_values[feature] = st.number_input(
                    f"{feature}",
                    min_value=float(info['min']),
                    max_value=float(info['max']),
                    value=float(info['min']),
                    help=info['help']
                )
        
        submit = st.form_submit_button("Predict Price")
    
    # Make prediction when form is submitted
    if submit:
        try:
            # Create DataFrame with raw input values
            input_df = pd.DataFrame([feature_values])[selected_features]
            
            # Scale the input data
            scaled_data = scaler.transform(input_df)
            
            # Make prediction and ensure it's positive
            prediction = abs(model.predict(scaled_data)[0])  # Add abs() here
            
            # Convert prediction to actual price (in thousands)
            predicted_price = max(0, prediction * 1000)  # Ensure non-negative
            
            # Show prediction with formatting
            if predicted_price > 0:
                st.success(f"### Predicted House Price: ${predicted_price:,.2f}")
            else:
                st.warning("Warning: Prediction seems unusually low. Please check your input values.")
            
            # Show input summary with original (unscaled) values
            st.write("### Input Features")
            st.dataframe(input_df)
            
            # Add model confidence based on input ranges
            within_normal_range = all(
                FEATURE_INFO[feature]['min'] <= feature_values[feature] <= FEATURE_INFO[feature]['max']
                for feature in selected_features if feature in FEATURE_INFO
            )
            
            confidence_message = """
            #### Prediction Confidence
            - ✅ Within expected range
            """ if within_normal_range else """
            #### Prediction Confidence
            - ⚠️ Some inputs are outside typical ranges
            """
            
            st.info(confidence_message)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
