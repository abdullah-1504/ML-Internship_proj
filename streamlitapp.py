import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="centered"
)

# Add title and description
st.title("Iris Species Classifier 🌸")
st.write("""
### Predict Iris Flower Species
This app predicts the Iris flower species based on the flower's measurements.
""")

def load_model():
    """Load the trained model and preprocessors"""
    try:
        model = joblib.load('iris_model.pkl')
        scaler = joblib.load('iris_scaler.pkl')
        le = joblib.load('iris_label_encoder.pkl')
        features = joblib.load('iris_features.pkl')
        return model, scaler, le, features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def make_prediction(features_input, model, scaler, le):
    """Make prediction using the trained model"""
    try:
        # Scale features
        features_scaled = scaler.transform([features_input])
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get species name and probabilities
        species = le.inverse_transform(prediction)[0]
        prob_dict = {
            le.inverse_transform([i])[0]: round(prob * 100, 2)
            for i, prob in enumerate(probabilities)
        }
        
        return species, prob_dict
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def main():
    # Load model and preprocessors
    model, scaler, le, features = load_model()
    
    if model is None:
        st.error("Please make sure model files exist in the directory")
        return

    # Create input form
    with st.form("prediction_form"):
        st.write("Enter flower measurements (cm):")
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.number_input("Sepal Length", 4.0, 8.0, 5.4)
            sepal_width = st.number_input("Sepal Width", 2.0, 5.0, 3.4)
        
        with col2:
            petal_length = st.number_input("Petal Length", 1.0, 7.0, 4.7)
            petal_width = st.number_input("Petal Width", 0.1, 2.5, 1.4)
        
        submit = st.form_submit_button("Predict Species")
    
    # Make prediction when form is submitted
    if submit:
        features_input = [sepal_length, sepal_width, petal_length, petal_width]
        species, probabilities = make_prediction(features_input, model, scaler, le)
        
        if species and probabilities:
            # Show prediction
            st.success(f"Predicted Species: **{species}**")
            
            # Show probability bars
            st.write("### Confidence Scores")
            for species_name, prob in probabilities.items():
                st.write(f"{species_name}")
                st.progress(prob/100)
                st.write(f"{prob}%")
            
            # Show input values
            st.write("### Input Values")
            input_df = pd.DataFrame([features_input], 
                                  columns=['Sepal Length', 'Sepal Width', 
                                         'Petal Length', 'Petal Width'])
            st.dataframe(input_df)

if __name__ == "__main__":
    main()