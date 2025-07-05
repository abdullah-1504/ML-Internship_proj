import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Boston House Price Prediction")

# Load saved model, scaler and features
try:
    model = joblib.load('linear_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    selected_features = joblib.load('selected_features.pkl')
except Exception as e:
    raise Exception("Error loading model files. Make sure all .pkl files exist")

# Dynamic creation of HouseFeatures class based on selected features
class HouseFeatures(BaseModel):
    def __init__(self, **data):
        super().__init__(**data)
    
    class Config:
        schema_extra = {
            "example": {feature: 0.0 for feature in selected_features}
        }

# Dynamically add fields to HouseFeatures
for feature in selected_features:
    setattr(HouseFeatures, feature, float)

@app.get("/")
def home():
    return {
        "message": "Boston House Price Prediction API",
        "required_features": selected_features,
        "example_input": {feature: 0.0 for feature in selected_features}
    }

@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        # Convert input to DataFrame with correct feature order
        df = pd.DataFrame([features.dict()])[selected_features]
        
        # Scale features
        scaled_features = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        return {
            "predicted_price": round(prediction * 1000, 2),  # Convert to actual dollars
            "prediction_unit": "USD",
            "features_used": selected_features
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)