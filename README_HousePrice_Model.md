# 🏠 House Price Prediction Using Linear Regression

This project uses the **Boston Housing dataset** to train a Linear Regression model that predicts house prices based on key socio-economic and geographical features.

---

## 📌 Objective

To build a robust and interpretable machine learning model using:
- Outlier removal
- Correlation-based feature selection
- Variance Inflation Factor (VIF) to handle multicollinearity
- Linear Regression for price prediction
- Visual error analysis and R² evaluation

---

## 🧰 Tech Stack

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn (LinearRegression, StandardScaler, Pipeline)
- Joblib for model serialization

---

## 🧪 Features Engineered

- Features selected based on correlation threshold > 0.5
- VIF applied to remove multicollinear predictors
- Outliers removed using the IQR method (2×IQR bound)

---

## 🧠 Model Training & Evaluation

- Model: `LinearRegression()`
- Scaler: `StandardScaler`
- Evaluation Metrics:
  - R² Score: **0.78**
  - Residual Plot
  - Actual vs Predicted Scatter Plot (with color-coded errors)

---

## 📈 Visualization Highlight

The final output includes a professional-quality scatter plot:
- Dashed line = perfect prediction
- Green dots = low error
- Red dots = high error
- R² Score annotated clearly

---

## 💾 Deployment Ready

The notebook includes:
- Pipeline creation (`Pipeline([('scaler', scaler), ('model', model)])`)
- Saving model via `joblib.dump(pipeline, 'model_pipeline.pkl')`
- Prediction function: `predict_house_price(features)`

---

## 👤 Author

**Abdullah Awan**  
Machine Learning Intern – Summer 2025

---
