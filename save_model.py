import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load and prepare data
df = pd.read_csv('E:/Downloads/boston.csv')

def get_important_features(df, target, threshold=0.5):
    """Select features based on correlation with target"""
    correlations = abs(df.corr()[target]).sort_values(ascending=False)
    return correlations[correlations > threshold].index.tolist()[1:]

def remove_outliers(df, columns):
    """Remove outliers using IQR method"""
    df_clean = df.copy()
    for column in columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[column] >= lower_bound) & 
                           (df_clean[column] <= upper_bound)]
    return df_clean

# Select features and clean data
selected_features = get_important_features(df, 'MEDV', threshold=0.5)
features_to_clean = selected_features + ['MEDV']
df_cleaned = remove_outliers(df[features_to_clean], features_to_clean)

# Prepare data for modeling
X = df_cleaned[selected_features]
y = df_cleaned['MEDV']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save model, scaler and features
joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_features, 'selected_features.pkl')

# Print summary
print("\nModel saved successfully!")
print("Selected features:", selected_features)
print("Number of features:", len(selected_features))