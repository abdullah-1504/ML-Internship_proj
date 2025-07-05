import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pytest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

#imported ALL libraries in this cell even the ones used in the last cells
# to avoid import errors in the pytest tests

#Load datasets for EDA

df = pd.read_csv('E:/Downloads/boston.csv')

#data loaded successfully

#Metadata

print ("\n Data Info: ")
df.info()

#basic Statistics

print ("\n Stats: ")
df.describe()


#Check for null values
print ("\n Null Values: ")
print (df.isnull().sum())

#check for duplicates
print ("\n Duplicates: ")
print ("\n Duplicates: ", df.duplicated().sum())


#shownig the first 5 rows
print ("Printing the first 5 rows: ")
print (df.head())

#printing last 5 rows

print ("The last 5 rows: ")
print (df.tail())


#Data Visualization since no null values or duplicates found
 
#box pot to identify outliers
# Large value ranges (e.g. CRIM, ZN) flatten smaller features, hiding outliers.

plt.figure(figsize=(15, 8))
sns.boxplot(data=df, palette='Set3')

# Customize the plot
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Features with Outliers', pad=20, fontsize=14)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#Large value ranges (e.g. CRIM, ZN) flatten smaller features, so individual boxplots are used.

# Create individual boxplots for each feature
fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(20, 8))
flat_axes = axes.flatten()

# Simple for loop using range and len
for i in range(len(df.columns)):
    sns.boxplot(data=df[df.columns[i]], ax=flat_axes[i], color='skyblue')
    flat_axes[i].set_title(df.columns[i], fontsize=10)
    flat_axes[i].tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.show()

#correlation matrix to assist in Linear Regression

plt.figure (figsize=(12, 8))
sns.heatmap (df.corr(), annot = True, cmap = 'coolwarm', center=0)
plt.title ('Correlation Matrix', fontsize=16)
plt.tight_layout
plt.show()


# Function to get important features based on correlation with target
def get_important_features(df, target, threshold=0.3):
    correlations = abs(df.corr()[target]).sort_values(ascending=False)
    return correlations[correlations > threshold].index.tolist()[1:]  # Exclude target itself


#Disttribution plot of target variable ('MEDV')

plt.figure (figsize=(12, 8))
sns.histplot (df['MEDV'], bins=30, kde=True, color='purple')
plt.title ('Distribution of Target Variable (MEDV)', fontsize=16)
plt.xlabel ('Median Value (MEDV)')
plt.ylabel ('Frequency/Count')
plt.show()


#rightly skewed
#clear cutoff at 50k


#Q-Q pliot to check normality of target variable (MEDV)
#if the points follow the diagonal line, the data is normally distributed.


plt.figure(figsize = (12,8))
stats.probplot(df['MEDV'], dist='norm', plot=plt)
plt.title("Q-Q plot of House Prices (MEDV)")
plt.show()

#display correlation with MEDV (Target Variable)
#r > 0.7  strong correlation
#r = 0.3–0.7  Moderate correlation
#r < 0.3  Weak correlation
#features with moderate or strong correlation are useful for linear regression


print ("\n Correlation with MEDV: ")
correlations = df.corr()['MEDV'].sort_values(ascending=False)
print (correlations)

def get_important_features(df, target, threshold=0.2):
    """Select features that have strong relationship with house prices"""
    correlations = abs(df.corr()[target]).sort_values(ascending=False)
    # Keep only features with correlation > 0.5 (strong relationships)
    return correlations[correlations > threshold].index.tolist()[1:]

def remove_outliers(df, columns):
    """Remove extreme values that could skew our predictions"""
    df_clean = df.copy()
    for column in columns:
        # Find the normal range for each feature
        Q1 = df_clean[column].quantile(0.25)  # 25th percentile
        Q3 = df_clean[column].quantile(0.75)  # 75th percentile
        IQR = Q3 - Q1  # Normal range
        
        # Define what we consider "extreme" values
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Keep only normal values
        df_clean = df_clean[(df_clean[column] >= lower_bound) & 
                           (df_clean[column] <= upper_bound)]
    return df_clean

def select_features(df, target_column):
    """Pick the best features for predicting house prices"""
    # 1. First find which features are strongly related to house prices
    important_features = get_important_features(df, target_column, threshold=0.5)
    print("\nFeatures strongly related to house prices:")
    print(important_features)
    
    # 2. Remove extreme values that could mislead our model
    features_to_clean = important_features + [target_column]
    df_cleaned = remove_outliers(df[features_to_clean], features_to_clean)
    print(f"\nRemoved {len(df) - len(df_cleaned)} unusual data points")
    
    return df_cleaned, important_features

# Clean our data and select features
df_cleaned, selected_features = select_features(df, 'MEDV')

# Prepare data for modeling
X = df_cleaned[selected_features]
y = df_cleaned['MEDV']



X_train, X_test, y_train, y_test = train_test_split(  #splitting for training and testing
X, y, test_size=0.2, random_state=42)   #20% test size, 80% train size
                                         # random state for reproducibility

scaler = StandardScaler() #features scaled so that all are on the same scale
X_train = scaler.fit_transform(X_train) #else tax or zn with heavy numbers will dominate model
X_test = scaler.transform(X_test)


model = LinearRegression() #Linear regression model initiated (empty)
model.fit(X_train, y_train) #model trained on training data .fit helps find the best fit 


y_pred = model.predict(X_test)  # predictions made on test data
print (y_pred)

mse = mean_squared_error(y_test, y_pred) #MSE caluculated to evaluate model performance
                                         #diff bw actual and predicted values (lower the better)
r2 = r2_score(y_test, y_pred) # R² score calculated to evaluate model performance
                              # 0-1 (1 is perfect fit, 0 is no fit)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}') 


residuals = y_test - y_pred #residual = actual - predicted
# Plotting residuals to check for patterns          
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Residuals')
plt.title('Residuals vs Actual Prices')
plt.show()


# Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=2)
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.show()