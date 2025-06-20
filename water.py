# -------------------------------
# 🌊 Water Quality Prediction
# Internship Project - Edunet AI-ML
# -------------------------------

# ✅ STEP 1: Install & Import Required Libraries
# !pip install pandas numpy matplotlib seaborn scikit-learn  # Uncomment if needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ✅ STEP 2: Load and Explore Dataset
print("\n--- Loading Dataset ---")
df = pd.read_csv('PB_All_2000_2021.csv', sep=';')
print(f"\nDataset Shape: {df.shape}")

# Initial Exploration
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Statistical Summary ---")
print(df.describe().T)

# Display sample data
print("\n--- Sample Data (Before Processing) ---")
print(df.head())

# ✅ STEP 3: Data Cleaning & Feature Engineering
print("\n--- Cleaning & Feature Engineering ---")

# Convert date column
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')

# Extract date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Sort by station and time
df = df.sort_values(by=['id', 'date'])

# Handle missing values with median imputation
print("\nPerforming median imputation for missing values...")
df.fillna(df.median(numeric_only=True), inplace=True)

# Verify cleaning
print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())

print("\n--- Sample Data (After Processing) ---")
print(df.head())

# ✅ STEP 4: Define Features and Targets
features = ['NH4', 'BSK5', 'Suspended', 'year', 'month']
targets = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

X = df[features]
y = df[targets]

# ✅ STEP 5: Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain/Test Split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

# ✅ STEP 6: Train MultiOutput RandomForest Regressor
print("\n--- Model Training ---")
model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model.fit(X_train, y_train)
print("Training completed!")

# ✅ STEP 7: Make Predictions and Evaluate
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

for i, col in enumerate(targets):
    print(f"\n--- {col} ---")
    print(f"R² Score: {r2_score(y_test[col], y_pred[:, i]):.4f}")
    print(f"MSE: {mean_squared_error(y_test[col], y_pred[:, i]):.4f}")

# ✅ STEP 8: Visualize Predictions
print("\nGenerating visualizations...")
plt.style.use('seaborn')

for i, col in enumerate(targets):
    plt.figure(figsize=(8, 4))
    sns.regplot(x=y_test[col], y=y_pred[:, i], 
                scatter_kws={'alpha':0.5, 'color':'blue', 'edgecolor':'k'},
                line_kws={'color':'red'})
    plt.xlabel(f"Actual {col}")
    plt.ylabel(f"Predicted {col}")
    plt.title(f"{col}: Actual vs Predicted (R² = {r2_score(y_test[col], y_pred[:, i]):.2f})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\n✅ Analysis Complete!")