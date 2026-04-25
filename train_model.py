import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the UCI Dataset
df = pd.read_csv('predictive_maintenance.csv')

# 2. Define base features
base_features = [
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]'
]

# ────────────────────────────────────────────────────────────────────────────
# 3. CREATE TIME-SERIES LAGGED FEATURES 
# ────────────────────────────────────────────────────────────────────────────

# Sort by index to ensure chronological order
df = df.sort_index().reset_index(drop=True)

# 3a. Trend Features (Rate of Change)
# These capture if a value is rising/falling, not just its absolute value
df['air_temp_trend_1'] = df['Air temperature [K]'].diff().fillna(0)  # Change from 1 reading ago
df['proc_temp_trend_1'] = df['Process temperature [K]'].diff().fillna(0)
df['torque_trend_1'] = df['Torque [Nm]'].diff().fillna(0)
df['wear_trend_1'] = df['Tool wear [min]'].diff().fillna(0)

# 3b. Velocity Features (Rate of change over 3 readings = acceleration)
df['air_temp_velocity_3'] = df['Air temperature [K]'].rolling(window=3, min_periods=1).mean().diff().fillna(0)
df['wear_velocity_3'] = df['Tool wear [min]'].rolling(window=3, min_periods=1).mean().diff().fillna(0)

# 3c. Rolling Statistics (Window of last 5 readings)
df['air_temp_rolling_std_5'] = df['Air temperature [K]'].rolling(window=5, min_periods=1).std().fillna(0)
df['torque_rolling_max_5'] = df['Torque [Nm]'].rolling(window=5, min_periods=1).max().fillna(0)
df['speed_rolling_min_5'] = df['Rotational speed [rpm]'].rolling(window=5, min_periods=1).min().fillna(0)

# 3d. Cumulative Features (How much total degradation)
df['wear_cumulative'] = df['Tool wear [min]'].cumsum()

# Combine all features (base + engineered)
all_features = base_features + [
    'air_temp_trend_1',
    'proc_temp_trend_1', 
    'torque_trend_1',
    'wear_trend_1',
    'air_temp_velocity_3',
    'wear_velocity_3',
    'air_temp_rolling_std_5',
    'torque_rolling_max_5',
    'speed_rolling_min_5',
    'wear_cumulative'
]

X = df[all_features]
y = df['Machine failure']

print(f"📊 Dataset Shape: {X.shape}")
print(f"📈 Base Features: {len(base_features)}")
print(f"⏱️  Engineered Features: {len(all_features) - len(base_features)}")
print(f"📋 Total Features: {len(all_features)}")

# 4. Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Train the Random Forest
print("\n🤖 Training Random Forest with time-series features...")
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
model.fit(X_train, y_train)

# 6. Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
predictions = model.predict(X_test)

print(f"\n✅ Training Complete!")
print(f"📊 Train Accuracy: {train_score * 100:.2f}%")
print(f"📊 Test Accuracy: {test_score * 100:.2f}%")
print(f"\n📈 Classification Report:")
print(classification_report(y_test, predictions, target_names=['HEALTHY', 'FAILURE']))

# 7. Feature Importance (Show which features matter most)
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔍 Top 10 Most Important Features:")
print(feature_importance.head(10).to_string())

# 8. Save the model + feature metadata
joblib.dump(model, 'machine_model.pkl')
joblib.dump(all_features, 'model_features.pkl')  # Save feature names for prediction
print(f"\n💾 Model saved as 'machine_model.pkl'")
print(f"💾 Features saved as 'model_features.pkl'")