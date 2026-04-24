import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the UCI Dataset
df = pd.read_csv('predictive_maintenance.csv')

# 2. Define Features and Target
features = [
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]'
]
X = df[features]
y = df['Machine failure']

# 3. THE SPLIT (80% Train, 20% Test)
# test_size=0.2 means 20% goes to the "Final Exam" pile
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train on the 80% pile
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training on 8,000 rows...")
model.fit(X_train, y_train)

# 5. THE TEST (Predicting the 20% pile)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)

print(f"✅ Training Complete!")
print(f"📊 Model Accuracy Score: {score * 100:.2f}%")

# 6. Save the refined Brain
joblib.dump(model, 'machine_model.pkl')
