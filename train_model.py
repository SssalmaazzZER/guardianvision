import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle 

# 1. Load the Data
print("⏳ Loading data from 'training_data.csv'...")
try:
    df = pd.read_csv('training_data.csv')
except FileNotFoundError:
    print("❌ Error: File not found. Did you name it correctly?")
    exit()

# 2. Separate Features (Coordinates) and Target (Class Name)
X = df.drop('class', axis=1) # The input (x, y, z...)
y = df['class']              # The output (Walking, Sitting...)

# 3. Split Data (Train on 70%, Test on 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# 4. Train the Brain (Random Forest)
print("🧠 Training the model... (This usually takes < 5 seconds)")
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
model = pipeline.fit(X_train, y_train)

# 5. Test Accuracy
y_predict = model.predict(X_test)
score = accuracy_score(y_test, y_predict)
print("------------------------------------------------")
print(f"✅ Model Accuracy: {score*100:.2f}%")
print("------------------------------------------------")

# 6. Save the Brain
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("💾 Saved model to 'model.pkl'")
print("🚀 Ready for Phase 3: Run your main app!")