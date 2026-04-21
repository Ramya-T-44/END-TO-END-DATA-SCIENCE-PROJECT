import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ✅ FIX: use comma separator
data = pd.read_csv("student.csv", sep=',')

# 🔧 FIX if still single column
if len(data.columns) == 1:
    data = data.iloc[:,0].str.split(',', expand=True)
    data.columns = [
        "Hours_Studied","Attendance","Parental_Involvement",
        "Access_to_Resources","Extracurricular_Activities",
        "Sleep_Hours","Previous_Scores","Motivation_Level",
        "Internet_Access","Tutoring_Sessions","Family_Income",
        "Teacher_Quality","School_Type","Peer_Influence",
        "Physical_Activity","Learning_Disabilities",
        "Parental_Education_Level","Distance_from_Home",
        "Gender","Exam_Score"
    ]

print(data.columns)

# ✅ SELECT CORRECT FEATURES
X = data[['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions']]
y = data['Exam_Score']

# Convert to Pass/Fail
y = y.apply(lambda x: 1 if int(x) >= 60 else 0)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc*100:.2f}%")

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")