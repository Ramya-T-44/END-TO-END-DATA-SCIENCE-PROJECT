from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    hours = float(data["Hours_Studied"])
    attendance = float(data["Attendance"])
    sleep = float(data["Sleep_Hours"])
    prev = float(data["Previous_Scores"])
    tutoring = float(data["Tutoring_Sessions"])

    features = np.array([[hours, attendance, sleep, prev, tutoring]])
    prediction = model.predict(features)[0]

    result = "Pass ✅" if prediction == 1 else "Fail ❌"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)