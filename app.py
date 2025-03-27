from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load trained model
model = joblib.load("rf_model.pkl")

# Feature names based on dataset
FEATURES = ["account length", "total day minutes", "total eve minutes", 
            "total night minutes", "total intl charge", "customer service calls"]

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data from request
        data = {feature: float(request.form[feature]) for feature in FEATURES}

        # Convert input data to NumPy array
        input_data = np.array([list(data.values())]).reshape(1, -1)

        # Predict churn (1 = churn, 0 = no churn)
        prediction = model.predict(input_data)[0]
        result = "Churn" if prediction == 1 else "No Churn"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400  # Return error with 400 status

if __name__ == "__main__":
    app.run(debug=True)
