from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('prediction_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    rd_spend = float(request.form["rd_spend"])
    admin_spend = float(request.form["admin_spend"])
    marketing_spend = float(request.form["marketing_spend"])

    # Make predictions using the loaded model
    prediction = model.predict([[rd_spend, admin_spend, marketing_spend]])

    output = round(prediction[0], 3)

    return render_template('index.html', prediction='Prediction {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)