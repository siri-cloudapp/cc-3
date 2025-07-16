from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "🚢 Titanic Survival ML Model is LIVE!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[
        data['Pclass'],
        data['Age'],
        data['SibSp'],
        data['Fare']
    ]])
    prediction = model.predict(features)
    return jsonify({'survived': int(prediction[0])})
