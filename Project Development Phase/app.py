from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("HDI.pkl", "rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict')
def predict_page():
    return render_template("indexnew.html")

@app.route('/predict', methods=['POST'])
def prediction():

    life = float(request.form['life_expectancy'])
    expected = float(request.form['expected_schooling'])
    mean = float(request.form['mean_schooling'])
    gni = float(request.form['gni'])

    data = np.array([[life, expected, mean, gni]])

    result = model.predict(data)

    prediction = round(float(result[0]), 2)

    return render_template(
        "results.html",
        prediction_text=f"Predicted HDI Rank (2021): {prediction}"
    )

if __name__ == "__main__":
    app.run(debug=True)