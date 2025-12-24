from flask import Flask, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    price = ""

    if request.method == "POST":
        # Create empty input dataframe
        new_house = pd.DataFrame(0, index=[0], columns=columns)

        # Numeric inputs
        new_house['area'] = int(request.form['area'])
        new_house['bedrooms'] = int(request.form['bedrooms'])
        new_house['bathrooms'] = int(request.form['bathrooms'])
        new_house['stories'] = int(request.form['stories'])
        new_house['parking'] = int(request.form['parking'])

        # Binary inputs
        binary_cols = [
            'mainroad', 'guestroom', 'basement',
            'hotwaterheating', 'airconditioning', 'prefarea'
        ]

        for col in binary_cols:
            new_house[col] = int(request.form[col])

        # Furnishing status
        furnishing = request.form['furnishing']
        furnish_col = "furnishingstatus_" + furnishing
        if furnish_col in new_house.columns:
            new_house[furnish_col] = 1

        # Scale and predict
        scaled_data = scaler.transform(new_house)
        price = int(model.predict(scaled_data)[0])

    # IMPORTANT: utf-8 encoding fix
    return open("index.html", encoding="utf-8").read().replace("{{price}}", str(price))


if __name__ == "__main__":
    app.run()
