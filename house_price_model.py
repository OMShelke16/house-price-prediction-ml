# ==========================================
# House Price Prediction – Final Corrected
# ==========================================

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Step 2: Load dataset
data = pd.read_csv("Housing.csv")

print("Dataset Loaded Successfully")
print(data.head())

# Step 3: Encode categorical features

# Convert yes/no columns to 1/0
binary_cols = [
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'prefarea'
]

for col in binary_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})

# One-hot encode furnishing status
data = pd.get_dummies(data, columns=['furnishingstatus'], drop_first=True)

# Step 4: Feature selection
X = data.drop('price', axis=1)
y = data['price']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Training Completed")

# Step 8: Predictions on test data
y_pred = model.predict(X_test)

# Step 9: Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Evaluation:")
print("R2 Score:", r2)
print("Mean Squared Error:", mse)

# Step 10: Predict price for a new house (SAFE METHOD)

# Create empty dataframe with same columns
new_house = pd.DataFrame(0, index=[0], columns=X.columns)

# Fill known values
new_house['area'] = 3000
new_house['bedrooms'] = 3
new_house['bathrooms'] = 2
new_house['stories'] = 2
new_house['parking'] = 1
new_house['mainroad'] = 1
new_house['guestroom'] = 0
new_house['basement'] = 1
new_house['hotwaterheating'] = 0
new_house['airconditioning'] = 1
new_house['prefarea'] = 0

# Furnishing status example: semi-furnished
if 'furnishingstatus_semi-furnished' in new_house.columns:
    new_house['furnishingstatus_semi-furnished'] = 1

# Scale and predict
new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)

print("\nPredicted House Price:")
print("₹", int(predicted_price[0]))


import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))
