import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Create directory for saving models
os.makedirs('models', exist_ok=True)

# ---------------------------------------------------------
# 1. Regression Model (Flight Prices) & MLFlow Tracking
# ---------------------------------------------------------
print("Training Flight Price Regression Model...")
flights = pd.read_csv('data/flights.csv')

le_flightType = LabelEncoder()
le_agency = LabelEncoder()
flights['flightType_enc'] = le_flightType.fit_transform(flights['flightType'])
flights['agency_enc'] = le_agency.fit_transform(flights['agency'])

# Features: time, distance, flightType, agency | Target: price
X_reg = flights[['time', 'distance', 'flightType_enc', 'agency_enc']]
y_reg = flights['price']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

mlflow.set_experiment("Flight_Price_Prediction")
with mlflow.start_run():
    reg_model = RandomForestRegressor(n_estimators=10, random_state=42)
    reg_model.fit(X_train_r, y_train_r)
    score = reg_model.score(X_test_r, y_test_r)
    
    # MLFlow Tracking
    mlflow.log_param("n_estimators", 10)
    mlflow.log_metric("r2_score", score)
    mlflow.sklearn.log_model(reg_model, "model")
    
    # Save model and encoders
    with open('models/flight_price_model.pkl', 'wb') as f:
        pickle.dump({'model': reg_model, 'le_flightType': le_flightType, 'le_agency': le_agency}, f)

# ---------------------------------------------------------
# 2. Gender Classification Model
# ---------------------------------------------------------
print("Training Gender Classification Model...")
users = pd.read_csv('data/users.csv')

users = users[users['gender'] != 'none'].reset_index(drop=True)

le_company = LabelEncoder()
users['company_enc'] = le_company.fit_transform(users['company'])

# Features: age, company | Target: gender
X_clf = users[['age', 'company_enc']]
y_clf = users['gender']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf_model = RandomForestClassifier(n_estimators=10, random_state=42)
clf_model.fit(X_train_c, y_train_c)

with open('models/gender_clf_model.pkl', 'wb') as f:
    pickle.dump({'model': clf_model, 'le_company': le_company}, f)

# ---------------------------------------------------------
# 3. Travel Recommendation Model (Popularity-based by Place)
# ---------------------------------------------------------
print("Generating Hotel Recommendations...")
hotels = pd.read_csv('data/hotels.csv')

# Recommend top 3 most booked hotels per place
top_hotels = hotels.groupby('place')['name'].value_counts().groupby(level=0, group_keys=False).nlargest(3).reset_index(name='bookings')
top_hotels.to_csv('models/hotel_recommendations.csv', index=False)

print("All models successfully trained and saved in the 'models/' directory.")