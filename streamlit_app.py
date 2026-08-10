import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

st.set_page_config(page_title="Voyage Analytics", page_icon="✈️", layout="wide")

# Custom CSS for polished metric cards
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E222D;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2E3440;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✈️ Voyage Analytics: ML Travel Dashboard")

# --- Load Models Safely ---
@st.cache_resource
def load_models():
    flight_model, clf_model = None, None
    flight_encoders, clf_encoders = {}, {}
    
    if os.path.exists('models/flight_price_model.pkl'):
        with open('models/flight_price_model.pkl', 'rb') as f:
            data = pickle.load(f)
            flight_model = data['model']
            flight_encoders['flightType'] = data['le_flightType']
            flight_encoders['agency'] = data['le_agency']
            
    if os.path.exists('models/gender_clf_model.pkl'):
        with open('models/gender_clf_model.pkl', 'rb') as f:
            data = pickle.load(f)
            clf_model = data['model']
            clf_encoders['company'] = data['le_company']
            
    return flight_model, flight_encoders, clf_model, clf_encoders

flight_model, flight_encoders, clf_model, clf_encoders = load_models()

# --- UI Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Flight Price Predictor", 
    "User Gender Classifier", 
    "Hotel Recommendations", 
    "Data Insights",
    "Model Performance (MLOps)"
])

# ---------------------------------------------------------
# Tab 1: Flight Price Predictor
# ---------------------------------------------------------
with tab1:
    st.header("Predict Flight Prices")
    if flight_model:
        col1, col2 = st.columns(2)
        with col1:
            time = st.number_input("Flight Duration (hours)", value=2.0, min_value=0.5, step=0.5)
            distance = st.number_input("Distance (km)", value=500.0, min_value=50.0, step=50.0)
        with col2:
            flight_type = st.selectbox("Flight Type", flight_encoders['flightType'].classes_)
            agency = st.selectbox("Agency", flight_encoders['agency'].classes_)
            
        if st.button("Predict Price", type="primary"):
            f_enc = flight_encoders['flightType'].transform([flight_type])[0]
            a_enc = flight_encoders['agency'].transform([agency])[0]
            prediction = flight_model.predict([[time, distance, f_enc, a_enc]])[0]
            
            st.markdown("---")
            st.metric(label="Estimated Flight Price", value=f"${prediction:,.2f}")
    else:
        st.error("Flight model not found. Run train_models.py first.")

# ---------------------------------------------------------
# Tab 2: Gender Classification
# ---------------------------------------------------------
with tab2:
    st.header("User Gender Classification")
    if clf_model:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("User Age", value=30, min_value=18, max_value=100)
        with col2:
            company = st.selectbox("Company", clf_encoders['company'].classes_)
        
        if st.button("Predict Gender", type="primary"):
            c_enc = clf_encoders['company'].transform([company])[0]
            prediction = clf_model.predict([[age, c_enc]])[0]
            
            st.markdown("---")
            st.metric(label="Predicted Gender Category", value=str(prediction).capitalize())
    else:
        st.error("Classification model not found. Run train_models.py first.")

# ---------------------------------------------------------
# Tab 3: Hotel Recommendations
# ---------------------------------------------------------
with tab3:
    st.header("Top Hotel Recommendations by Destination")
    if os.path.exists('models/hotel_recommendations.csv'):
        recs = pd.read_csv('models/hotel_recommendations.csv')
        places = recs['place'].unique()
        selected_place = st.selectbox("Select Destination", places)
        
        filtered_recs = recs[recs['place'] == selected_place][['name', 'bookings']].reset_index(drop=True)
        filtered_recs.columns = ['Hotel Name', 'Historical Bookings']
        
        # Display clean dataframe without raw index column
        st.dataframe(filtered_recs, hide_index=True, use_container_width=True)
    else:
        st.error("Recommendation data not found. Run train_models.py first.")

# ---------------------------------------------------------
# Tab 4: Data Insights (Dark Theme Plot Fix)
# ---------------------------------------------------------
with tab4:
    st.header("Dataset Visualizations")
    if os.path.exists('data/flights.csv'):
        flights_df = pd.read_csv('data/flights.csv')
        st.subheader("Price Distribution by Agency")
        
        # Apply dark background matching Streamlit theme
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        
        sns.boxplot(x='agency', y='price', data=flights_df, palette='Set2', ax=ax)
        ax.set_xlabel("Agency", color="white")
        ax.set_ylabel("Price ($)", color="white")
        
        st.pyplot(fig)
    else:
        st.warning("Flight dataset missing from data/ folder.")

# ---------------------------------------------------------
# Tab 5: Model Performance & MLOps Metrics
# ---------------------------------------------------------
with tab5:
    st.header("📊 Model Performance & MLOps Tracking")
    st.caption("Key metrics calculated dynamically from testing data")
    
    # 1. Calculate Real Flight Model Metrics
    if flight_model and os.path.exists('data/flights.csv'):
        flights_df = pd.read_csv('data/flights.csv')
        f_type_enc = flight_encoders['flightType'].transform(flights_df['flightType'])
        a_enc = flight_encoders['agency'].transform(flights_df['agency'])
        
        X_test_r = pd.DataFrame({
            'time': flights_df['time'], 
            'distance': flights_df['distance'], 
            'flightType_enc': f_type_enc, 
            'agency_enc': a_enc
        })
        y_true_r = flights_df['price']
        
        y_pred_r = flight_model.predict(X_test_r)
        real_r2 = r2_score(y_true_r, y_pred_r)
        real_mae = mean_absolute_error(y_true_r, y_pred_r)
    else:
        real_r2, real_mae = 0, 0

    # 2. Calculate Real Gender Classifier Metrics
    if clf_model and os.path.exists('data/users.csv'):
        users_df = pd.read_csv('data/users.csv')
        users_df = users_df[users_df['gender'] != 'none'].reset_index(drop=True)
        
        c_enc = clf_encoders['company'].transform(users_df['company'])
        X_test_c = pd.DataFrame({'age': users_df['age'], 'company_enc': c_enc})
        y_true_c = users_df['gender']
        
        y_pred_c = clf_model.predict(X_test_c)
        real_acc = accuracy_score(y_true_c, y_pred_c) * 100
    else:
        real_acc = 0
    
    # 3. Display the Dynamic Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Flight Model R² Score", value=f"{real_r2:.2f}")
    with col2:
        st.metric(label="Flight Model MAE", value=f"${real_mae:.2f}")
    with col3:
        st.metric(label="Gender Classifier Accuracy", value=f"{real_acc:.1f}%")
        
    st.markdown("---")
    st.subheader("Experiment Tracking Details")
    st.json({
        "Experiment Name": "Flight_Price_Prediction",
        "Tracking Tool": "MLflow",
        "Model Type": "RandomForestRegressor",
        "Parameters": {"n_estimators": 10, "random_state": 42},
        "Deployment Framework": "Streamlit Cloud"
    })