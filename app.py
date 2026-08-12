import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Page Configuration
st.set_page_config(
    page_title="Voyage Analytics - Travel Intelligence Hub",
    page_icon="✈️",
    layout="wide"
)

# ==========================================
# 1. LOAD DATASETS
# ==========================================
@st.cache_data
def load_data():
    flights = pd.read_csv('flights.csv')
    hotels = pd.read_csv('hotels.csv')
    users = pd.read_csv('users.csv')
    
    users.rename(columns={'code': 'userCode', 'name': 'userName'}, inplace=True)
    flights['date'] = pd.to_datetime(flights['date'])
    flights['month'] = flights['date'].dt.month
    flights['dayofweek'] = flights['date'].dt.dayofweek
    
    master = pd.merge(flights, users, on='userCode', how='left')
    return flights, hotels, users, master

# ==========================================
# 2. OPTIMIZED ML MODEL TRAINING & CACHING
# ==========================================
@st.cache_resource
def load_model():
    flights = pd.read_csv('flights.csv')
    users = pd.read_csv('users.csv')
    
    # Subsample 20,000 rows for sub-5 second training on Streamlit Cloud
    if len(flights) > 20000:
        flights = flights.sample(n=20000, random_state=42)
        
    users.rename(columns={'code': 'userCode', 'name': 'userName'}, inplace=True)
    flights['date'] = pd.to_datetime(flights['date'])
    flights['month'] = flights['date'].dt.month
    flights['dayofweek'] = flights['date'].dt.dayofweek
    
    df = pd.merge(flights, users, on='userCode', how='left')
    
    features = ['distance', 'time', 'flightType', 'agency', 'company', 'gender', 'age', 'month', 'dayofweek', 'from', 'to']
    target = 'price'
    
    X = df[features]
    y = df[target]
    
    categorical_cols = ['flightType', 'agency', 'company', 'gender', 'from', 'to']
    numerical_cols = ['distance', 'time', 'age', 'month', 'dayofweek']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # n_jobs=1 prevents thread locking in free container environments
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=1))
    ])
    
    pipeline.fit(X, y)
    return pipeline

# Load data and ML model
flights_df, hotels_df, users_df, master_df = load_data()
model = load_model()

# Header
st.title("✈️ Voyage Analytics Platform")
st.markdown("An end-to-end travel analytics & flight price estimation application.")
st.markdown("---")

# Navigation Tabs
tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "🤖 Flight Price Predictor"])

# ==========================================
# TAB 1: EXECUTIVE DASHBOARD
# ==========================================
with tab1:
    st.header("Executive Summary")
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flight Revenue", f"${flights_df['price'].sum():,.2f}")
    col2.metric("Total Hotel Revenue", f"${hotels_df['total'].sum():,.2f}")
    col3.metric("Total Passengers", f"{users_df['userCode'].nunique():,}")
    col4.metric("Avg Flight Ticket Price", f"${flights_df['price'].mean():.2f}")
    
    st.markdown("---")
    
    # Sidebar Filters
    st.sidebar.header("Filter Options")
    selected_agency = st.sidebar.multiselect(
        "Select Agency", 
        options=master_df['agency'].unique(), 
        default=master_df['agency'].unique()
    )
    selected_class = st.sidebar.multiselect(
        "Select Flight Class", 
        options=master_df['flightType'].unique(), 
        default=master_df['flightType'].unique()
    )
    
    filtered_df = master_df[
        (master_df['agency'].isin(selected_agency)) & 
        (master_df['flightType'].isin(selected_class))
    ]
    
    # Visualizations
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Revenue by Agency")
        agency_rev = filtered_df.groupby('agency')['price'].sum().reset_index()
        fig_agency = px.bar(
            agency_rev, 
            x='agency', 
            y='price', 
            color='agency', 
            text_auto='.2s',
            labels={'agency': 'Travel Agency', 'price': 'Revenue ($)'}
        )
        st.plotly_chart(fig_agency, use_container_width=True)
        
    with col_right:
        st.subheader("Flight Class Distribution")
        class_counts = filtered_df['flightType'].value_counts().reset_index()
        fig_class = px.pie(
            class_counts, 
            names='flightType', 
            values='count', 
            hole=0.4
        )
        st.plotly_chart(fig_class, use_container_width=True)

    st.subheader("Top 10 Busiest Travel Routes")
    filtered_df['route'] = filtered_df['from'] + " ➔ " + filtered_df['to']
    top_routes = filtered_df['route'].value_counts().head(10).reset_index()
    top_routes.columns = ['Route', 'Flight Count']
    fig_routes = px.bar(
        top_routes, 
        x='Flight Count', 
        y='Route', 
        orientation='h', 
        color='Flight Count', 
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_routes, use_container_width=True)

# ==========================================
# TAB 2: ML PREDICTOR
# ==========================================
with tab2:
    st.header("Estimate Flight Ticket Price")
    st.write("Provide flight and passenger details below to generate an ML-powered price estimate.")
    
    cities = sorted(master_df['from'].unique())
    agencies = sorted(master_df['agency'].unique())
    companies = sorted(master_df['company'].unique())
    
    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            origin = st.selectbox("Origin City", cities, index=0)
            destination = st.selectbox("Destination City", cities, index=1 if len(cities) > 1 else 0)
            flight_type = st.selectbox("Flight Class", ['economic', 'premium', 'firstClass'])
            agency = st.selectbox("Agency", agencies)
            distance = st.number_input("Distance (km)", min_value=50.0, max_value=3000.0, value=650.0)
            flight_time = st.number_input("Flight Duration (hours)", min_value=0.5, max_value=12.0, value=1.8)
            
        with col_b:
            company = st.selectbox("Company", companies)
            gender = st.selectbox("Gender", ['male', 'female'])
            age = st.slider("Passenger Age", 18, 70, 30)
            month = st.slider("Travel Month", 1, 12, 6)
            dayofweek = st.selectbox("Day of Week (0=Mon, 6=Sun)", [0, 1, 2, 3, 4, 5, 6], index=2)
            
        submit_btn = st.form_submit_button("Predict Flight Price 🚀")
        
    if submit_btn:
        input_data = pd.DataFrame([{
            'distance': distance,
            'time': flight_time,
            'flightType': flight_type,
            'agency': agency,
            'company': company,
            'gender': gender,
            'age': age,
            'month': month,
            'dayofweek': dayofweek,
            'from': origin,
            'to': destination
        }])
        
        prediction = model.predict(input_data)[0]
        
        st.success(f"### Estimated Flight Price: **${prediction:,.2f}**")
