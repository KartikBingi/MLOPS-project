import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource
def load_model():
    # 1. Load training data
    flights = pd.read_csv('flights.csv')
    users = pd.read_csv('users.csv')
    
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
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1))
    ])
    
    # Train pipeline once and cache in memory
    pipeline.fit(X, y)
    return pipeline

# Load model pipeline seamlessly
model = load_model()
