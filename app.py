import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Voyage Analytics | Enterprise Travel Intelligence Hub",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS STYLING
# ==========================================
st.markdown("""
<style>
    /* App Background */
    .stApp {
        background-color: #0B0F19;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Executive Header */
    .header-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 28px 36px;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        color: #F8FAFC;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #94A3B8;
        font-size: 1.0rem;
        margin-top: 6px;
        margin-bottom: 12px;
    }
    .status-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* KPI Cards */
    .kpi-card {
        background: #161E2E;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #283548;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }
    .kpi-label {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .kpi-sub {
        color: #38BDF8;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    /* Form and Predictor Section */
    .prediction-card {
        background: linear-gradient(135deg, #1E1B4B 0%, #171026 100%);
        border: 1px solid #4338CA;
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
    }
    .class-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .class-card-featured {
        background: linear-gradient(135deg, #1E1B4B 0%, #2E1065 100%);
        border: 1px solid #6366F1;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }

    /* Tab Label Enhancements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        background-color: #161E2E;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
    }
    
    /* Footer Styling */
    .footer-container {
        border-top: 1px solid #1E293B;
        margin-top: 40px;
        padding-top: 20px;
        color: #64748B;
        font-size: 0.85rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATA INGESTION & CACHING
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
    flights['route'] = flights['from'] + " ➔ " + flights['to']
    
    master = pd.merge(flights, users, on='userCode', how='left')
    return flights, hotels, users, master

# ==========================================
# 2. MODEL TRAINING & PIPELINE CACHING
# ==========================================
@st.cache_resource
def load_model():
    flights = pd.read_csv('flights.csv')
    users = pd.read_csv('users.csv')
    
    # Subsample 25,000 rows for fast load time & CPU efficiency on Streamlit Cloud
    if len(flights) > 25000:
        flights = flights.sample(n=25000, random_state=42)
        
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
    
    # Single-thread execution (n_jobs=1) ensures smooth execution on Streamlit Cloud
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=40, max_depth=12, random_state=42, n_jobs=1))
    ])
    
    pipeline.fit(X, y)
    return pipeline

# Load data and model
flights_df, hotels_df, users_df, master_df = load_data()
model = load_model()

# ==========================================
# EXECUTIVE HEADER BANNER
# ==========================================
st.markdown("""
<div class="header-card">
    <div class="header-title">✈️ Voyage Analytics Platform</div>
    <div class="header-subtitle">Enterprise Data Science & Predictive Pricing Intelligence Hub</div>
    <span class="status-badge">● Production Model Online (RF Regressor v1.2)</span>
</div>
""", unsafe_allow_html=True)

# Main Navigation
tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "🤖 Smart Price Predictor", "📁 Data Insights & Export"])

# PLOTLY GLOBAL THEME HELPER
PLOTLY_THEME = {
    'layout': go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#CBD5E1', family='Inter, sans-serif'),
        xaxis=dict(gridcolor='#1E293B', zerolinecolor='#1E293B'),
        yaxis=dict(gridcolor='#1E293B', zerolinecolor='#1E293B'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
}

# ==========================================
# TAB 1: EXECUTIVE DASHBOARD
# ==========================================
with tab1:
    # Sidebar Filters
    st.sidebar.markdown("### 🎛️ Global Analytics Filters")
    selected_agency = st.sidebar.multiselect(
        "Filter Travel Agency", 
        options=sorted(master_df['agency'].unique()), 
        default=sorted(master_df['agency'].unique())
    )
    selected_class = st.sidebar.multiselect(
        "Filter Flight Class", 
        options=['economic', 'premium', 'firstClass'], 
        default=['economic', 'premium', 'firstClass']
    )
    
    filtered_df = master_df[
        (master_df['agency'].isin(selected_agency)) & 
        (master_df['flightType'].isin(selected_class))
    ]
    
    # KPI Grid
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    total_flight_rev = filtered_df['price'].sum()
    total_hotel_rev = hotels_df['total'].sum()
    total_passengers = users_df['userCode'].nunique()
    avg_ticket = filtered_df['price'].mean() if len(filtered_df) > 0 else 0
    
    kpi_col1.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Flight Revenue</div>
        <div class="kpi-value">${total_flight_rev:,.2f}</div>
        <div class="kpi-sub">Total Ticket Volume</div>
    </div>
    """, unsafe_allow_html=True)
    
    kpi_col2.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Hotel Bookings</div>
        <div class="kpi-value">${total_hotel_rev:,.2f}</div>
        <div class="kpi-sub">Cross-sell Revenue</div>
    </div>
    """, unsafe_allow_html=True)
    
    kpi_col3.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Active Users</div>
        <div class="kpi-value">{total_passengers:,}</div>
        <div class="kpi-sub">Corporate & Individual</div>
    </div>
    """, unsafe_allow_html=True)
    
    kpi_col4.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Average Ticket Price</div>
        <div class="kpi-value">${avg_ticket:,.2f}</div>
        <div class="kpi-sub">Across All Agencies</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts Row 1
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Agency Revenue Contribution")
        agency_rev = filtered_df.groupby('agency')['price'].sum().reset_index()
        fig_agency = px.bar(
            agency_rev, 
            x='agency', 
            y='price', 
            color='agency',
            text_auto='.2s',
            color_discrete_sequence=['#38BDF8', '#818CF8', '#F472B6']
        )
        fig_agency.update_layout(PLOTLY_THEME['layout'], showlegend=False, yaxis_title="Revenue ($)", xaxis_title="Agency")
        st.plotly_chart(fig_agency, use_container_width=True)
        
    with c2:
        st.subheader("Flight Class Market Share")
        class_counts = filtered_df['flightType'].value_counts().reset_index()
        fig_class = px.pie(
            class_counts, 
            names='flightType', 
            values='count', 
            hole=0.55,
            color_discrete_sequence=['#6366F1', '#38BDF8', '#EC4899']
        )
        fig_class.update_layout(PLOTLY_THEME['layout'])
        fig_class.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_class, use_container_width=True)

    # Charts Row 2
    st.subheader("Top 10 High-Volume Flight Routes")
    top_routes = filtered_df['route'].value_counts().head(10).reset_index()
    top_routes.columns = ['Route', 'Flight Count']
    top_routes = top_routes.sort_values(by='Flight Count', ascending=True)
    
    fig_routes = px.bar(
        top_routes, 
        x='Flight Count', 
        y='Route', 
        orientation='h', 
        color='Flight Count',
        color_continuous_scale='Viridis'
    )
    fig_routes.update_layout(PLOTLY_THEME['layout'], coloraxis_showscale=False)
    st.plotly_chart(fig_routes, use_container_width=True)

# ==========================================
# TAB 2: SMART PRICE PREDICTOR
# ==========================================
with tab2:
    st.header("🤖 Machine Learning Price Estimator")
    st.write("Input travel dimensions to generate real-time pricing estimates across flight classes.")
    
    cities = sorted(master_df['from'].unique())
    agencies = sorted(master_df['agency'].unique())
    companies = sorted(master_df['company'].unique())
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        origin = st.selectbox("Origin City", cities, index=0)
        destination = st.selectbox("Destination City", cities, index=1 if len(cities) > 1 else 0)
        
        # Auto-fill route distance & duration defaults from historical data
        route_match = master_df[(master_df['from'] == origin) & (master_df['to'] == destination)]
        default_dist = float(route_match['distance'].iloc[0]) if len(route_match) > 0 else 650.0
        default_time = float(route_match['time'].iloc[0]) if len(route_match) > 0 else 1.8
        
        agency = st.selectbox("Travel Agency", agencies)
        distance = st.number_input("Distance (km)", min_value=50.0, max_value=3000.0, value=default_dist)
        flight_time = st.number_input("Flight Duration (hours)", min_value=0.5, max_value=12.0, value=default_time)
        
    with col_input2:
        company = st.selectbox("Corporate Sponsor / Company", companies)
        gender = st.selectbox("Passenger Gender", ['male', 'female'])
        age = st.slider("Passenger Age", 18, 75, 32)
        month = st.slider("Travel Month", 1, 12, 7)
        dayofweek = st.select_slider("Day of Departure", options=[0, 1, 2, 3, 4, 5, 6], value=2, format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])

    st.markdown("---")
    predict_clicked = st.button("Calculate Multi-Class Price Estimates 🚀", use_container_width=True, type="primary")
    
    if predict_clicked:
        # Predict across all three classes simultaneously
        class_predictions = {}
        for f_class in ['economic', 'premium', 'firstClass']:
            row_df = pd.DataFrame([{
                'distance': distance,
                'time': flight_time,
                'flightType': f_class,
                'agency': agency,
                'company': company,
                'gender': gender,
                'age': age,
                'month': month,
                'dayofweek': dayofweek,
                'from': origin,
                'to': destination
            }])
            class_predictions[f_class] = model.predict(row_df)[0]
            
        st.subheader("Price Breakdown Comparison")
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.markdown(f"""
            <div class="class-card">
                <div style="color: #94A3B8; font-weight:600;">ECONOMIC CLASS</div>
                <div style="color: #38BDF8; font-size: 2.2rem; font-weight:800; margin:10px 0;">${class_predictions['economic']:,.2f}</div>
                <div style="color: #64748B; font-size: 0.8rem;">Standard Cabin Seating</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown(f"""
            <div class="class-card-featured">
                <div style="color: #A5B4FC; font-weight:600;">PREMIUM CLASS</div>
                <div style="color: #FFFFFF; font-size: 2.2rem; font-weight:800; margin:10px 0;">${class_predictions['premium']:,.2f}</div>
                <div style="color: #C7D2FE; font-size: 0.8rem;">Priority Check-in & Legroom</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col3:
            st.markdown(f"""
            <div class="class-card">
                <div style="color: #F472B6; font-weight:600;">FIRST CLASS</div>
                <div style="color: #F472B6; font-size: 2.2rem; font-weight:800; margin:10px 0;">${class_predictions['firstClass']:,.2f}</div>
                <div style="color: #64748B; font-size: 0.8rem;">Full Service Luxury Lounge</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 3: DATA INSIGHTS & EXPORT
# ==========================================
with tab3:
    st.header("📁 Data Inspection & Report Export")
    st.write("Explore filtered travel transaction records and export clean summary CSV reports.")
    
    st.dataframe(
        filtered_df[['travelCode', 'userName', 'company', 'agency', 'from', 'to', 'flightType', 'price', 'date']], 
        use_container_width=True,
        height=350
    )
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Filtered Flight Data (CSV)",
            data=csv_data,
            file_name="filtered_flight_analytics.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl2:
        agency_summary = filtered_df.groupby(['agency', 'flightType'])['price'].agg(['count', 'mean', 'sum']).reset_index()
        csv_summary = agency_summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Export Agency Aggregation Summary (CSV)",
            data=csv_summary,
            file_name="agency_summary_report.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer-container">
    Voyage Analytics Intelligence Platform | Final Internship Capstone Project<br>
    <b>Tech Stack:</b> Python 3.12 | Streamlit | Scikit-Learn (Random Forest) | Plotly | Pandas
</div>
""", unsafe_allow_html=True)