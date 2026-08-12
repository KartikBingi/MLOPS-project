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
    page_title="Voyage Analytics | Enterprise Travel & Expense Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS STYLING (MATCHING REFERENCE UI)
# ==========================================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0A0D14;
        color: #F1F5F9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #111827 0%, #1E1B4B 50%, #0F172A 100%);
        padding: 30px 36px;
        border-radius: 18px;
        border: 1px solid #1E293B;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        color: #FFFFFF;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin: 0;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 8px;
        margin-bottom: 12px;
        font-weight: 400;
    }
    
    /* Executive Metric Box */
    .kpi-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #6366F1;
        transform: translateY(-2px);
    }
    .kpi-label {
        color: #94A3B8;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .kpi-value {
        color: #F8FAFC;
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 6px;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .kpi-badge-purple {
        background-color: rgba(139, 92, 246, 0.15);
        color: #C084FC;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .kpi-badge-amber {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .kpi-badge-rose {
        background-color: rgba(244, 63, 94, 0.15);
        color: #FB7185;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .kpi-badge-green {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Custom Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 22px;
        border-radius: 10px;
        background-color: #111827;
        color: #94A3B8;
        font-weight: 600;
        border: 1px solid #1F2937;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
        color: #FFFFFF !important;
        border-color: #6366F1 !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
    }

    /* Predictor Card */
    .pred-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .pred-card-featured {
        background: linear-gradient(135deg, #1E1B4B 0%, #311042 100%);
        border: 1px solid #818CF8;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.25);
    }

    /* Footer */
    .footer-container {
        border-top: 1px solid #1E293B;
        margin-top: 50px;
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
# 2. MODEL TRAINING & CACHING
# ==========================================
@st.cache_resource
def load_model():
    flights = pd.read_csv('flights.csv')
    users = pd.read_csv('users.csv')
    
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
# HERO HEADER BANNER
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Voyage Travel & Expense Analytics</div>
    <div class="hero-subtitle">Diagnostic analytics platform for enterprise mobility to identify expenditure drivers, track booking behaviors, and optimize travel procurement.</div>
    <span style="background-color: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-block;">● Production Engine Active (Scikit-Learn ML Core v1.2)</span>
</div>
""", unsafe_allow_html=True)

# Compute Top-Level Metrics
total_spend = flights_df['price'].sum() + hotels_df['total'].sum()
total_users = users_df['userCode'].nunique()
hotel_attach_rate = (hotels_df['travelCode'].nunique() / flights_df['travelCode'].nunique()) * 100
first_class_rate = (flights_df['flightType'] == 'firstClass').mean() * 100

# Metric Grid
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

m_col1.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Total Travel Spend</div>
    <div class="kpi-value">${total_spend / 1e6:,.1f}M</div>
    <span class="kpi-badge-purple">👤 ${total_spend / total_users:,.0f} Avg / Traveler</span>
</div>
""", unsafe_allow_html=True)

m_col2.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Active Cohort Sample</div>
    <div class="kpi-value">{total_users:,}</div>
    <span class="kpi-badge-amber">👥 Total Analyzed Employees</span>
</div>
""", unsafe_allow_html=True)

m_col3.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Hotel Attach Rate</div>
    <div class="kpi-value">{hotel_attach_rate:.1f}%</div>
    <span class="kpi-badge-rose">🏨 {hotels_df['travelCode'].nunique():,} Linked Stays</span>
</div>
""", unsafe_allow_html=True)

m_col4.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">First Class Tier Share</div>
    <div class="kpi-value">{first_class_rate:.1f}%</div>
    <span class="kpi-badge-green">✈️ {(flights_df['flightType'] == 'firstClass').sum():,} Executive Tickets</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Executive Insights", 
    "🏢 Corporate & Agency Drivers", 
    "🏨 Hotel & Stay Analytics", 
    "🎯 Dynamic Rate Predictor", 
    "🔍 Data Explorer"
])

# Global Plotly Dark Theme
PLOTLY_THEME = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#CBD5E1', family='Inter, sans-serif'),
    xaxis=dict(gridcolor='#1F2937', zerolinecolor='#1F2937'),
    yaxis=dict(gridcolor='#1F2937', zerolinecolor='#1F2937'),
    margin=dict(l=20, r=20, t=35, b=20)
)

# Sidebar Filter
st.sidebar.markdown("### 🎛️ Analytics Controls")
selected_agencies = st.sidebar.multiselect(
    "Agency Filter", 
    options=sorted(master_df['agency'].unique()), 
    default=sorted(master_df['agency'].unique())
)
selected_classes = st.sidebar.multiselect(
    "Flight Class Filter", 
    options=['economic', 'premium', 'firstClass'], 
    default=['economic', 'premium', 'firstClass']
)

filtered_df = master_df[
    (master_df['agency'].isin(selected_agencies)) & 
    (master_df['flightType'].isin(selected_classes))
]

# ==========================================
# TAB 1: EXECUTIVE INSIGHTS
# ==========================================
with tab1:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Agency Revenue Generation")
        agency_rev = filtered_df.groupby('agency')['price'].sum().reset_index()
        fig_agency = px.bar(
            agency_rev, 
            x='agency', 
            y='price', 
            color='agency',
            text_auto='.2s',
            color_discrete_sequence=['#6366F1', '#38BDF8', '#EC4899']
        )
        fig_agency.update_layout(PLOTLY_THEME, showlegend=False, yaxis_title="Flight Spend ($)", xaxis_title="Agency")
        st.plotly_chart(fig_agency, use_container_width=True)
        
    with col_chart2:
        st.subheader("Cabin Class Volume Distribution")
        class_counts = filtered_df['flightType'].value_counts().reset_index()
        fig_class = px.pie(
            class_counts, 
            names='flightType', 
            values='count', 
            hole=0.55,
            color_discrete_sequence=['#818CF8', '#38BDF8', '#F472B6']
        )
        fig_class.update_layout(PLOTLY_THEME)
        fig_class.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_class, use_container_width=True)

    st.subheader("Top 10 High-Volume Travel Routes")
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
    fig_routes.update_layout(PLOTLY_THEME, coloraxis_showscale=False)
    st.plotly_chart(fig_routes, use_container_width=True)

# ==========================================
# TAB 2: CORPORATE & AGENCY DRIVERS
# ==========================================
with tab2:
    st.header("Corporate Travel & Expenditure Patterns")
    
    col_corp1, col_corp2 = st.columns(2)
    
    with col_corp1:
        st.subheader("Top Companies by Flight Expenditure")
        comp_spend = filtered_df.groupby('company')['price'].sum().reset_index().sort_values(by='price', ascending=False).head(10)
        fig_comp = px.bar(
            comp_spend, 
            x='price', 
            y='company', 
            orientation='h',
            text_auto='.2s',
            color='price',
            color_continuous_scale='Cividis'
        )
        fig_comp.update_layout(PLOTLY_THEME, coloraxis_showscale=False, xaxis_title="Total Spend ($)", yaxis_title="Company")
        st.plotly_chart(fig_comp, use_container_width=True)
        
    with col_corp2:
        st.subheader("Average Ticket Price by Agency & Class")
        agency_class_avg = filtered_df.groupby(['agency', 'flightType'])['price'].mean().reset_index()
        fig_heatmap = px.bar(
            agency_class_avg, 
            x='agency', 
            y='price', 
            color='flightType', 
            barmode='group',
            color_discrete_sequence=['#34D399', '#38BDF8', '#818CF8']
        )
        fig_heatmap.update_layout(PLOTLY_THEME, yaxis_title="Avg Price ($)", xaxis_title="Agency")
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ==========================================
# TAB 3: HOTEL & STAY ANALYTICS
# ==========================================
with tab3:
    st.header("Hotel Booking & Accommodations Analytics")
    
    col_h1, col_h2 = st.columns(2)
    
    with col_h1:
        st.subheader("Top Hotel Destinations by Revenue")
        hotel_place = hotels_df.groupby('place')['total'].sum().reset_index().sort_values(by='total', ascending=False)
        fig_hotel_place = px.bar(
            hotel_place, 
            x='place', 
            y='total', 
            color='total',
            text_auto='.2s',
            color_continuous_scale='Blues'
        )
        fig_hotel_place.update_layout(PLOTLY_THEME, coloraxis_showscale=False, yaxis_title="Hotel Spend ($)", xaxis_title="Destination Place")
        st.plotly_chart(fig_hotel_place, use_container_width=True)
        
    with col_h2:
        st.subheader("Stay Duration (Days) Distribution")
        days_count = hotels_df['days'].value_counts().reset_index()
        days_count.columns = ['days', 'count']
        fig_days = px.bar(
            days_count, 
            x='days', 
            y='count',
            text_auto=True,
            color_discrete_sequence=['#F472B6']
        )
        fig_days.update_layout(PLOTLY_THEME, xaxis_title="Duration of Stay (Days)", yaxis_title="Number of Bookings")
        st.plotly_chart(fig_days, use_container_width=True)

# ==========================================
# TAB 4: DYNAMIC RATE PREDICTOR
# ==========================================
with tab4:
    st.header("🎯 Machine Learning Flight Rate Predictor")
    st.write("Configure travel parameters to generate price predictions across cabin classes.")
    
    cities = sorted(master_df['from'].unique())
    agencies = sorted(master_df['agency'].unique())
    companies = sorted(master_df['company'].unique())
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        origin = st.selectbox("Origin City", cities, index=0)
        destination = st.selectbox("Destination City", cities, index=1 if len(cities) > 1 else 0)
        
        route_match = master_df[(master_df['from'] == origin) & (master_df['to'] == destination)]
        default_dist = float(route_match['distance'].iloc[0]) if len(route_match) > 0 else 650.0
        default_time = float(route_match['time'].iloc[0]) if len(route_match) > 0 else 1.8
        
        agency = st.selectbox("Travel Agency", agencies)
        distance = st.number_input("Distance (km)", min_value=50.0, max_value=3000.0, value=default_dist)
        flight_time = st.number_input("Flight Duration (hours)", min_value=0.5, max_value=12.0, value=default_time)
        
    with col_p2:
        company = st.selectbox("Corporate Sponsor", companies)
        gender = st.selectbox("Passenger Gender", ['male', 'female'])
        age = st.slider("Passenger Age", 18, 75, 32)
        month = st.slider("Travel Month", 1, 12, 7)
        dayofweek = st.select_slider("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], value=2, format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Calculate Rate Estimates 🚀", use_container_width=True, type="primary"):
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
            
        st.markdown("### Estimated Ticket Rates")
        rc1, rc2, rc3 = st.columns(3)
        
        with rc1:
            st.markdown(f"""
            <div class="pred-card">
                <div style="color: #94A3B8; font-weight:700;">ECONOMIC</div>
                <div style="color: #38BDF8; font-size: 2.2rem; font-weight:800; margin:8px 0;">${class_predictions['economic']:,.2f}</div>
                <div style="color: #64748B; font-size: 0.8rem;">Standard Cabin Seating</div>
            </div>
            """, unsafe_allow_html=True)
            
        with rc2:
            st.markdown(f"""
            <div class="pred-card-featured">
                <div style="color: #A5B4FC; font-weight:700;">PREMIUM</div>
                <div style="color: #FFFFFF; font-size: 2.2rem; font-weight:800; margin:8px 0;">${class_predictions['premium']:,.2f}</div>
                <div style="color: #C7D2FE; font-size: 0.8rem;">Priority Check-in & Extra Legroom</div>
            </div>
            """, unsafe_allow_html=True)
            
        with rc3:
            st.markdown(f"""
            <div class="pred-card">
                <div style="color: #F472B6; font-weight:700;">FIRST CLASS</div>
                <div style="color: #F472B6; font-size: 2.2rem; font-weight:800; margin:8px 0;">${class_predictions['firstClass']:,.2f}</div>
                <div style="color: #64748B; font-size: 0.8rem;">Full Luxury Lounge Access</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 5: DATA EXPLORER
# ==========================================
with tab5:
    st.header("🔍 Interactive Data Explorer")
    st.dataframe(
        filtered_df[['travelCode', 'userName', 'company', 'agency', 'from', 'to', 'flightType', 'price', 'date']], 
        use_container_width=True,
        height=380
    )
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="📥 Download Filtered Dataset (CSV)",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name="voyage_filtered_analytics.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_d2:
        summary_df = filtered_df.groupby(['agency', 'flightType'])['price'].agg(['count', 'mean', 'sum']).reset_index()
        st.download_button(
            label="📊 Download Agency Summary Report (CSV)",
            data=summary_df.to_csv(index=False).encode('utf-8'),
            file_name="agency_summary_report.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer-container">
    Voyage Analytics Intelligence Platform | Final Internship Group Capstone Project<br>
    <b>Built with:</b> Python 3.12 • Streamlit • Scikit-Learn (Random Forest) • Plotly • Pandas
</div>
""", unsafe_allow_html=True)