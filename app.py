# app.py
# This is the "World-Class Cockpit" for your Betting Co-Pilot.
# It is a complete, multi-page, interactive Streamlit application.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Betting Co-Pilot v14.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# IMPORTANT: CONFIGURE YOUR GITHUB REPOSITORY DETAILS HERE
# ==============================================================================
GITHUB_USERNAME = "jd0913"
GITHUB_REPO = "betting-copilot-pro"
# ==============================================================================

# Construct the full URL to the raw CSV file in your GitHub repository
DATA_URL = f"https://raw.githubusercontent.com/jd0913/betting-copilot-pro/main/latest_bets.csv"

# ==============================================================================
# Data Loading Function
# ==============================================================================

@st.cache_data(ttl=600) # Cache the data for 10 minutes
def load_data():
    """
    Intelligently loads data from the GitHub repo.
    Returns a DataFrame or a string indicating the status.
    """
    try:
        # Use requests to check for file existence first
        import requests
        response = requests.get(DATA_URL)
        if response.status_code == 404:
            return "FILE_NOT_FOUND"
        
        df = pd.read_csv(DATA_URL)
        if df.empty:
            return "NO_BETS_FOUND"
        
        # Ensure correct data types for filtering
        df['Edge'] = pd.to_numeric(df['Edge'])
        df['Confidence'] = pd.to_numeric(df['Confidence'])
        df['Odds'] = pd.to_numeric(df['Odds'])
        return df
    except Exception:
        return "FILE_NOT_FOUND"

# ==============================================================================
# Main Application Logic
# ==============================================================================

def main_dashboard():
    """
    The main dashboard page of the application.
    """
    st.title("🚀 Betting Co-Pilot")
    st.markdown("Your daily source for data-driven betting analysis. Recommendations are updated automatically by the backend engine.")

    data_result = load_data()

    if isinstance(data_result, pd.DataFrame):
        value_df = data_result
        
        # --- Sidebar Filtering Controls ---
        st.sidebar.header("Dashboard Controls")
        min_edge = st.sidebar.slider("Filter by Minimum Edge (%)", 0, 100, 5) / 100.0
        min_confidence = st.sidebar.slider("Filter by Minimum Confidence (%)", 0, 100, 30) / 100.0
        odds_range = st.sidebar.slider("Filter by Odds Range", 1.0, 15.0, (1.0, 15.0))
        
        filtered_df = value_df[
            (value_df['Edge'] >= min_edge) &
            (value_df['Confidence'] >= min_confidence) &
            (value_df['Odds'] >= odds_range[0]) &
            (value_df['Odds'] <= odds_range[1])
        ]

        # --- KPI Row ---
        st.header("Key Performance Indicators")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value Bets Found", len(filtered_df))
        if not filtered_df.empty:
            col2.metric("Highest Edge Available", f"{filtered_df['Edge'].max():.2%}")
            col3.metric("Most Confident Bet", f"{filtered_df['Confidence'].max():.2%}")
        
        # --- Main Value Dashboard ---
        st.header("📈 Value Dashboard")
        if not filtered_df.empty:
            st.dataframe(filtered_df.style.format({
                'Odds': '{:.2f}', 
                'Edge': '{:.2%}', 
                'Confidence': '{:.2%}',
                'Stake (Kelly/4)': '{:.2%}'
            }).background_gradient(cmap='Greens', subset=['Edge']))
        else:
            st.info("No bets match the current filter criteria.")

    elif data_result == "NO_BETS_FOUND":
        st.success("✅ Backend analysis complete. No value bets were found for the current fixtures.")
        st.info("This is normal during the off-season or mid-week. The system is in standby mode.")

    elif data_result == "FILE_NOT_FOUND":
        st.error("Could not load the latest bets from GitHub.")
        st.warning(f"Attempted to load from: {DATA_URL}")
        st.info("Please check your GitHub configuration and ensure the backend has run successfully at least once.")

def market_map_page():
    """
    A page to visualize the entire market landscape.
    """
    st.title("🗺️ Market Map")
    st.markdown("Visualize your model's edge against the bookmaker's implied odds.")
    
    data_result = load_data()
    
    if isinstance(data_result, pd.DataFrame):
        value_df = data_result
        value_df['Implied_Prob'] = 1 / value_df['Odds']
        
        fig = go.Figure()
        
        # Add the "no value" line
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No Value Line', line=dict(color='red', dash='dash')))
        
        # Add the scatter plot of bets
        fig.add_trace(go.Scatter(
            x=value_df['Implied_Prob'],
            y=value_df['Confidence'],
            mode='markers',
            marker=dict(
                size=10,
                color=value_df['Edge'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Edge')
            ),
            text=value_df['Match'] + "<br>" + value_df['Bet'],
            name='Value Bets'
        ))
        
        fig.update_layout(
            title='Model Confidence vs. Market Implied Probability',
            xaxis_title='Bookmaker Implied Probability (1 / Odds)',
            yaxis_title='Co-Pilot Model Confidence',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("All profitable bets appear above the red 'No Value Line'. The brighter the point, the higher the edge.")
        
    else:
        st.info("No data loaded to display the market map.")

def bet_tracker_page():
    """
    A page for the personal bet slip and tracker.
    """
    st.title("🎟️ Personal Bet Slip & Tracker")
    
    if 'bet_slip' not in st.session_state:
        st.session_state.bet_slip = []
        
    data_result = load_data()
    
    if isinstance(data_result, pd.DataFrame):
        st.subheader("Add Recommendations to Your Slip")
        
        # Create a unique key for each bet
        value_df['key'] = value_df['Match'] + "_" + value_df['Bet']
        
        for i, row in value_df.iterrows():
            # Check if the bet is already in the slip
            is_in_slip = any(bet['key'] == row['key'] for bet in st.session_state.bet_slip)
            
            if not is_in_slip:
                if st.checkbox(f"Add to slip: {row['Match']} - {row['Bet']} @ {row['Odds']:.2f}", key=row['key']):
                    st.session_state.bet_slip.append(row.to_dict())
    
    st.subheader("My Current Bet Slip")
    if st.session_state.bet_slip:
        slip_df = pd.DataFrame(st.session_state.bet_slip)
        st.dataframe(slip_df[['Match', 'Bet', 'Odds', 'Edge', 'Confidence']].style.format({'Odds': '{:.2f}', 'Edge': '{:.2%}', 'Confidence': '{:.2%}'}))
        
        if st.button("Clear Bet Slip"):
            st.session_state.bet_slip = []
            st.rerun()
    else:
        st.info("Your bet slip is empty. Select bets from the dashboard to add them.")

def about_page():
    """
    A page explaining how the Co-Pilot works.
    """
    st.title("📖 About the Co-Pilot")
    st.markdown("""
    The Betting Co-Pilot is a sophisticated, multi-model system designed to find profitable betting opportunities (value) in football markets. It operates on a professional architecture, separating its heavy-duty backend analysis from this lightweight, interactive frontend.

    ### The Backend Engine (Runs on GitHub Actions)
    - **Data Acquisition:** Every day, the backend automatically downloads the latest historical results and upcoming fixtures from `football-data.co.uk`.
    - **Model Training:** It trains a portfolio of independent analytical models from scratch:
        - **Model Alpha (Power-Form):** A robust model based on the Elo rating system to gauge the long-term strength of teams, combined with their recent goal-difference form.
        - **Model Bravo (Goal-Expectancy):** A statistical model based on the Poisson distribution that predicts the likely number of goals each team will score.
        - **Model Charlie (Draw Specialist):** A specialized model trained specifically to identify the characteristics of matches that are likely to end in a draw.
    - **Analysis & Output:** The backend analyzes all upcoming fixtures with its portfolio of models, identifies bets where its calculated probability is higher than the bookmaker's (finding "edge"), and saves these recommendations to a file (`latest_bets.csv`) in the public GitHub repository.

    ### This Dashboard (Runs on Streamlit)
    - **Data Loading:** This app's only job is to read the `latest_bets.csv` file produced by the backend. This makes the app incredibly fast and efficient.
    - **Visualization & Interaction:** It then displays this data through various analytical tools like the Value Dashboard, the Market Map, and the Bet Tracker, allowing you, the operator, to make informed decisions.
    """)

# ==============================================================================
# Sidebar Navigation
# ==============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Dashboard", "Market Map", "Bet Tracker", "About the Co-Pilot"])

if page == "Main Dashboard":
    main_dashboard()
elif page == "Market Map":
    market_map_page()
elif page == "Bet Tracker":
    bet_tracker_page()
elif page == "About the Co-Pilot":
    about_page()

st.sidebar.info(
    "This dashboard reads data updated daily by an automated backend. "
    "If the dashboard is empty, the backend has correctly determined there are no value bets for the current fixtures."
)
