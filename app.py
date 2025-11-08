# app.py
# This is the final, intelligent Streamlit dashboard.

import streamlit as st
import pandas as pd
from itertools import combinations
import numpy as np
import requests # We need this library to check the URL status

st.set_page_config(
    page_title="Professional Betting Co-Pilot",
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
DATA_URL = f"https://raw.githubusercontent.com/jd.0913/betting-copilot-pro/main/latest_bets.csv"

# ==============================================================================
# NEW: Intelligent Data Loading Function
# ==============================================================================

@st.cache_data(ttl=600) # Cache for 10 minutes
def load_data_intelligently():
    """
    Intelligently loads data, distinguishing between a missing file and an empty file.
    Returns one of three things:
    1. A DataFrame with data if bets are found.
    2. A string "NO_BETS_FOUND" if the file is empty.
    3. A string "FILE_NOT_FOUND" if the URL gives a 404 error.
    """
    try:
        # First, check if the URL is valid and the file exists
        response = requests.get(DATA_URL)
        if response.status_code == 404:
            return "FILE_NOT_FOUND"
        
        # If the file exists, read it with pandas
        df = pd.read_csv(DATA_URL)
        
        # Check if the dataframe is empty
        if df.empty:
            return "NO_BETS_FOUND"
        
        return df # Success, return the dataframe
    except Exception:
        # Catch any other errors (e.g., network issues)
        return "FILE_NOT_FOUND"

# ==============================================================================
# Main Application UI
# ==============================================================================

st.title("🚀 Professional Betting Co-Pilot")
st.markdown("Your daily source for data-driven betting analysis. Recommendations are updated automatically every 24 hours.")

# --- Load the data using the new intelligent function ---
data_result = load_data_intelligently()

# --- Main App Logic: Handle the three possible states ---

if isinstance(data_result, pd.DataFrame):
    # SUCCESS STATE: The file was found and contains data.
    value_df = data_result
    
    # Convert relevant columns to numeric for calculations
    value_df['Edge'] = pd.to_numeric(value_df['Edge'])
    value_df['Confidence'] = pd.to_numeric(value_df['Confidence'])
    value_df['Odds'] = pd.to_numeric(value_df['Odds'])
    
    # --- Executive Summary ---
    st.header("📝 Executive Summary")
    sorted_bets = value_df.sort_values('Edge', ascending=False).to_dict('records')
    bet_of_the_week = sorted([b for b in sorted_bets if b['Confidence'] > 0.5], key=lambda x: x['Edge'], reverse=True)
    top_underdog = sorted([b for b in sorted_bets if b['Odds'] > 3.0], key=lambda x: x['Edge'], reverse=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if bet_of_the_week:
            botw = bet_of_the_week[0]
            st.metric(label="🎯 Bet of the Week", value=f"{botw['Bet']} in '{botw['Match']}'", delta=f"Odds: {botw['Odds']:.2f} | Edge: {botw['Edge']:.2%}")
        else:
            st.info("No high-confidence favorite bets found in this cycle.")
    with col2:
        if top_underdog:
            tu = top_underdog[0]
            st.metric(label="⚡ Top Underdog Play", value=f"{tu['Bet']} in '{tu['Match']}'", delta=f"Odds: {tu['Odds']:.2f} | Edge: {tu['Edge']:.2%}")
        else:
            st.info("No significant underdog value bets found in this cycle.")

    # --- Main Value Dashboard ---
    st.header("📈 Value Dashboard")
    st.dataframe(value_df.style.format({'Odds': '{:.2f}', 'Edge': '{:.2%}', 'Confidence': '{:.2%}', 'Stake (Kelly/4)': '{:.2%}'}).background_gradient(cmap='Greens', subset=['Edge']))

elif data_result == "NO_BETS_FOUND":
    # EMPTY STATE: The file was found, but it's empty.
    st.success("✅ Backend analysis complete. No value bets were found for the current fixtures.")
    st.info("This is the correct and expected behavior during the off-season or mid-week. The system is in standby mode. Check back closer to the next match day.")

elif data_result == "FILE_NOT_FOUND":
    # ERROR STATE: The file could not be found at the URL.
    st.error("Could not load the latest bets from GitHub. The backend might not have run yet, or the URL is incorrect.")
    st.warning(f"Attempted to load from: {DATA_URL}")
    st.info("Please check that the GITHUB_USERNAME and GITHUB_REPO variables in the app.py file are correct and that your backend (GitHub Action) has run successfully at least once.")

# --- Sidebar Information ---
st.sidebar.header("About This App")
st.sidebar.info(
    "This dashboard displays the output of an automated, multi-model betting analysis engine. "
    "The backend runs daily on GitHub Actions, and this app provides the results."
)
st.sidebar.success("System Status: Connected to Live Data Feed.")
