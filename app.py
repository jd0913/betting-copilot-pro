# app.py
# This is your fast, lightweight Streamlit dashboard.
# Its only job is to read and display the results from the backend.

import streamlit as st
import pandas as pd
from itertools import combinations
import numpy as np

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Professional Betting Co-Pilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# IMPORTANT: CONFIGURE YOUR GITHUB REPOSITORY DETAILS HERE
# ==============================================================================
# Replace "YOUR_USERNAME" with your actual GitHub username.
# The GITHUB_REPO should match the name of the repository you created.
GITHUB_USERNAME = "jd0913"
GITHUB_REPO = "betting-copilot-pro"
# ==============================================================================

# Construct the full URL to the raw CSV file in your GitHub repository
DATA_URL = f"https://raw.githubusercontent.com/jd0913/betting-copilot-pro/main/latest_bets.csv"

# ==============================================================================
# Data Loading Function
# ==============================================================================

@st.cache_data(ttl=600) # Cache the data for 10 minutes to avoid re-downloading on every interaction
def load_data():
    """
    Reads the latest betting recommendations from your public GitHub repo.
    Returns an empty DataFrame if the file is not found or an error occurs.
    """
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception as e:
        # This error is common on the first run before the backend has created the file.
        st.error(f"Could not load the latest bets from GitHub. The backend might not have run yet, or the URL is incorrect.")
        st.info(f"Attempted to load from: {DATA_URL}")
        return pd.DataFrame()

# ==============================================================================
# Main Application UI
# ==============================================================================

st.title("🚀 Professional Betting Co-Pilot")
st.markdown("Your daily source for data-driven betting analysis. Recommendations are updated automatically every 24 hours.")

# --- Load the data ---
value_df = load_data()

if not value_df.empty:
    # Convert relevant columns to numeric for calculations
    value_df['Edge'] = pd.to_numeric(value_df['Edge'])
    value_df['Confidence'] = pd.to_numeric(value_df['Confidence'])
    value_df['Odds'] = pd.to_numeric(value_df['Odds'])
    
    # --- Executive Summary ---
    st.header("📝 Executive Summary")
    sorted_bets = value_df.sort_values('Edge', ascending=False).to_dict('records')
    
    # Find the best high-confidence bet
    bet_of_the_week = sorted([b for b in sorted_bets if b['Confidence'] > 0.5], key=lambda x: x['Edge'], reverse=True)
    # Find the best high-odds bet
    top_underdog = sorted([b for b in sorted_bets if b['Odds'] > 3.0], key=lambda x: x['Edge'], reverse=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if bet_of_the_week:
            botw = bet_of_the_week[0]
            st.metric(
                label="🎯 Bet of the Week (Highest Edge w/ >50% Confidence)",
                value=f"{botw['Bet']} in '{botw['Match']}'",
                delta=f"Odds: {botw['Odds']:.2f} | Edge: {botw['Edge']:.2%}"
            )
        else:
            st.info("No high-confidence favorite bets found for this cycle.")
    with col2:
        if top_underdog:
            tu = top_underdog[0]
            st.metric(
                label="⚡ Top Underdog Play (Highest Edge w/ >3.0 Odds)",
                value=f"{tu['Bet']} in '{tu['Match']}'",
                delta=f"Odds: {tu['Odds']:.2f} | Edge: {tu['Edge']:.2%}"
            )
        else:
            st.info("No significant underdog value bets found for this cycle.")

    # --- Main Value Dashboard ---
    st.header("📈 Value Dashboard")
    st.dataframe(value_df.style.format({
        'Odds': '{:.2f}', 
        'Edge': '{:.2%}', 
        'Confidence': '{:.2%}',
        'Stake (Kelly/4)': '{:.2%}'
    }).background_gradient(cmap='Greens', subset=['Edge']))
    
    # --- Smart Parlay Builder ---
    st.header("🧩 Smart Parlay Builder")
    if len(value_bets) >= 2:
        # Use the dictionary records for easier processing
        parlay_legs = value_df.sort_values('Edge', ascending=False).to_dict('records')
        
        for n in [2, 3]:
            if len(parlay_legs) < n: continue
            
            best_parlay_edge = -1
            best_parlay_combo = None
            
            # Find the best combination of n bets
            for combo in combinations(parlay_legs, n):
                # Ensure one bet per match
                if len(set([c['Match'] for c in combo])) != n: continue
                
                # Calculate the combined edge
                parlay_edge = np.prod([1 + c['Edge'] for c in combo]) - 1
                
                if parlay_edge > best_parlay_edge:
                    best_parlay_edge = parlay_edge
                    best_parlay_combo = combo
            
            if best_parlay_combo:
                total_odds = np.prod([leg['Odds'] for leg in best_parlay_combo])
                st.subheader(f"🔥 Top {n}-Team Parlay")
                st.metric(label=f"+EV Edge: {best_parlay_edge:.2%}", value=f"Total Odds: {total_odds:.2f}")
                for leg in best_parlay_combo:
                    st.markdown(f"  - **Leg:** {leg['Match']} -> **{leg['Bet']}** @ {leg['Odds']:.2f} (Edge: {leg['Edge']:.2%})")
    else:
        st.info("Not enough value bets to build a compelling parlay.")

else:
    st.warning("No betting recommendations loaded. This could be because:")
    st.markdown("- The daily backend analysis has not run yet. (Check your GitHub Actions tab).")
    st.markdown("- No value bets were found in the most recent analysis.")
    st.markdown("- The GitHub URL in the `app.py` file is incorrect.")

# --- Sidebar Information ---
st.sidebar.header("About This App")
st.sidebar.info(
    "This dashboard displays the output of an automated, multi-model betting analysis engine. "
    "The backend runs daily on GitHub Actions, and this app provides the results."
)
st.sidebar.success("System Status: Connected to Live Data Feed.")
