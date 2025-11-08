# app.py
# This is the "Everything Machine" Streamlit Cockpit.

import streamlit as st
import pandas as pd
from itertools import combinations
import numpy as np

st.set_page_config(page_title="The Everything Machine", layout="wide")
st.title("🚀 The Everything Machine v16.0")

# ==============================================================================
# IMPORTANT: CONFIGURE YOUR GITHUB REPOSITORY DETAILS HERE
# ==============================================================================
GITHUB_USERNAME = "jd0913"
GITHUB_REPO = "betting-copilot-pro"
# ==============================================================================

DATA_URL = f"https://raw.githubusercontent.com/jd0913/betting-copilot-pro/main/latest_bets.csv"

@st.cache_data(ttl=600)
def load_data():
    """Reads the latest recommendations from your GitHub repo."""
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception:
        return pd.DataFrame()

# --- Main App Logic ---
st.sidebar.header("Controls")
all_bets_df = load_data()

if not all_bets_df.empty:
    available_sports = all_bets_df['Sport'].unique()
    selected_sport = st.sidebar.selectbox("Select a Sport", available_sports)
    
    sport_df = all_bets_df[all_bets_df['Sport'] == selected_sport]
    
    st.header(f"📈 {selected_sport} Value Dashboard")
    
    if not sport_df.empty:
        if selected_sport == "Soccer":
            st.dataframe(sport_df[['League', 'Match', 'Bet Type', 'Bet', 'Odds', 'Edge', 'Confidence']].style.format({
                'Odds': '{:.2f}', 'Edge': '{:.2%}', 'Confidence': '{:.2%}'
            }).background_gradient(cmap='Greens', subset=['Edge']))
            
        elif selected_sport == "NFL":
            st.dataframe(sport_df[['Match', 'Bet', 'Edge (Points)', 'Model Predicts']].style.format({
                'Edge (Points)': '{:+.1f}', 'Model Predicts': '{}'
            }).background_gradient(cmap='Greens', subset=['Edge (Points)']))
            
        elif selected_sport == "NBA":
            st.dataframe(sport_df[['Match', 'Bet', 'Edge (Points)', 'Model Predicts']].style.format({
                'Edge (Points)': '{:+.1f}', 'Model Predicts': '{}'
            }).background_gradient(cmap='Greens', subset=['Edge (Points)']))
            
    else:
        st.info(f"No value bets are currently recommended for {selected_sport}.")
        
    # --- Smart Parlay Engine ---
    st.header("🧩 Smart Parlay Engine")
    parlay_legs = all_bets_df[all_bets_df['Sport'] == 'Soccer'].sort_values('Edge', ascending=False).to_dict('records')
    if len(parlay_legs) >= 2:
        st.subheader("Top Soccer Parlay")
        combo = parlay_legs[:2]
        total_odds = np.prod([leg['Odds'] for leg in combo])
        parlay_edge = np.prod([1 + leg['Edge'] for leg in combo]) - 1
        st.metric(label=f"+EV Edge: {parlay_edge:.2%}", value=f"Total Odds: {total_odds:.2f}")
        for leg in combo:
            st.markdown(f"  - **Leg:** {leg['Match']} ({leg['League']}) -> **{leg['Bet']}** @ {leg['Odds']:.2f}")
    else:
        st.info("Not enough soccer value bets to build a compelling parlay.")

else:
    st.warning("No betting recommendations loaded. The backend may be running or all markets are quiet.")

st.sidebar.info("This dashboard reads data updated daily by an automated backend.")
