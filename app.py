# app.py
# This is your fast, lightweight Streamlit dashboard.

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Professional Betting Co-Pilot", layout="wide")
st.title("🚀 Professional Betting Co-Pilot")

# The URL to the raw CSV file in your GitHub repository
DATA_URL = "https://raw.githubusercontent.com/jd0913/betting-model/main/latest_bets.csv"

@st.cache_data(ttl=3600) # Cache the data for 1 hour
def load_data():
    """Reads the latest recommendations from your GitHub repo."""
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception as e:
        st.error(f"Could not load the latest bets. The backend might not have run yet. Error: {e}")
        return pd.DataFrame()

st.header("📈 Value Dashboard")
st.markdown("These recommendations are automatically updated daily.")

value_df = load_data()

if not value_df.empty:
    # (All the Streamlit display logic from the previous versions goes here)
    # For example:
    st.dataframe(value_df.style.format({'Odds': '{:.2f}', 'Edge': '{:.2%}', 'Confidence': '{:.2%}'}))
    
    # (Add the Parlay Builder, etc., reading from value_df)
    
else:
    st.info("No value bets are currently recommended, or the backend is still running its daily analysis.")

st.sidebar.success("System Status: Connected to Live Data Feed.")
