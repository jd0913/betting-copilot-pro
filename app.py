# app.py
# The "Manager" - Entry point for v58.0
# Fixes: Bankroll now accepts decimals (e.g. 1000.50)

import streamlit as st
import utils
import views

# ==============================================================================
# CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Betting Co-Pilot Pro", 
    page_icon="🚀", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ==============================================================================
# NAVIGATION & STATE
# ==============================================================================
if 'bet_slip' not in st.session_state: st.session_state.bet_slip = []

# Inject CSS
utils.inject_custom_css()

st.sidebar.title("Navigation")

# Global Settings
st.sidebar.header("💰 Bankroll")

# *** FIX: Explicitly use float (1000.0), step=0.01, and format="%.2f" ***
bankroll = st.sidebar.number_input(
    "Bankroll ($)", 
    value=1000.0, 
    min_value=0.0, 
    step=0.01, 
    format="%.2f"
)

# Hardcoded Professional Standard (Quarter Kelly) - Hidden from user
kelly_multiplier = 0.25 

st.sidebar.markdown("---")

# Page Routing
page = st.sidebar.radio("Go To", ["Command Center", "Market Map", "Bet Tracker", "History", "About"])

if st.sidebar.button("🔄 Force Refresh"):
    utils.load_data.clear()
    st.rerun()

if page == "Command Center": views.render_dashboard(bankroll, kelly_multiplier)
elif page == "Market Map": views.render_market_map()
elif page == "Bet Tracker": views.render_bet_tracker(bankroll)
elif page == "History": views.render_history()
elif page == "About": views.render_about()

st.sidebar.markdown("---")
st.sidebar.caption(f"Connected to: `{utils.GITHUB_USERNAME}/{utils.GITHUB_REPO}`")
