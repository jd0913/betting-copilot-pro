# app.py
# The "Manager" - Entry point for v45.0

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
bankroll = st.sidebar.number_input("Bankroll ($)", value=1000, step=100)
kelly_multiplier = st.sidebar.slider("Kelly Multiplier", 0.1, 1.0, 0.25)

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
