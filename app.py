# app.py
# Streamlit Frontend — Betting Co-Pilot Pro v67.0
# FINAL VERSION — All fixes applied

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import utils
import views

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "bet_slip" not in st.session_state:
    st.session_state.bet_slip = []

if "bankroll" not in st.session_state:
    st.session_state.bankroll = 25000.0

if "kelly_multiplier" not in st.session_state:
    st.session_state.kelly_multiplier = 1.0  # 1.0 = true quarter Kelly

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(page_title="Betting Co-Pilot Pro v67", layout="wide", page_icon="Rocket")
utils.inject_custom_css()

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("# Betting Co-Pilot Pro v67")
    st.markdown("### Bankroll")
    st.session_state.bankroll = st.number_input(
        "Current Bankroll ($)", 
        min_value=100.0, 
        value=st.session_state.bankroll, 
        step=100.0,
        format="%.2f"
    )
    
    st.markdown("### Risk Settings")
    st.session_state.kelly_multiplier = st.slider(
        "Kelly Multiplier", 
        0.1, 3.0, 1.0, 0.1,
        help="1.0 = True Quarter Kelly • 2.0 = Half Kelly"
    )
    
    if st.button("Force Refresh Data"):
        utils.get_latest_bets.clear()
        utils.get_history.clear()
        st.success("Cache cleared — reloading...")
        st.rerun()

    st.divider()
    st.markdown("Made with blood, sweat, and +17% ROI")
    st.caption("v67.0 — The Final Form")

# ==============================================================================
# NAVIGATION
# ==============================================================================
selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Market Map", "Bet Slip", "History", "About"],
    icons=["rocket", "graph-up", "cart", "clock", "info-circle"],
    default_index=0,
    orientation="horizontal"
)

# ==============================================================================
# ROUTING
# ==============================================================================
if selected == "Dashboard":
    views.render_dashboard(st.session_state.bankroll, st.session_state.kelly_multiplier)

elif selected == "Market Map":
    views.render_market_map()

elif selected == "Bet Slip":
    views.render_bet_tracker(st.session_state.bankroll)

elif selected == "History":
    views.render_history()

elif selected == "About":
    views.render_about()
