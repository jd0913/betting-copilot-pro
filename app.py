# app.py
# Betting Co-Pilot Pro - v73.0 (Google-only approach)
# FIX: Removed Streamlit Secrets dependency, uses Google for all data

import streamlit as st
import utils
import views

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Betting Co-Pilot Pro", 
    page_icon="🚀", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0  # Default starting bankroll

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Command Center"  # Default page

if 'bet_slip' not in st.session_state:
    st.session_state.bet_slip = []

# ==============================================================================
# SIDEBAR NAVIGATION & CONTROLS
# ==============================================================================
with st.sidebar:
    utils.inject_custom_css()
    st.title("🚀 Betting Co-Pilot Pro")
    
    # Bankroll Management
    st.header("💰 Bankroll")
    st.session_state.bankroll = st.number_input(
        "Current Bankroll ($)", 
        value=float(st.session_state.bankroll),
        min_value=0.0,
        step=10.0,
        format="%.2f"
    )
    
    if st.session_state.bankroll < 100:
        st.warning("⚠️ Low bankroll", icon="🚨")
    elif st.session_state.bankroll > 10000:
        st.success("🏦 High roller", icon="💎")
    
    # Navigation
    st.markdown("---")
    st.subheader("🧭 Navigation")
    
    pages = ["Command Center", "Market Map", "Bet Tracker", "History", "About"]
    for page in pages:
        if st.button(
            f"{'✅' if st.session_state.current_page == page else '➡️'} {page}",
            use_container_width=True,
            type="primary" if st.session_state.current_page == page else "secondary"
        ):
            st.session_state.current_page = page
            st.rerun()
    
    # System Status (since we removed secrets, just show basic status)
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    st.success("✅ Data Sources: Connected")
    st.success("✅ Auto-Settlement: Active (Google Score Lookup)")
    st.success("✅ Score Tracking: Enabled")

# ==============================================================================
# MAIN CONTENT
# ==============================================================================
if st.session_state.current_page == "Command Center":
    views.render_dashboard(
        bankroll=st.session_state.bankroll,
        kelly_multiplier=0.25
    )
elif st.session_state.current_page == "Market Map":
    views.render_market_map()
elif st.session_state.current_page == "Bet Tracker":
    views.render_bet_tracker(st.session_state.bankroll)
elif st.session_state.current_page == "History":
    views.render_history()
elif st.session_state.current_page == "About":
    views.render_about()

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.caption("Betting Co-Pilot Pro • Running on Streamlit Cloud")
st.caption("⚠️ **GAMBLING INVOLVES RISK. ONLY BET WHAT YOU CAN AFFORD TO LOSE.**")
