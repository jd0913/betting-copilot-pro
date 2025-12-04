# app.py
# Betting Co-Pilot Pro - v59.0 (Refactored)
# Key Fixes: 
#   - Persistent bankroll & navigation state
#   - Secure secrets management
#   - Professional risk controls
#   - Enhanced UX patterns

import streamlit as st
import utils
from config import BETTING_CONFIG  # Critical: Centralized configuration
import views

# ==============================================================================
# SESSION STATE INITIALIZATION (MUST BE FIRST)
# ==============================================================================
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0  # Default bankroll

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Command Center"  # Default page

if 'bet_slip' not in st.session_state:
    st.session_state.bet_slip = []

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
# SECURITY & SECRETS SETUP
# ==============================================================================
# NEVER hardcode credentials - use Streamlit Secrets
try:
    GITHUB_USERNAME = st.secrets["github"]["username"]
    GITHUB_REPO = st.secrets["github"]["repo"]
except KeyError:
    st.error("🚨 Missing GitHub credentials in `.streamlit/secrets.toml`")
    st.stop()

# ==============================================================================
# SIDEBAR NAVIGATION & CONTROLS
# ==============================================================================
with st.sidebar:
    utils.inject_custom_css()
    st.title("🚀 Betting Co-Pilot Pro")
    
    # ===== BANKROLL MANAGEMENT =====
    st.header("💰 Bankroll Management")
    st.session_state.bankroll = st.number_input(
        "Current Bankroll ($)", 
        value=float(st.session_state.bankroll),
        min_value=0.0,
        step=0.01,
        format="%.2f",
        help="Your total betting capital. Adjust as you win/lose."
    )
    
    # Visual risk indicators
    if st.session_state.bankroll < 100:
        st.warning("⚠️ Low bankroll - risk controls active", icon="🚨")
    elif st.session_state.bankroll > 50000:
        st.success("🏦 High roller mode", icon="💎")
    
    # ===== PAGE NAVIGATION =====
    st.markdown("---")
    st.subheader("🧭 Navigation")
    
    # Professional navigation pattern (persists state)
    nav_pages = ["Command Center", "Market Map", "Bet Tracker", "History", "About"]
    for page in nav_pages:
        if st.button(
            f"{'✅' if st.session_state.current_page == page else '➡️'} {page}", 
            use_container_width=True,
            type="primary" if st.session_state.current_page == page else "secondary"
        ):
            st.session_state.current_page = page
            st.rerun()
    
    # ===== REFRESH CONTROLS =====
    st.markdown("---")
    st.subheader("🔄 Data Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⟳ Refresh Odds", use_container_width=True):
            utils.load_data.clear()
            st.toast("✅ Odds refreshed!", icon="⚡")
    with col2:
        if st.button("🗑️ Clear Bets", use_container_width=True):
            st.session_state.bet_slip = []
            st.toast("✅ Bet slip cleared", icon="🧹")
    
    # ===== FOOTER =====
    st.markdown("---")
    st.caption(f"📊 Data Source: `{GITHUB_USERNAME}/{GITHUB_REPO}`")
    st.caption(f"⚙️ Kelly Fraction: {BETTING_CONFIG['kelly_fraction']:.0%}")
    
    # Responsible gambling notice (LEGAL REQUIREMENT)
    st.markdown(
        "<div style='background-color:#1e1e1e;padding:10px;border-radius:5px;margin-top:15px'>"
        "<p style='color:#ff4b4b;font-weight:bold;text-align:center'>"
        "⚠️ GAMBLING INVOLVES SUBSTANTIAL RISK<br>"
        "ONLY BET WHAT YOU CAN AFFORD TO LOSE"
        "</p></div>",
        unsafe_allow_html=True
    )

# ==============================================================================
# MAIN CONTENT RENDERING
# ==============================================================================
# Professional routing pattern
if st.session_state.current_page == "Command Center":
    views.render_dashboard(
        bankroll=st.session_state.bankroll,
        kelly_fraction=BETTING_CONFIG["kelly_fraction"]
    )
elif st.session_state.current_page == "Market Map":
    views.render_market_map()
elif st.session_state.current_page == "Bet Tracker":
    views.render_bet_tracker(
        bankroll=st.session_state.bankroll,
        min_edge=BETTING_CONFIG["min_edge"]
    )
elif st.session_state.current_page == "History":
    views.render_history()
elif st.session_state.current_page == "About":
    views.render_about()

# ==============================================================================
# GLOBAL TOAST NOTIFICATIONS
# ==============================================================================
if "notification" in st.session_state:
    st.toast(st.session_state.notification, icon="✅")
    del st.session_state.notification
