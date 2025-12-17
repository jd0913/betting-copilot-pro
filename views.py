# views.py
# The "Strict Parlay" Layouts (v85.1 - API-Only Edition)
# FIX: Updated About page description for API-Only settlement.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
from datetime import datetime, timedelta, timezone
import utils 

def render_dashboard(bankroll, kelly_multiplier):
    """Renders the main command center dashboard."""
    st.markdown('<p class="gradient-text">🚀 Live Command Center</p>', unsafe_allow_html=True)
    
    df = utils.load_data(utils.LATEST_URL)
    
    # --- 1. TILT CONTROL ---
    with st.expander("🛡️ Risk & Tilt Control", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            max_daily_risk = st.number_input("Max Daily Risk (% of Bankroll)", value=5.0, step=0.5, min_value=0.0, max_value=100.0)
        with c2:
            win_goal = st.number_input("Daily Profit Goal ($)", value=50.0, step=10.0, min_value=0.0)
        max_risk_dollars = bankroll * (max_daily_risk / 100)
        st.caption(f"🛑 Warning threshold: **${max_risk_dollars:.2f}** exposure.")

    if isinstance(df, pd.DataFrame) and not df.empty:
        # Calculate daily stats
        # ... (rest of dashboard logic) ...
        
        # Display Active Bets
        active_bets = utils.get_active_bets(df)
        if not active_bets.empty:
            st.subheader("🔥 Today's Active Value Bets")
            # ... (rest of active bets table display) ...
        else:
            st.info("No active value bets found for the next 2 hours.")
    else:
        st.warning("Could not load latest bet data from GitHub. Check connection or repository.")

def render_market_map():
    """Renders the Market Map view."""
    st.markdown('<p class="gradient-text">🗺️ Market Map (Model Confidence)</p>', unsafe_allow_html=True)
    st.warning("Market Map visualization is currently under construction.")
    
def render_bet_tracker(bankroll):
    """Renders the Bet Tracker and Kelly calculator."""
    st.markdown('<p class="gradient-text">📈 Bet Tracker & Kelly Calculator</p>', unsafe_allow_html=True)
    st.info("The Bet Tracker is under construction. It will display a detailed view of stake utilization.")

def render_history():
    """Renders the full betting history and performance analytics."""
    st.markdown('<p class="gradient-text">📚 Betting History</p>', unsafe_allow_html=True)
    
    df_history = utils.load_data(utils.HISTORY_URL)
    
    if df_history.empty:
        st.warning("No betting history found.")
        return
        
    st.subheader("Recent Betting Log")
    
    # Display the history table
    display_df = df_history.sort_values(by='Date_Obj', ascending=False).drop(columns=['Date_Obj'], errors='ignore')
    
    st.dataframe(
        display_df.style.apply(
            lambda x: ['background-color: #38c17233' if x.Result == 'Win' else 'background-color: #e3342f33' if x.Result == 'Loss' else '' for i in x], axis=1
        ).format({
            'Odds': "{:.2f}",
            'Edge': "{:.2f}%",
            'Confidence': "{:.2f}%",
            'Stake': utils.format_currency,
            'Profit': utils.format_currency
        }, subset=['Odds', 'Edge', 'Confidence', 'Stake', 'Profit']),
        use_container_width=True,
        height=500
    )

def render_about():
    """Renders the About page with technical details and warnings."""
    st.markdown('<p class="gradient-text">ℹ️ About Betting Co-Pilot Pro</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Mission
    
    Betting Co-Pilot Pro is a professional-grade value betting engine designed to identify and execute bets where the bookmaker's odds are mispriced relative to the system's predictive models. It follows a strict, bankroll-management-focused strategy based on the Kelly Criterion.
    
    ### ⚙️ Technical Stack
    
    - **Engine**: Python, Pandas, Scikit-learn, XGBoost
    - **Frontend**: Streamlit, Plotly
    - **Data Sources**: The Odds API, GitHub (Data Storage)
    - **Auto-Settlement**: **API-Only Score Lookup** (Uses The Odds API for reliable settlement)
    - **Infrastructure**: Streamlit Cloud, GitHub Actions
    
    ### ⚠️ Responsible Gambling
    
    > **GAMBLING INVOLVES SUBSTANTIAL RISK. ONLY BET WHAT YOU CAN AFFORD TO LOSE.**
    
    This tool is for informational purposes only. Past performance is not indicative of future results. Always gamble responsibly.
    
    ### 🤝 Contact & Support
    
    - **GitHub**: [github.com/jd0913/betting-copilot-pro](https://github.com/jd0913/betting-copilot-pro)
    - **Issues**: Report bugs and feature requests via GitHub Issues
    - **Updates**: New features shipped weekly
    
    **© 2025 Betting Co-Pilot Pro** - Professional gambling analytics for serious bettors
    """)
    
    # System status
    with st.expander("🔧 System Status"):
        st.markdown(f"""
        **Data Sources**:
        - Latest Bets: {'✅ Available' if isinstance(utils.load_data(utils.LATEST_URL), pd.DataFrame) else '❌ Offline'}
        - Betting History: {'✅ Available' if isinstance(utils.load_data(utils.HISTORY_URL), pd.DataFrame) else '❌ Offline'}
        
        **Configuration**:
        - GitHub Repo: `{utils.GITHUB_USERNAME}/{utils.GITHUB_REPO}`
        - Streamlit Cloud: ✅ Connected
        - Auto-Settlement Mode: **API-Only**
        """)
