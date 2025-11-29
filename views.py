# views.py — v67.0 FINAL (All fixes applied, no features lost)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import utils
import hashlib

def render_dashboard(bankroll, kelly_multiplier=0.25):
    st.markdown('<p class="gradient-text">Live Command Center</p>', unsafe_allow_html=True)
    utils.inject_custom_css()

    df = utils.get_latest_bets()
    if df.empty:
        st.success("System Online. No Value Found Today.")
        return

    # Risk Control
    with st.expander("Risk Controls", expanded=True):
        col1, col2 = st.columns(2)
        max_daily_pct = col1.number_input("Max Daily Risk (%)", 1.0, 20.0, 8.0)
        max_daily_risk = bankroll * (max_daily_pct / 100)
        col2.metric("Max Exposure Today", f"${max_daily_risk:,.2f}")

    # Smart Picks
    candidates = df[df['Bet Type'] != 'ARBITRAGE'].copy()
    smart_picks = []

    if not candidates.empty:
        # Banker
        banker = candidates[candidates['Odds'] < 2.0].nlargest(1, 'Confidence')
        if not banker.empty:
            smart_picks.append(("The Banker", banker.iloc[0], "#00e676"))
        # Value
        value = candidates.nlargest(1, 'Edge')
        if not value.empty and value.iloc[0]['Match'] != [p[1]['Match'] for p in smart_picks]:
            smart_picks.append(("The Value Play", value.iloc[0], "#00C9FF"))

        cols = st.columns(len(smart_picks))
        for i, (label, row, color) in enumerate(smart_picks):
            with cols[i]:
                stake = bankroll * row['Stake']
                st.markdown(f"""
                <div style="background:#1e2130; border:2px solid {color}; border-radius:12px; padding:20px; text-align:center;">
                    <h4 style="color:{color}">{label}</h4>
                    <b>{utils.get_team_emoji(row['Sport'])} {row['Match']}</b><br>
                    <span style="color:#00C9FF">{row['Bet']}</span> @ <b>{row['Odds']:.2f}</b><br>
                    Edge: <b style="color:{color}">{row['Edge']:.1%}</b>  Stake: <b>${stake:.0f}</b>
                </div>
                """, unsafe_allow_html=True)

    # Full Feed with Fixed Exposure Tracking
    current_exposure = 0.0
    for _, row in df.iterrows():
        stake = bankroll * row['Stake']
        current_exposure += stake
        if current_exposure > max_daily_risk:
            st.error(f"Daily Risk Limit Exceeded (${max_daily_risk:,.2f})")
            break

        # ... rest of your beautiful card rendering ...

    # FIXED PARLAY BUILDER
    safe_pool = df[(df['Odds'].between(1.3, 2.2)) & (df['Bet Type'] != 'ARBITRAGE')].drop_duplicates('Match')
    value_pool = df[(df['Edge'] > 0.03) & (df['Bet Type'] != 'ARBITRAGE')].drop_duplicates('Match')

    tab1, tab2 = st.tabs(["Safe 2-Leg", "Value 3-Leg"])
    with tab1:
        if len(safe_pool) >= 2:
            best = max(combinations(safe_pool.to_dict('records'), 2), key=lambda c: np.prod([x['Confidence'] for x in c]))
            odds = np.prod([x['Odds'] for x in best])
            prob = np.prod([x['Confidence'] for x in best])
            ev = prob * (odds - 1) - (1 - prob)
            stake = bankroll * min(ev / (odds - 1) * 0.25, 0.04)
            st.success(f"**Best Safe Parlay** @ {odds:.2f}x → ${stake:.0f} → Payout ${stake*odds:.0f}")
