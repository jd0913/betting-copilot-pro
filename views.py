# views.py — v67.0 FINAL (Strict Parlay + All Fixes)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import utils
import hashlib

def render_dashboard(bankroll, kelly_multiplier=1.0):
    st.markdown('<p class="gradient-text">Live Command Center</p>', unsafe_allow_html=True)
    utils.inject_custom_css()
    
    df = utils.get_latest_bets()
    if df.empty:
        st.success("System Online • Market Scanned • No +EV Found")
        return

    # Risk Control
    with st.expander("Risk Controls", expanded=True):
        c1, c2 = st.columns(2)
        max_daily_pct = c1.number_input("Max Daily Risk (%)", 1.0, 25.0, 10.0, 0.5)
        c2.metric("Max Exposure", f"${bankroll * max_daily_pct / 100:,.0f}")
    max_daily_risk = bankroll * (max_daily_pct / 100)

    # Smart Picks
    candidates = df[df['Bet Type'] != 'ARBITRAGE'].copy()
    smart_picks = []
    
    if not candidates.empty:
        banker = candidates[candidates['Odds'] < 2.0].nlargest(1, 'Confidence')
        if not banker.empty:
            smart_picks.append(("The Banker", banker.iloc[0], "#00e676"))
        
        value = candidates.nlargest(1, 'Edge')
        if not value.empty and value.iloc[0].name not in [p[1].name for p in smart_picks]:
            smart_picks:
            smart_picks.append(("The Value Play", value.iloc[0], "#00C9FF"))

        cols = st.columns(len(smart_picks))
        for i, (label, row, color) in enumerate(smart_picks):
            with cols[i]:
                stake_cash = bankroll * row['Stake'] * kelly_multiplier
                st.markdown(f"""
                <div style="background:#1e2130;border:3px solid {color};border-radius:16px;padding:20px;text-align:center;">
                    <div style="color:{color};font-weight:900;font-size:1.1em;">{label}</div>
                    <div style="font-size:1.2em;margin:8px 0;"><b>{utils.get_team_emoji(row['Sport'])} {row['Match']}</b></div>
                    <div style="color:#00C9FF;font-size:1.3em;font-weight:bold;">{row['Bet']}</div>
                    <div style="margin:10px 0;">
                        <span style="font-size:0.9em;color:#888;">ODDS</span><br>
                        <span style="font-size:1.8em;font-weight:800;color:white;">{row['Odds']:.2f}</span>
                    </div>
                    <div style="display:flex;justify-content:space-around;">
                        <div><small>EDGE</small><br><b style="color:{color}">{row['Edge']:.1%}</b></div>
                        <div><small>STAKE</small><br><b>${stake_cash:.0f}</b></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Main Feed with exposure tracking
    current_exposure = 0.0
    for _, row in df.iterrows():
        stake_cash = bankroll * row['Stake'] * kelly_multiplier
        current_exposure += stake_cash
        
        if current_exposure > max_daily_risk:
            st.error("DAILY RISK LIMIT EXCEEDED — Remaining bets skipped")
            break
            
        # your original beautiful card rendering here (unchanged)
        # ... [keep all your HTML cards exactly as you wrote them] ...

    # FIXED PARLAY BUILDER (correlated legs removed, correct Kelly)
    st.markdown("---")
    st.subheader("Smart Parlay Builder")
    
    safe_pool = df[(df['Odds'].between(1.25, 2.20)) & (df['Bet Type'] != 'ARBITRAGE')].drop_duplicates('Match')
    value_pool = df[(df['Edge'] > 0.04) & (df['Bet Type'] != 'ARBITRAGE')].drop_duplicates('Match')
    
    if len(value_pool) >= 2:
        t1, t2, t3, t4 = st.tabs(["Safe 2-Leg", "Value 3-Leg", "Lotto 4-Leg", "Hail Mary 5-Leg"])
        
        def build_best_parlay(pool, legs, title):
            if len(pool) < legs:
                st.info(f"Not enough bets for {title}")
                return
            best_ev = -999
            best_combo = None
            for combo in combinations(pool.to_dict('records'), legs):
                prob = np.prod([c['Confidence'] for c in combo])
                odds = np.prod([c['Odds'] for c in combo])
                ev = prob * odds - 1
                if ev > best_ev:
                    best_ev = ev
                    best_combo = combo
            if best_combo:
                odds = np.prod([c['Odds'] for c in best_combo])
                stake = bankroll * min(best_ev / (odds - 1) * 0.25 * kelly_multiplier, 0.06)
                st.success(f"**{title}** @ {odds:.2f}x → Stake ${stake:.0f} → Payout ${stake*odds:.0f}")
                for leg in best_combo:
                    st.markdown(f"• {leg['Bet']} @ {leg['Odds']:.2f} — {leg['Match']}")
        
        with t1: build_best_parlay(safe_pool, 2, "Bankroll Builder")
        with t2: build_best_parlay(value_pool, 3, "Value Stack")
        with t3: build_best_parlay(value_pool, 4, "Lotto Ticket")
        with t4: build_best_parlay(value_pool, 5, "Hail Mary")

# Keep render_market_map(), render_bet_tracker(), render_history(), render_about() exactly as you wrote them
