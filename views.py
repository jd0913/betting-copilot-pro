# views.py
# v69.0 — Shows ALL active bets + correct history with scores

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import utils

def render_dashboard(bankroll, kelly_multiplier):
    st.markdown('<p class="gradient-text">Live Command Center</p>', unsafe_allow_html=True)
    
    df = utils.load_data(utils.LATEST_URL)
    
    with st.expander("Risk Controls", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            max_risk_pct = st.number_input("Max Daily Risk %", 1.0, 20.0, 5.0)
        with col2:
            st.caption("Quarter Kelly enforced")
        st.warning(f"Max daily risk: **${bankroll * max_risk_pct/100:.0f}**")

    if isinstance(df, str):
        if "NOT_FOUND" in df or "ERROR" in df:
            st.error("No bets loaded — run backend_runner.py first!")
            st.stop()
        if df == "NO_BETS_FOUND":
            st.success("System running — no value today")
            st.balloons()
            st.stop()

    # ==================== ALL ACTIVE BETS TABLE ====================
    st.markdown("### All Active Value Bets")
    active = df[df['Bet Type'] != 'ARBITRAGE'].copy()
    active = active[active['Odds'] > 1.01]

    if active.empty:
        st.info("No value bets right now.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Bets", len(active))
        c2.metric("Best Edge", f"{active['Edge'].max():.1%}")
        c3.metric("Avg Edge", f"{active['Edge'].mean():.1%}")
        c4.metric("Total Rec. Stake", f"${(active['Stake'] * bankroll * 4).sum():.0f}")

        disp = active.copy()
        disp['Risk'] = disp.apply(utils.get_risk_badge, axis=1)
        disp['Edge'] = (disp['Edge']*100).round(1).astype(str) + "%"
        disp['Conf'] = (disp['Confidence']*100).round(0).astype(int).astype(str) + "%"
        disp['Stake $'] = (disp['Stake'] * bankroll * 4).round(0).astype(int)

        disp = disp[['Risk', 'Sport', 'Match', 'Bet', 'Odds', 'Edge', 'Conf', 'Stake $']]
        st.markdown(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

        if st.button("Add ALL to Bet Slip", type="primary"):
            for _, r in active.iterrows():
                bet = r.to_dict()
                bet['User_Stake'] = round(bankroll * r['Stake'] * 4, 2)
                if bet not in st.session_state.bet_slip:
                    st.session_state.bet_slip.append(bet)
            st.success(f"Added {len(active)} bets!")
            st.rerun()

    # Optional: keep your Smart Picks below this line if you want them too

def render_market_map():
    st.markdown('<p class="gradient-text">Market Map</p>', unsafe_allow_html=True)
    df = utils.load_data(utils.LATEST_URL)
    if isinstance(df, str) or df.empty:
        st.info("No data")
        return
    df = df[(df['Bet Type'] != 'ARBITRAGE') & (df['Odds'] > 1.01)].copy()
    if df.empty:
        st.info("No value bets")
        return
    df['Imp'] = 1/df['Odds']
    df['Model'] = df['Confidence']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Imp'], y=df['Model'], mode='markers',
                             marker=dict(size=df['Edge']*300+10, color=df['Edge'], colorscale='Viridis', showscale=True),
                             text=df['Match'] + " " + df['Bet']))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash')))
    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

def render_bet_tracker(bankroll):
    st.markdown('<p class="gradient-text">Bet Slip</p>', unsafe_allow_html=True)
    if st.session_state.bet_slip:
        slip = pd.DataFrame(st.session_state.bet_slip)
        total_stake = slip['User_Stake'].sum()
        odds = np.prod(slip['Odds']) if len(slip)>1 else slip['Odds'].iloc[0]
        c1,c2,c3 = st.columns(3)
        c1.metric("Stake", f"${total_stake:.0f}")
        c2.metric("Odds", f"{odds:.2f}x")
        c3.metric("Potential", f"${total_stake*odds:.0f}")
        if st.button("Clear Slip"): st.session_state.bet_slip = []; st.rerun()
    else:
        st.info("Slip empty")

def render_history():
    st.markdown('<p class="gradient-text">Betting History</p>', unsafe_allow_html=True)
    df = utils.load_data(utils.HISTORY_URL)
    if isinstance(df, str):
        st.info("No history yet")
        return

    stats = utils.get_performance_stats(df)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Profit", f"${stats['total_profit']:.2f}", delta=f"{stats['roi']:.1%} ROI")
    c2.metric("Win Rate", f"{stats['win_rate']:.1%}")
    c3.metric("Total Bets", stats['total_bets'])
    c4.metric("Units", f"{stats['total_staked']:.1f}")

    show = df[['Formatted_Date','Sport','Match','Bet','Odds','Result','Profit','Score']].copy()
    show['Result'] = show['Result'].fillna('Pending')
    show['Badge'] = show['Result'].apply(utils.format_result_badge)
    show['Profit'] = show.apply(lambda x: f"+${x['Profit']:.2f}" if x['Result']=='Win' else f"${x['Profit']:.2f}" if x['Result']=='Loss' else "-", axis=1)
    show = show.drop('Result', axis=1)
    st.markdown(show.to_html(escape=False, index=False), unsafe_allow_html=True)

def render_about():
    st.markdown("# Betting Co-Pilot Pro v69")
    st.success("Live Settlement • All Bets Visible • Cloud Ready")
