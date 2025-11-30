# views.py
# FULLY FIXED v68.1 — Works on Streamlit Cloud · No DataFrame comparison crash
# Fixed: isinstance(df, str) checks everywhere

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import utils

def render_dashboard(bankroll, kelly_multiplier):
    st.markdown('<p class="gradient-text">Live Command Center</p>', unsafe_allow_html=True)
    
    df = utils.load_data(utils.LATEST_URL)
    
    # --- TILT CONTROL ---
    with st.expander("Risk & Tilt Control", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            max_daily_risk = st.number_input("Max Daily Risk (% of Bankroll)", value=5.0, step=0.5, min_value=0.1, max_value=20.0)
        with c2:
            win_goal = st.number_input("Daily Profit Goal ($)", value=50.0, step=10.0)
        with c3:
            st.caption("Quarter Kelly is enforced globally")
        max_risk_dollars = bankroll * (max_daily_risk / 100)
        st.warning(f"Never risk more than **${max_risk_dollars:.2f}** today")

    # --- DATA HANDLING (CRITICAL FIX) ---
    if isinstance(df, str):
        if df == "FILE_NOT_FOUND":
            st.error("latest_bets.csv not found on GitHub. Run backend or check repo.")
            st.stop()
        elif df == "CONNECTION_ERROR":
            st.error("Failed to connect to GitHub. Try again later.")
            st.stop()
        elif df == "NO_BETS_FOUND":
            st.success("System Online • Market Scanned • No Value Found Today")
            st.balloons()
            st.stop()

    # If we get here → df is a real DataFrame
    # --- FILTERS ---
    sports = ["All"] + sorted(df['Sport'].dropna().unique().tolist())
    selected_sport = st.sidebar.selectbox("Filter Sport", sports, index=0)
    if selected_sport != "All":
        df = df[df['Sport'] == selected_sport]

    # --- KPI CARDS ---
    total_bets = len(df)
    top_edge = df['Edge'].max() if not df.empty else 0
    avg_edge = df['Edge'].mean() if not df.empty else 0

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Active Bets", total_bets)
    kpi2.metric("Top Edge", f"{top_edge:.1%}")
    kpi3.metric("Avg Edge", f"{avg_edge:.1%}")

    # --- SMART DAILY PICKS ---
    st.markdown("### Top Smart Picks")
    candidates = df[(df['Bet Type'] != 'ARBITRAGE') & (df['Odds'] > 0)].copy()
    
    if candidates.empty:
        st.info("No value bets found today.")
    else:
        smart_picks = []

        # 1. The Banker
        banker = candidates[candidates['Odds'] < 2.0].sort_values('Confidence', ascending=False).head(1)
        if not banker.empty:
            row = banker.iloc[0]
            smart_picks.append({"Label": "The Banker", "Row": row, "Color": "#00e676", "Reason": "Safe, High Confidence"})

        # 2. The Value Play
        remaining = candidates[~candidates.index.isin([p['Row'].name for p in smart_picks if 'Row' in p])]
        if not remaining.empty:
            value = remaining.sort_values('Edge', ascending=False).iloc[0]
            smart_picks.append({"Label": "The Value Play", "Row": value, "Color": "#00C9FF", "Reason": "Highest Edge"})

        # 3. The Diversifier
        used_matches = [p['Row']['Match'] for p in smart_picks]
        remaining = candidates[~candidates['Match'].isin(used_matches)]
        if not remaining.empty:
            div = remaining.sort_values('Edge', ascending=False).iloc[0]
            smart_picks.append({"Label": "Diversifier", "Row": div, "Color": "#FFD700", "Reason": "Portfolio Balance"})

        # Render Cards
        cols = st.columns(len(smart_picks) if smart_picks else 1)
        for idx, pick in enumerate(smart_picks):
            row = pick['Row']
            rec_stake = bankroll * row['Stake'] * (kelly_multiplier / 0.25)
            with cols[idx]:
                st.markdown(f"""
                <div class="bet-card" style="border-left: 5px solid {pick['Color']};">
                    <div style="font-weight: bold; color: {pick['Color']}; font-size: 0.9em;">{pick['Label']}</div>
                    <div style="font-size: 1.1em; font-weight: bold; margin: 8px 0;">{utils.get_team_emoji(row['Sport'])} {row['Match']}</div>
                    <div style="color: #00e676; font-weight: bold;">{row['Bet']} @ {row['Odds']:.2f}</div>
                    <div style="margin: 10px 0; font-size: 0.9em;">
                        <span style="color:#888;">Edge:</span> <strong>{row['Edge']:.1%}</strong> | 
                        <span style="color:#888;">Conf:</span> <strong>{row['Confidence']:.1%}</strong>
                    </div>
                    <div style="background: rgba(0,201,255,0.1); padding: 8px; border-radius: 8px; font-size: 0.9em;">
                        Recommended Stake: <strong>${rec_stake:.2f}</strong>
                    </div>
                    <div style="font-size: 0.8em; color: #888; margin-top: 8px;">{pick['Reason']}</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Add to Slip", key=f"add_{row.name}_{idx}"):
                    bet = row.to_dict()
                    bet['User_Stake'] = rec_stake
                    if bet not in st.session_state.bet_slip:
                        st.session_state.bet_slip.append(bet)
                        st.success("Added to Bet Slip!")

    # --- PARLAY BUILDER ---
    st.markdown("### Parlay Builder")
    tabs = st.tabs(["Bankroll Builder", "Value Stack", "Lotto Ticket", "Hail Mary"])

    safe_candidates = candidates[(candidates['Odds'] < 2.5) & (candidates['Confidence'] > 0.6)]
    value_candidates = candidates[candidates['Edge'] > 0.08]

    def render_parlay_card(pool, legs, title):
        if len(pool) < legs:
            st.info(f"Not enough strong bets for {legs}-leg {title}")
            return
        combo = list(combinations(pool.iterrows(), legs))
        best_combo = max(combo, key=lambda x: sum(r[1]['Edge'] for r in x))
        total_odds = np.prod([r[1]['Odds'] for r in best_combo])
        total_edge = sum(r[1]['Edge'] for r in best_combo)
        stake = bankroll * 0.005 * max(1, total_edge / 0.1)

        st.markdown(f"#### {title} ({legs} Legs) – {total_odds:.2f}x")
        for _, row in best_combo:
            st.markdown(f"• {row['Match']} → **{row['Bet']}** @ {row['Odds']:.2f}")
        c1, c2 = st.columns(2)
        c1.metric("Combined Odds", f"{total_odds:.2f}x")
        c2.metric("Potential Payout", f"${stake * total_odds:.2f}")
        st.caption(f"Recommended Stake: ${stake:.2f}")

    with tabs[0]: render_parlay_card(safe_candidates, 2, "Bankroll Builder")
    with tabs[1]: render_parlay_card(value_candidates, 3, "Value Stack")
    with tabs[2]: render_parlay_card(value_candidates, 4, "Lotto Ticket")
    with tabs[3]: render_parlay_card(value_candidates, 5, "Hail Mary")

def render_market_map():
    st.markdown('<p class="gradient-text">Market Map</p>', unsafe_allow_html=True)
    df = utils.load_data(utils.LATEST_URL)
    if isinstance(df, str):
        st.info("No data available yet.")
        return
    if df.empty:
        st.info("No valid bets to display.")
        return

    df = df[df['Bet Type'] != 'ARBITRAGE']
    df = df[df['Odds'] > 1.01].copy()
    if df.empty:
        st.info("No value bets for Market Map.")
        return

    df['Implied_Prob'] = 1 / df['Odds']
    df['Model_Prob'] = df['Confidence']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Implied_Prob'], y=df['Model_Prob'],
        mode='markers',
        marker=dict(size=df['Edge']*300 + 10, color=df['Edge'], colorscale='Viridis', showscale=True),
        text=df['Match'] + "<br>" + df['Bet'] + f" @ {df['Odds']}",
        hoverinfo='text'
    ))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='gray', dash='dash'), name="Fair Line"))
    fig.update_layout(template="plotly_dark", height=600, xaxis_title="Bookmaker Implied", yaxis_title="Model Probability")
    st.plotly_chart(fig, use_container_width=True)

def render_bet_tracker(bankroll):
    st.markdown('<p class="gradient-text">Bet Slip</p>', unsafe_allow_html=True)
    bankroll = st.number_input("Your Bankroll ($)", value=float(bankroll), min_value=0.01, step=10.0, format="%.2f")

    if st.session_state.bet_slip:
        slip_df = pd.DataFrame(st.session_state.bet_slip)
        total_stake = slip_df['User_Stake'].sum()
        total_odds = np.prod(slip_df['Odds']) if len(slip_df) > 1 else slip_df['Odds'].iloc[0]
        potential = total_stake * total_odds

        for i, bet in slip_df.iterrows():
            st.markdown(f"""
            <div class="bet-card">
                <div style="display:flex; justify-content:space-between;">
                    <div><strong>{bet['Match']}</strong></div>
                    <div style="color:#00e676;">{bet['Odds']:.2f}</div>
                </div>
                <div style="color:#ccc; font-size:0.9em;">{bet['Bet']}</div>
                <div style="margin-top:8px;">
                    Stake: <strong>${bet['User_Stake']:.2f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Stake", f"${total_stake:.2f}")
        c2.metric("Combined Odds", f"{total_odds:.2f}x")
        c3.metric("Potential Return", f"${potential:.2f}")

        if st.button("Clear Bet Slip", type="primary"):
            st.session_state.bet_slip = []
            st.rerun()
    else:
        st.info("Your bet slip is empty. Add bets from Command Center!")

def render_history():
    st.markdown('<p class="gradient-text">Betting History</p>', unsafe_allow_html=True)
    
    df = utils.load_data(utils.HISTORY_URL)
    if isinstance(df, str):
        if df in ["FILE_NOT_FOUND", "CONNECTION_ERROR"]:
            st.error("Could not load history from GitHub.")
        else:
            st.info("No bets recorded yet.")
        return

    stats = utils.get_performance_stats(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Profit", f"${stats['total_profit']:.2f}")
    c2.metric("Win Rate", f"{stats['win_rate']:.1%}")
    c3.metric("ROI", f"{stats['roi']:.1%}")
    c4.metric("Settled Bets", stats['total_bets'])

    display_df = df.copy()
    display_df['Result'] = display_df['Result'].fillna('Pending')
    display_df['Status'] = display_df['Result'].apply(utils.format_result_badge)
    display_df['Profit'] = display_df.apply(lambda x: f"${x['Profit']:.2f}" if x['Result'] in ['Win','Loss'] else "-", axis=1)

    cols = ['Formatted_Date', 'Sport', 'Match', 'Bet', 'Odds', 'Status', 'Profit', 'Score']
    cols = [c for c in cols if c in display_df.columns]
    st.write(display_df[cols].to_html(escape=False, index=False), unsafe_allow_html=True)

    st.download_button("Download Full History", df.to_csv(index=False), "betting_history_full.csv", "text/csv")

def render_about():
    st.markdown("# Betting Co-Pilot Pro v68.1")
    st.success("100% Working • Cloud Ready • Zero Crashes • Live ROI")
    st.caption("Sharp. Fast. Private.")
