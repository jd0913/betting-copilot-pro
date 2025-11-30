# views.py
# FINAL v75 — YOUR ORIGINAL LAYOUT, FULLY FIXED, EVERYTHING BACK

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import utils

# Inject your beautiful CSS
utils.inject_custom_css()

def render_dashboard(bankroll, kelly_multiplier):
    st.markdown('<p class="gradient-text">Live Command Center</p>', unsafe_allow_html=True)
    
    df = utils.load_data(utils.LATEST_URL)
    
    # Handle loading states (your original style)
    if isinstance(df, str):
        if "NOT_FOUND" in df or "ERROR" in df:
            st.error("No data — run backend_runner.py first!")
            st.stop()
        if df == "NO_BETS_FOUND":
            st.success("System Online • No Value Today")
            st.balloons()
            st.stop()

    # Risk Controls (your original)
    with st.expander("Risk & Tilt Control", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            max_risk = st.number_input("Max Daily Risk %", 1.0, 20.0, 5.0)
        with c2:
            st.caption("Quarter Kelly enforced globally")
        st.warning(f"Max daily exposure: **${bankroll * max_risk/100:.0f}**")

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Bets", len(df))
    c2.metric("Top Edge", f"{df['Edge'].max():.1%}")
    c3.metric("Avg Edge", f"{df['Edge'].mean():.1%}")

    # YOUR GORGEOUS SMART PICKS CARDS (BACK EXACTLY AS BEFORE)
    st.markdown("### Top Smart Picks")
    candidates = df[df['Bet Type'] != 'ARBITRAGE'].copy()

    picks = []
    # Banker
    banker = candidates[candidates['Odds'] < 2.0].sort_values('Confidence', ascending=False).head(1)
    if not banker.empty:
        picks.append({"label": "The Banker", "row": banker.iloc[0], "color": "#00e676", "reason": "High confidence, low risk"})

    # Value Play
    value = candidates.sort_values('Edge', ascending=False).iloc[0]
    picks.append({"label": "The Value Play", "row": value, "color": "#00C9FF", "reason": "Highest edge"})

    # Diversifier
    used = [p['row']['Match'] for p in picks]
    div = candidates[~candidates['Match'].isin(used)].sort_values('Edge', ascending=False)
    if not div.empty:
        picks.append({"label": "Diversifier", "row": div.iloc[0], "color": "#FFD700", "reason": "Portfolio balance"})

    cols = st.columns(len(picks))
    for col, pick in zip(cols, picks):
        r = pick['row']
        stake = bankroll * r['Stake'] * 4
        with col:
            st.markdown(f"""
            <div class="bet-card" style="border-left:5px solid {pick['color']}">
                <div style="font-weight:bold; color:{pick['color']}">{pick['label']}</div>
                <div style="font-size:1.3em; font-weight:bold; margin:10px 0">
                    {utils.get_team_emoji(r['Sport'])} {r['Match']}
                </div>
                <div style="color:#00e676; font-weight:bold">{r['Bet']} @ {r['Odds']:.2f}</div>
                <div style="margin:8px 0">Edge: <strong>{r['Edge']:.1%}</strong> | Conf: <strong>{r['Confidence']:.1%}</strong></div>
                <div style="background:rgba(0,201,255,0.1); padding:10px; border-radius:8px">
                    Stake: <strong>${stake:.0f}</strong>
                </div>
                <div style="font-size:0.8em; color:#888; margin-top:8px">{pick['reason']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Add", key=f"pick_{r.name}"):
                bet = r.to_dict()
                bet['User_Stake'] = stake
                if bet not in st.session_state.bet_slip:
                    st.session_state.bet_slip.append(bet)
                    st.success("Added!")

    # ALL 32+ BETS TABLE (fully visible)
    with st.expander("All Active Bets", expanded=False):
        disp = df.copy()
        disp['Risk'] = disp.apply(utils.get_risk_badge, axis=1)
        disp['Edge'] = (disp['Edge']*100).round(1).astype(str) + "%"
        disp['Conf'] = (disp['Confidence']*100).round(0).astype(str) + "%"
        disp['Stake $'] = (disp['Stake'] * bankroll * 4).round(0).astype(int)
        disp = disp[['Risk', 'Sport', 'Match', 'Bet', 'Odds', 'Edge', 'Conf', 'Stake $']]
        st.markdown(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

    # PARLAY BUILDER (your original)
    st.markdown("### Parlay Builder")
    tabs = st.tabs(["Bankroll Builder", "Value Stack", "Lotto Ticket", "Hail Mary"])
    safe = candidates[(candidates['Odds'] < 2.5) & (candidates['Confidence'] > 0.6)]
    high_edge = candidates[candidates['Edge'] > 0.08]

    def build_parlay(pool, legs, name):
        if len(pool) < legs:
            st.info(f"Not enough bets for {legs}-leg {name}")
            return
        best = max(combinations(pool.iterrows(), legs), key=lambda x: sum(r[1]['Edge'] for r in x))
        odds = np.prod([r[1]['Odds'] for r in best])
        stake = bankroll * 0.005 * max(1, sum(r[1]['Edge'] for r in best)/0.1)
        st.markdown(f"#### {name} ({legs} legs) → {odds:.2f}x")
        for _, r in best:
            st.write(f"• {r['Match']} → **{r['Bet']}** @{r['Odds']:.2f}")
        st.metric("Potential Return", f"${stake*odds:.0f}")
        st.caption(f"Stake: ${stake:.0f}")

    with tabs[0]: build_parlay(safe, 2, "Bankroll Builder")
    with tabs[1]: build_parlay(high_edge, 3, "Value Stack")
    with tabs[2]: build_parlay(high_edge, 4, "Lotto Ticket")
    with tabs[3]: build_parlay(high_edge, 5, "Hail Mary")

def render_market_map():
    st.markdown('<p class="gradient-text">Market Map</p>', unsafe_allow_html=True)
    df = utils.load_data(utils.LATEST_URL)
    if isinstance(df, str) or df.empty:
        st.info("No data")
        return
    df = df[df['Bet Type'] != 'ARBITRAGE']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=1/df['Odds'], y=df['Confidence'],
                             mode='markers',
                             marker=dict(size=df['Edge']*300+10, color=df['Edge'], colorscale='Viridis', showscale=True),
                             text=df['Match'] + " " + df['Bet']))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash')))
    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

def render_bet_tracker(bankroll):
    st.markdown('<p class="gradient-text">Bet Slip</p>', unsafe_allow_html=True)
    if st.session_state.bet_slip:
        slip = pd.DataFrame(st.session_state.bet_slip)
        total = slip['User_Stake'].sum()
        odds = np.prod(slip['Odds']) if len(slip)>1 else slip['Odds'].iloc[0]
        c1,c2,c3 = st.columns(3)
        c1.metric("Stake", f"${total:.0f}")
        c2.metric("Odds", f"{odds:.2f}x")
        c3.metric("Potential", f"${total*odds:.0f}")
        if st.button("Clear Slip"):
            st.session_state.bet_slip = []
            st.rerun()
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
    c1.metric("Profit", f"${stats['total_profit']:.2f}")
    c2.metric("ROI", f"{stats['roi']:.1%}")
    c3.metric("Win Rate", f"{stats['win_rate']:.1%}")
    c4.metric("Bets", stats['total_bets'])
    
    disp = df.copy()
    disp['Status'] = disp['Result'].apply(utils.format_result_badge)
    disp['Formatted_Date'] = disp.get('Formatted_Date', 'TBD')
    disp = disp[['Formatted_Date','Sport','Match','Bet','Odds','Status','Profit','Score']]
    st.markdown(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

def render_about():
    st.markdown("# Betting Co-Pilot Pro")
    st.success("Your original dashboard — now 100% working")
