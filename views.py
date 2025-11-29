# views.py
# The "Self-Aware" Layouts (v62.0)
# Features: Live Performance Ticker, Sport-Specific Accuracy Context

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import utils

def render_dashboard(bankroll, kelly_multiplier):
    st.markdown('<p class="gradient-text">🚀 Live Command Center</p>', unsafe_allow_html=True)
    
    # Load Data
    df = utils.load_data(utils.LATEST_URL)
    history_df = utils.load_data(utils.HISTORY_URL)
    
    # Calculate Live Stats
    stats = utils.get_performance_stats(history_df)
    
    # --- 1. SYSTEM HEALTH & PERFORMANCE TICKER ---
    # This makes the model "Self-Aware"
    st.markdown("### 🧠 System Intelligence")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Real-World Win Rate", f"{stats['win_rate']:.1%}", delta="Live Performance")
    k2.metric("Real-World ROI", f"{stats['roi']:.1%}", delta="Profitability")
    k3.metric("Knowledge Base", f"{stats['total_bets'] + 15000:,} Matches", help="Historical training data + Live tracked bets")
    k4.metric("Active Opportunities", len(df) if isinstance(df, pd.DataFrame) else 0)
    st.markdown("---")

    if isinstance(df, pd.DataFrame):
        # --- FILTERS ---
        sports = ["All"] + list(df['Sport'].unique()) if 'Sport' in df.columns else ["All"]
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_sport = st.sidebar.selectbox("Filter Sport", sports)
        
        if selected_sport != "All": df = df[df['Sport'] == selected_sport]
        
        # --- THE BETTING FEED ---
        st.markdown("### 📋 Actionable Recommendations")
        
        if not df.empty:
            df['key'] = df['Match'] + "_" + df['Bet']

        for i, row in df.iterrows():
            sport = row.get('Sport', 'Soccer')
            sport_icon = utils.get_team_emoji(sport)
            match_time = row.get('Formatted_Date', 'Time TBD')
            risk_badge = utils.get_risk_badge(row)
            bookie = row.get('Info', 'Best Price')
            if pd.isna(bookie): bookie = "Best Price"
            
            stake_pct = row.get('Stake', 0.01)
            cash_stake = bankroll * stake_pct * (kelly_multiplier / 0.25)
            
            # Get Sport-Specific Accuracy
            sport_acc = stats['sport_stats'].get(sport, 0.0)
            sport_acc_str = f"{sport_acc:.0%}" if sport_acc > 0 else "New"
            
            with st.container():
                c1, c2 = st.columns([3, 1])
                
                with c1:
                    st.markdown(f"""
                    <div class="bet-ticket">
                        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                            <span style="color:#8b92a5; font-size:0.8em;">{match_time} • {row.get('League', 'League')}</span>
                            {risk_badge}
                        </div>
                        <div style="font-size:1.3em; font-weight:700; margin-bottom:5px;">
                            {sport_icon} {row['Match']}
                        </div>
                        <div style="display:flex; align-items:center; gap:10px;">
                            <span style="color:#00C9FF; font-weight:600;">{row['Bet']}</span>
                            <span style="color:#555;">|</span>
                            <span style="color:#8b92a5; font-size:0.9em;">{bookie}</span>
                        </div>
                        <div style="margin-top:15px; display:flex; gap:20px;">
                            <div><div class="metric-label">EDGE</div><div class="metric-value" style="color:#00e676;">{row['Edge']:.1%}</div></div>
                            <div><div class="metric-label">CONF</div><div class="metric-value">{row['Confidence']:.1%}</div></div>
                            <div><div class="metric-label">STAKE</div><div class="metric-value">${cash_stake:.2f}</div></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c2:
                    st.markdown(f"""<div style="height:100%; display:flex; align-items:center; justify-content:center;"><div class="odds-box">{row['Odds']:.2f}</div></div>""", unsafe_allow_html=True)
                    
                    with st.expander("Details"):
                        st.write(f"**Implied:** {(1/row['Odds']):.1%}")
                        st.write(f"**Model Accuracy ({sport}):** {sport_acc_str}")
                        
                        key = row['key']
                        is_in_slip = any(b['key'] == key for b in st.session_state.bet_slip)
                        if st.checkbox("Add to Slip", value=is_in_slip, key=key):
                            if not is_in_slip:
                                row_data = row.to_dict(); row_data['User_Stake'] = cash_stake
                                st.session_state.bet_slip.append(row_data); st.rerun()
                        else:
                            if is_in_slip:
                                st.session_state.bet_slip = [b for b in st.session_state.bet_slip if b['key'] != key]; st.rerun()

        # --- SMART PARLAY BUILDER ---
        st.markdown("---")
        st.subheader("🧩 Smart Parlay Builder")
        
        parlay_candidates = df[(df['Odds'] > 1.1) & (df['Odds'] < 3.0) & (df['Bet Type'] != 'ARBITRAGE')]
        
        if len(parlay_candidates) >= 2:
            # 1. Bankroll Builder (Safe)
            safe_legs = parlay_candidates.sort_values('Confidence', ascending=False).to_dict('records')
            safe_combo = None
            for combo in combinations(safe_legs[:6], 2):
                if combo[0]['Match'] != combo[1]['Match']: safe_combo = combo; break
            
            # 2. Value Stack (Medium)
            value_legs = parlay_candidates.sort_values('Edge', ascending=False).to_dict('records')
            value_combo = None
            for combo in combinations(value_legs[:6], 3):
                if len(set([c['Match'] for c in combo])) == 3: value_combo = combo; break

            # 3. Moonshot (High Risk)
            lotto_legs = df[(df['Odds'] > 2.5) & (df['Edge'] > 0)].sort_values('Edge', ascending=False).to_dict('records')
            lotto_combo = None
            for combo in combinations(lotto_legs[:8], 4):
                if len(set([c['Match'] for c in combo])) == 4: lotto_combo = combo; break

            tab1, tab2, tab3 = st.tabs(["🛡️ Safe (2-Leg)", "🚀 Value (3-Leg)", "🎰 Lotto (4-Leg)"])

            def render_parlay_card(combo, title):
                if not combo: st.info("Not enough bets found."); return
                tot_odds = np.prod([c['Odds'] for c in combo])
                tot_prob = np.prod([c['Confidence'] for c in combo])
                tot_edge = (tot_prob * tot_odds) - 1
                kelly_stake_pct = (tot_edge / (tot_odds - 1)) * kelly_multiplier if tot_odds > 1 else 0
                kelly_stake_cash = bankroll * kelly_stake_pct
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown(f"#### {title}")
                    for leg in combo: st.markdown(f"• **{leg['Bet']}** @ {leg['Odds']:.2f} ({leg['Match']})")
                with c2:
                    st.metric("Total Odds", f"{tot_odds:.2f}")
                    st.metric("Win Prob", f"{tot_prob:.1%}")
                    st.metric("Rec. Stake", f"${kelly_stake_cash:.2f}")
                
                user_stake = st.number_input(f"Wager ($) - {title}", value=float(int(kelly_stake_cash)) if kelly_stake_cash > 1 else 5.0, step=5.0)
                st.success(f"💰 Potential Payout: **${user_stake * tot_odds:.2f}**")

            with tab1: render_parlay_card(safe_combo, "Bankroll Builder")
            with tab2: render_parlay_card(value_combo, "Value Stack")
            with tab3: render_parlay_card(lotto_combo, "Moonshot Ticket")

        else: st.info("Not enough value bets to build a parlay.")

    elif df == "NO_BETS_FOUND":
        st.success("✅ System Online. Market Scanned. No Value Found.")
    else:
        st.error("Connection Error. Check GitHub configuration.")

def render_market_map():
    st.markdown('<p class="gradient-text">🗺️ Market Map</p>', unsafe_allow_html=True)
    df = utils.load_data(utils.LATEST_URL)
    if isinstance(df, pd.DataFrame):
        if 'Bet Type' in df.columns: df = df[df['Bet Type'] != 'ARBITRAGE']
        df = df[df['Odds'] > 0]
        if not df.empty:
            df['Implied'] = 1 / df['Odds']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Fair Value', line=dict(color='#444', dash='dash')))
            fig.add_trace(go.Scatter(x=df['Implied'], y=df['Confidence'], mode='markers', marker=dict(size=df['Edge']*150 + 10, color=df['Edge'], colorscale='Viridis', showscale=True), text=df['Match'] + '<br>' + df['Bet'], hoverinfo='text'))
            fig.update_layout(template="plotly_dark", height=600, xaxis_title="Implied Prob", yaxis_title="Model Prob")
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No valid bets for Market Map.")
    else: st.info("No data loaded.")

def render_bet_tracker(bankroll):
    st.markdown('<p class="gradient-text">🎟️ Bet Slip</p>', unsafe_allow_html=True)
    if st.session_state.bet_slip:
        slip_df = pd.DataFrame(st.session_state.bet_slip)
        total_stake = 0; potential_return = 0
        for i, bet in slip_df.iterrows():
            st.markdown(f"""<div class="bet-ticket" style="border-left: 4px solid #00C9FF;"><div style="display:flex; justify-content:space-between;"><div style="font-weight:bold;">{bet['Match']}</div><div style="color:#00e676;">{bet['Odds']:.2f}</div></div><div style="font-size:0.9em; color:#ccc;">{bet['Bet']}</div><div style="margin-top:10px; font-size:0.8em; color:#888;">Stake: <span style="color:white;">${bet.get('User_Stake', 0):.2f}</span></div></div>""", unsafe_allow_html=True)
            total_stake += bet.get('User_Stake', 0); potential_return += bet.get('User_Stake', 0) * bet['Odds']
        st.divider(); c1, c2 = st.columns(2)
        c1.metric("Total Stake", f"${total_stake:.2f}"); c2.metric("Potential Return", f"${potential_return:.2f}")
        if st.button("Clear Slip"): st.session_state.bet_slip = []; st.rerun()
    else: st.info("Your bet slip is empty.")

def render_history():
    st.markdown('<p class="gradient-text">📜 History</p>', unsafe_allow_html=True)
    st.markdown("""<style>th { text-align: center !important; } td { text-align: center !important; }</style>""", unsafe_allow_html=True)
    df = utils.load_data(utils.HISTORY_URL)
    if isinstance(df, pd.DataFrame):
        if 'Result' not in df.columns: st.info("No results settled yet."); st.dataframe(df); return
        settled = df[df['Result'].isin(['Win', 'Loss', 'Push'])]
        if not settled.empty:
            total_profit = settled['Profit'].sum(); win_rate = len(settled[settled['Result'] == 'Win']) / len(settled)
            c1, c2 = st.columns(2); c1.metric("Total Profit", f"{total_profit:.2f}u"); c2.metric("Win Rate", f"{win_rate:.1%}"); st.divider()
        display_df = df.copy(); display_df['Result'] = display_df['Result'].fillna('Pending'); display_df['Status'] = display_df['Result'].apply(utils.format_result_badge)
        display_df['Profit'] = np.where(display_df['Result'] == 'Pending', '-', display_df['Profit'].fillna(0.0).map('{:.2f}'.format))
        if 'Formatted_Date' in display_df.columns: display_df = display_df.rename(columns={'Formatted_Date': 'Match Time'})
        elif 'Date' in display_df.columns: display_df = display_df.rename(columns={'Date': 'Match Time'})
        cols = ['Match Time', 'Sport', 'Match', 'Bet', 'Odds', 'Status', 'Profit']
        cols = [c for c in cols if c in display_df.columns]
        st.write(display_df[cols].to_html(escape=False, index=False), unsafe_allow_html=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "history.csv", "text/csv")
    else: st.info("No history found.")

def render_about():
    st.markdown("# 📖 About"); st.info("Betting Co-Pilot v62.0 (Self-Aware Edition)")
