# views.py
# The "Advanced Parlay" Layouts (v60.0)
# Features: Multi-Leg Parlays (3-5 legs), Anchor Logic, Cross-Sport Stacking

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import utils

def render_dashboard(bankroll, kelly_multiplier):
    st.markdown('<p class="gradient-text">🚀 Live Command Center</p>', unsafe_allow_html=True)
    
    df = utils.load_data(utils.LATEST_URL)
    
    if isinstance(df, pd.DataFrame):
        # --- FILTERS ---
        sports = ["All"] + list(df['Sport'].unique()) if 'Sport' in df.columns else ["All"]
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_sport = st.sidebar.selectbox("Filter Sport", sports)
        
        if selected_sport != "All": df = df[df['Sport'] == selected_sport]
        
        # --- KPI CARDS ---
        total_bets = len(df)
        top_edge = df['Edge'].max() if not df.empty else 0
        
        kpi_html = f"""
        <div style="display:flex; gap:10px; margin-bottom:20px;">
            <div style="flex:1; background:#1e2130; padding:15px; border-radius:10px; border:1px solid #2b2f44; text-align:center;">
                <div style="color:#8b92a5; font-size:0.8em; font-weight:bold;">ACTIVE BETS</div>
                <div style="font-size:1.8em; font-weight:800; color:white;">{total_bets}</div>
            </div>
            <div style="flex:1; background:#1e2130; padding:15px; border-radius:10px; border:1px solid #2b2f44; text-align:center;">
                <div style="color:#8b92a5; font-size:0.8em; font-weight:bold;">TOP EDGE</div>
                <div style="font-size:1.8em; font-weight:800; color:#00e676;">{top_edge:.1%}</div>
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

        # --- THE BETTING FEED ---
        st.markdown("### 📋 Actionable Recommendations")
        
        if not df.empty:
            df['key'] = df['Match'] + "_" + df['Bet']

        for i, row in df.iterrows():
            sport_icon = utils.get_team_emoji(row.get('Sport', 'Soccer'))
            match_time = row.get('Formatted_Date', 'Time TBD')
            risk_badge = utils.get_risk_badge(row)
            bookie = row.get('Info', 'Best Price')
            if pd.isna(bookie): bookie = "Best Price"
            
            stake_pct = row.get('Stake', 0.01)
            cash_stake = bankroll * stake_pct * (kelly_multiplier / 0.25)
            
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
                        if row['Odds'] > 0: st.write(f"**Implied:** {(1/row['Odds']):.1%}")
                        else: st.write("**Implied:** N/A")
                        key = row['key']
                        is_in_slip = any(b['key'] == key for b in st.session_state.bet_slip)
                        if st.checkbox("Add to Slip", value=is_in_slip, key=key):
                            if not is_in_slip:
                                row_data = row.to_dict(); row_data['User_Stake'] = cash_stake
                                st.session_state.bet_slip.append(row_data); st.rerun()
                            else:
                                if is_in_slip:
                                    st.session_state.bet_slip = [b for b in st.session_state.bet_slip if b['key'] != key]; st.rerun()

        # --- ADVANCED PARLAY ENGINE ---
        st.markdown("---")
        st.subheader("🧩 Advanced Parlay Engine")
        
        # Filter valid parlay candidates (No Arbs, Odds > 1.1)
        candidates = df[(df['Odds'] > 1.1) & (df['Bet Type'] != 'ARBITRAGE')]
        
        if len(candidates) >= 2:
            # Strategy 1: The "Anchor" (Safe 2-Leg)
            # Logic: High Confidence (>60%), Low Odds (<1.80)
            anchors = candidates[(candidates['Confidence'] > 0.60) & (candidates['Odds'] < 1.80)].sort_values('Confidence', ascending=False).to_dict('records')
            
            # Strategy 2: The "Value Stack" (3-Leg)
            # Logic: High Edge (>10%), Medium Odds (1.5 - 2.5)
            value_plays = candidates[(candidates['Edge'] > 0.10) & (candidates['Odds'].between(1.5, 2.5))].sort_values('Edge', ascending=False).to_dict('records')
            
            # Strategy 3: The "Lotto" (4-Leg)
            # Logic: High Odds (>2.5), Positive Edge
            lottos = candidates[(candidates['Odds'] > 2.5) & (candidates['Edge'] > 0)].sort_values('Edge', ascending=False).to_dict('records')

            tab1, tab2, tab3 = st.tabs(["🛡️ Safe Anchors", "🚀 Value Stack", "🎰 Lotto Ticket"])

            with tab1:
                if len(anchors) >= 2:
                    combo = anchors[:2] # Top 2 safest
                    if combo[0]['Match'] != combo[1]['Match']:
                        tot_odds = combo[0]['Odds'] * combo[1]['Odds']
                        prob = combo[0]['Confidence'] * combo[1]['Confidence']
                        st.success(f"**The Bankroll Builder** | Odds: {tot_odds:.2f} | Win Prob: {prob:.1%}")
                        for leg in combo: st.write(f"• {leg['Match']} -> **{leg['Bet']}** ({leg['Odds']})")
                    else: st.info("Need distinct matches for a parlay.")
                else: st.info("Not enough high-confidence anchors found.")

            with tab2:
                if len(value_plays) >= 3:
                    # Find valid 3-leg combo (distinct matches)
                    valid_combo = []
                    seen_matches = set()
                    for play in value_plays:
                        if play['Match'] not in seen_matches:
                            valid_combo.append(play)
                            seen_matches.add(play['Match'])
                        if len(valid_combo) == 3: break
                    
                    if len(valid_combo) == 3:
                        tot_odds = np.prod([p['Odds'] for p in valid_combo])
                        tot_edge = (np.prod([1+p['Edge'] for p in valid_combo])) - 1
                        st.warning(f"**The Value Stack** | Odds: {tot_odds:.2f} | Combined Edge: {tot_edge:.1%}")
                        for leg in valid_combo: st.write(f"• {leg['Match']} -> **{leg['Bet']}** ({leg['Odds']})")
                    else: st.info("Not enough distinct value plays.")
                else: st.info("Not enough value plays found.")

            with tab3:
                if len(lottos) >= 4:
                    # Find valid 4-leg combo
                    valid_combo = []
                    seen_matches = set()
                    for play in lottos:
                        if play['Match'] not in seen_matches:
                            valid_combo.append(play)
                            seen_matches.add(play['Match'])
                        if len(valid_combo) == 4: break
                    
                    if len(valid_combo) == 4:
                        tot_odds = np.prod([p['Odds'] for p in valid_combo])
                        st.error(f"**The Moonshot** | Odds: {tot_odds:.2f} | High Risk")
                        for leg in valid_combo: st.write(f"• {leg['Match']} -> **{leg['Bet']}** ({leg['Odds']})")
                    else: st.info("Not enough longshots found.")
                else: st.info("Not enough longshots found.")

        else:
            st.info("Not enough bets to build parlays.")

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
    st.markdown("# 📖 About"); st.info("Betting Co-Pilot v60.0 (Advanced Parlay Edition)")
