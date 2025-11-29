# views.py
# The "Unlimited" Layouts (v65.1)
# Fixes: Relaxed filters so "Smart Picks" always show up

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import utils

def render_dashboard(bankroll, kelly_multiplier):
    st.markdown('<p class="gradient-text">🚀 Live Command Center</p>', unsafe_allow_html=True)
    
    df = utils.load_data(utils.LATEST_URL)
    
    # --- 1. TILT CONTROL ---
    with st.expander("🛡️ Risk & Tilt Control", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            max_daily_risk = st.number_input("Max Daily Risk (% of Bankroll)", value=5.0, step=0.5)
        with c2:
            win_goal = st.number_input("Daily Profit Goal ($)", value=50.0, step=10.0)
        max_risk_dollars = bankroll * (max_daily_risk / 100)
        st.caption(f"🛑 Warning threshold: **${max_risk_dollars:.2f}** exposure.")

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

        # --- DYNAMIC SMART PICKS ENGINE ---
        st.markdown("### 🔥 Smart Daily Picks")
        
        # *** FIX: Relaxed filters so this section always appears ***
        # We removed (df['Odds'] < 4.0) so it shows high-odds bets if that's all we have.
        candidates = df[df['Bet Type'] != 'ARBITRAGE'].copy()

        if not candidates.empty:
            smart_picks = []
            
            # Slot 1: THE BANKER (Highest Confidence)
            banker = candidates.sort_values('Confidence', ascending=False).iloc[0]
            smart_picks.append({"Label": "🛡️ The Banker", "Row": banker, "Color": "#00e676", "Reason": "Highest Win Probability"})
            
            # Slot 2: THE VALUE PLAY (Highest Edge)
            remaining = candidates[candidates['Match'] != banker['Match']]
            if not remaining.empty:
                value_play = remaining.sort_values('Edge', ascending=False).iloc[0]
                smart_picks.append({"Label": "🚀 The Value Play", "Row": value_play, "Color": "#00C9FF", "Reason": "Maximum Mathematical Edge"})
            
            # Slot 3: THE DIVERSIFIER
            if len(smart_picks) == 2:
                used_matches = [p['Row']['Match'] for p in smart_picks]
                used_sports = [p['Row']['Sport'] for p in smart_picks]
                
                diversifier_candidates = candidates[~candidates['Match'].isin(used_matches)]
                diff_sport = diversifier_candidates[~diversifier_candidates['Sport'].isin(used_sports)]
                
                if not diff_sport.empty:
                    div_pick = diff_sport.sort_values('Edge', ascending=False).iloc[0]
                    smart_picks.append({"Label": f"⚖️ {div_pick['Sport']} Diversifier", "Row": div_pick, "Color": "#FFD700", "Reason": "Portfolio Balance"})
                elif not diversifier_candidates.empty:
                    div_pick = diversifier_candidates.sort_values('Edge', ascending=False).iloc[0]
                    smart_picks.append({"Label": "🔥 Heat Check", "Row": div_pick, "Color": "#FFD700", "Reason": "Strong Momentum"})

            # Slot 4: THE UNDERDOG
            used_matches = [p['Row']['Match'] for p in smart_picks]
            dogs = candidates[(~candidates['Match'].isin(used_matches)) & (candidates['Odds'] > 2.2)]
            if not dogs.empty:
                dog_pick = dogs.sort_values('Edge', ascending=False).iloc[0]
                smart_picks.append({"Label": "🐺 Top Underdog", "Row": dog_pick, "Color": "#ff4b4b", "Reason": "High Upside Value"})

            # Render Dynamic Columns
            cols = st.columns(len(smart_picks))
            for idx, pick in enumerate(smart_picks):
                row = pick['Row']
                sport_icon = utils.get_team_emoji(row.get('Sport', 'Soccer'))
                rec_stake = bankroll * row.get('Stake', 0.01) * (kelly_multiplier / 0.25)
                
                with cols[idx]:
                    st.markdown(f"""
                    <div style="background-color: #1e2130; border: 1px solid {pick['Color']}; border-radius: 12px; padding: 15px; text-align: center; height: 100%;">
                        <div style="color: {pick['Color']}; font-weight: bold; text-transform: uppercase; font-size: 0.8em; margin-bottom: 5px;">{pick['Label']}</div>
                        <div style="font-size: 1.0em; font-weight: bold; margin-bottom: 5px;">{sport_icon} {row['Match']}</div>
                        <div style="color: white; font-weight: bold; font-size: 1.1em; margin-bottom: 10px;">{row['Bet']}</div>
                        <div style="display: flex; justify-content: center; gap: 15px; margin-bottom: 10px;">
                            <div><div style="font-size: 0.7em; color: #888;">ODDS</div><div style="font-weight: bold; color: #fff;">{row['Odds']:.2f}</div></div>
                            <div><div style="font-size: 0.7em; color: #888;">EDGE</div><div style="font-weight: bold; color: {pick['Color']};">{row['Edge']:.1%}</div></div>
                        </div>
                        <div style="margin-top: 10px; font-size: 0.9em;">Stake: <span style="color: {pick['Color']}; font-weight: bold;">${rec_stake:.2f}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No bets available for Smart Picks.")

        # --- THE MAIN FEED ---
        st.markdown("---")
        st.markdown("### 📋 Full Recommendations")
        
        current_exposure = 0.0
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
            
            # Circuit Breaker Logic
            current_exposure += cash_stake
            risk_warning = ""
            if current_exposure > max_risk_dollars:
                risk_warning = f"⚠️ **SKIP:** Daily risk limit (${max_risk_dollars:.2f}) exceeded."
                cash_stake = 0
            
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
                    if risk_warning: st.error(risk_warning)
                
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

        # --- EXPANDED PARLAY BUILDER (5+ LEGS) ---
        st.markdown("---")
        st.subheader("🧩 Smart Parlay Builder")
        
        # Relaxed parlay filters to ensure it shows up
        parlay_candidates = df[(df['Odds'] > 1.1) & (df['Bet Type'] != 'ARBITRAGE')]
        
        if len(parlay_candidates) >= 2:
            value_legs = parlay_candidates.sort_values('Edge', ascending=False).to_dict('records')
            safe_legs = parlay_candidates.sort_values('Confidence', ascending=False).to_dict('records')

            tab1, tab2, tab3, tab4 = st.tabs(["🛡️ Safe (2-Leg)", "🚀 Value (3-Leg)", "🎰 Lotto (4-Leg)", "☄️ Hail Mary (5-Leg)"])

            def render_parlay_card(legs_source, num_legs, title):
                pool = legs_source[:12]
                if len(pool) < num_legs:
                    st.info(f"Not enough bets found for a {num_legs}-leg parlay.")
                    return

                best_combo = None
                best_score = -1
                
                for combo in combinations(pool, num_legs):
                    if len(set([c['Match'] for c in combo])) == num_legs:
                        score = (np.prod([1 + c['Edge'] for c in combo])) - 1
                        if score > best_score:
                            best_score = score
                            best_combo = combo
                
                if best_combo:
                    tot_odds = np.prod([c['Odds'] for c in best_combo])
                    tot_prob = np.prod([c['Confidence'] for c in best_combo])
                    kelly_stake_pct = (best_score / (tot_odds - 1)) * kelly_multiplier if tot_odds > 1 else 0
                    kelly_stake_cash = bankroll * kelly_stake_pct
                    
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown(f"#### {title}")
                        for leg in best_combo:
                            st.markdown(f"• **{leg['Bet']}** @ {leg['Odds']:.2f} ({leg['Match']})")
                    with c2:
                        st.metric("Total Odds", f"{tot_odds:.2f}")
                        st.metric("Win Prob", f"{tot_prob:.1%}")
                        st.metric("Rec. Stake", f"${kelly_stake_cash:.2f}")
                    
                    user_stake = st.number_input(f"Wager ($) - {title}", min_value=0.0, value=float(int(kelly_stake_cash)) if kelly_stake_cash > 1 else 5.0, step=5.0)
                    st.success(f"💰 Potential Payout: **${user_stake * tot_odds:.2f}**")
                else:
                    st.info("Could not build a valid parlay (conflicting matches).")

            with tab1: render_parlay_card(safe_legs, 2, "Bankroll Builder")
            with tab2: render_parlay_card(value_legs, 3, "Value Stack")
            with tab3: render_parlay_card(value_legs, 4, "Lotto Ticket")
            with tab4: render_parlay_card(value_legs, 5, "Hail Mary")

        else:
            st.info("Not enough value bets to build a parlay.")

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
    bankroll = st.number_input("Your Bankroll ($)", value=float(bankroll), min_value=0.0, step=0.01, format="%.2f", key="tracker_bankroll")
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
    st.markdown("# 📖 About"); st.info("Betting Co-Pilot v65.1 (Unlimited Edition)")
