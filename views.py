# views.py
# The "Vegas Edition" Layouts (v53.1)
# Fixes: Added missing numpy import

import streamlit as st
import pandas as pd
import numpy as np  # <--- THIS WAS MISSING
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
                        if row['Odds'] > 0:
                            st.write(f"**Implied:** {(1/row['Odds']):.1%}")
                        else:
                            st.write("**Implied:** N/A")
                            
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
        
        parlay_candidates = df[df['Odds'] > 1.0]
        
        if len(parlay_candidates) >= 2:
            parlay_legs = parlay_candidates.sort_values('Edge', ascending=False).to_dict('records')
            best_parlay_edge = -1; best_parlay_combo = None
            for combo in combinations(parlay_legs[:5], 2): 
                if combo[0]['Match'] != combo[1]['Match']:
                    parlay_edge = ((1 + combo[0]['Edge']) * (1 + combo[1]['Edge'])) - 1
                    if parlay_edge > best_parlay_edge: best_parlay_edge = parlay_edge; best_parlay_combo = combo
            if best_parlay_combo:
                total_odds = best_parlay_combo[0]['Odds'] * best_parlay_combo[1]['Odds']
                st.success(f"🔥 **Top 2-Leg Parlay** | Total Odds: **{total_odds:.2f}** | Combined Edge: **{best_parlay_edge:.2%}**")
                c1, c2 = st.columns(2)
                with c1: st.info(f"Leg 1: {best_parlay_combo[0]['Match']} -> {best_parlay_combo[0]['Bet']}")
                with c2: st.info(f"Leg 2: {best_parlay_combo[1]['Match']} -> {best_parlay_combo[1]['Bet']}")
            else: st.info("Could not build a valid parlay.")
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
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='line
