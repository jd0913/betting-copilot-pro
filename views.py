# views.py
# Contains the layout logic for each page.

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from itertools import combinations
import utils # Import the tools we just made

def render_dashboard(bankroll, kelly_multiplier):
    st.markdown('<p class="gradient-text">🚀 Live Command Center</p>', unsafe_allow_html=True)
    df = utils.load_data(utils.LATEST_URL)
    
    if isinstance(df, pd.DataFrame):
        # Filters
        sports = ["All"] + list(df['Sport'].unique()) if 'Sport' in df.columns else ["All"]
        selected_sport = st.sidebar.selectbox("Filter Sport", sports)
        if selected_sport != "All": df = df[df['Sport'] == selected_sport]
        
        st.markdown("### 📋 Actionable Recommendations")
        
        if not df.empty:
            df['key'] = df['Match'] + "_" + df['Bet']

        for i, row in df.iterrows():
            sport_icon = utils.get_team_emoji(row.get('Sport', 'Soccer'))
            match_time = row.get('Formatted_Date', 'Time TBD')
            risk_badge = utils.get_risk_badge(row)
            bookie_info = row.get('Info', 'Check Books')
            if pd.isna(bookie_info): bookie_info = "Check Books"
            
            stake_pct = row.get('Stake', 0.01)
            cash_stake = bankroll * stake_pct * (kelly_multiplier / 0.25)
            
            with st.container():
                st.markdown(f"""
                <div class="bet-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="font-size:0.8em; color:#888;">{match_time} • {row.get('League', '')}</div>
                            <div style="font-size:1.2em; font-weight:bold;">{sport_icon} {row['Match']}</div>
                            <div style="margin-top:5px;">{risk_badge} <span style="font-weight:bold; color:#00C9FF;">{row['Bet']}</span></div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:1.5em; font-weight:bold;">{row['Odds']:.2f}</div>
                            <div style="font-size:0.8em; color:#888;">Odds</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Edge", f"{row['Edge']:.2%}")
                c2.metric("Confidence", f"{row['Confidence']:.2%}")
                c3.metric("Bet Size", f"${cash_stake:.2f}", delta=f"{stake_pct:.2%}")
                
                with st.expander(f"🔍 Deep Dive & Bet Slip: {row['Match']}"):
                    dd1, dd2 = st.columns(2)
                    with dd1:
                        st.markdown("**Analysis Breakdown:**")
                        st.write(f"- **Implied Probability:** {(1/row['Odds']):.2%}")
                        st.write(f"- **Model Probability:** {row['Confidence']:.2%}")
                        st.write(f"- **Bookmaker:** {bookie_info}")
                    with dd2:
                        if 'News Alert' in row and pd.notna(row['News Alert']):
                            st.error(f"**News Alert:** {row['News Alert']}")
                        else:
                            st.success("No critical injury news detected.")
                        
                        key = row['key']
                        is_in_slip = any(b['key'] == key for b in st.session_state.bet_slip)
                        if st.checkbox("Add to my personal Bet Slip", value=is_in_slip, key=key):
                            if not is_in_slip:
                                row_data = row.to_dict()
                                row_data['User_Stake'] = cash_stake
                                st.session_state.bet_slip.append(row_data)
                                st.rerun()
                        else:
                            if is_in_slip:
                                st.session_state.bet_slip = [b for b in st.session_state.bet_slip if b['key'] != key]
                                st.rerun()
        
        # Parlay Builder
        st.markdown("---")
        st.subheader("🧩 Smart Parlay Builder")
        if len(df) >= 2:
            parlay_legs = df.sort_values('Edge', ascending=False).to_dict('records')
            best_parlay_edge = -1
            best_parlay_combo = None
            for combo in combinations(parlay_legs[:5], 2): 
                if combo[0]['Match'] != combo[1]['Match']:
                    parlay_edge = ((1 + combo[0]['Edge']) * (1 + combo[1]['Edge'])) - 1
                    if parlay_edge > best_parlay_edge:
                        best_parlay_edge = parlay_edge
                        best_parlay_combo = combo
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
        df['Implied'] = 1 / df['Odds']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Fair Value', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(
            x=df['Implied'], y=df['Confidence'], mode='markers',
            marker=dict(size=df['Edge']*100 + 10, color=df['Edge'], colorscale='RdYlGn', showscale=True, colorbar=dict(title="Edge")),
            text=df['Match'] + '<br>' + df['Bet'], hoverinfo='text'
        ))
        fig.update_layout(title="Market Inefficiency Map", xaxis_title="Bookmaker Implied Probability", yaxis_title="Co-Pilot Calculated Probability", height=600, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No data loaded.")

def render_bet_tracker(bankroll):
    st.markdown('<p class="gradient-text">🎟️ Personal Bet Slip</p>', unsafe_allow_html=True)
    if st.session_state.bet_slip:
        slip_df = pd.DataFrame(st.session_state.bet_slip)
        display_cols = ['Match', 'Bet', 'Odds', 'Edge', 'Confidence']
        if 'Info' in slip_df.columns: display_cols.append('Info')
        st.dataframe(slip_df[display_cols].style.format({'Odds': '{:.2f}', 'Edge': '{:.2%}', 'Confidence': '{:.2%}'}))
        
        total_stake = 0
        potential_profit = 0
        for bet in st.session_state.bet_slip:
            cash_stake = bet.get('User_Stake', bankroll * 0.01)
            total_stake += cash_stake
            potential_profit += cash_stake * (bet['Odds'] - 1)
            
        c1, c2 = st.columns(2)
        c1.metric("Total Stake Required", f"${total_stake:.2f}")
        c2.metric("Total Potential Profit", f"${potential_profit:.2f}")
        
        if st.button("Clear Bet Slip"):
            st.session_state.bet_slip = []
            st.rerun()
    else: st.info("Your bet slip is empty.")

def render_history():
    st.markdown('<p class="gradient-text">📜 Performance Archive</p>', unsafe_allow_html=True)
    df = utils.load_data(utils.HISTORY_URL)
    if isinstance(df, pd.DataFrame):
        if 'Result' not in df.columns:
            st.info("No results settled yet.")
            st.dataframe(df)
            return
        
        settled = df[df['Result'].isin(['Win', 'Loss', 'Push'])]
        if not settled.empty:
            total_profit = settled['Profit'].sum()
            win_rate = len(settled[settled['Result'] == 'Win']) / len(settled)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Profit (Units)", f"{total_profit:.2f}")
            c2.metric("Win Rate", f"{win_rate:.1%}")
            c3.metric("Total Bets", len(settled))
            st.divider()

        display_df = df.copy()
        display_df['Result'] = display_df['Result'].fillna('Pending')
        display_df['Status'] = display_df['Result'].apply(utils.format_result_badge)
        cols = ['Formatted_Date', 'Sport', 'Match', 'Bet', 'Odds', 'Status', 'Profit']
        display_df = display_df.rename(columns={'Formatted_Date': 'Date'})
        cols = [c for c in cols if c in display_df.columns]
        st.write(display_df[cols].to_html(escape=False, index=False), unsafe_allow_html=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "history.csv", "text/csv")
    else: st.info("No history found.")

def render_about():
    st.title("📖 About")
    st.markdown("### Betting Co-Pilot Pro v42.0 (Enterprise Edition)")
    st.markdown("Automated quantitative analysis engine running on GitHub Actions.")
