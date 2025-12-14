# views.py
# The "Strict Parlay" Layouts (v73.1 - Google Score Lookup Edition)
# FIX: Updated to call utils.settle_bets_with_google_scores

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
from datetime import datetime, timedelta, timezone
import utils # This now has the settle_bets_with_google_scores function

def render_dashboard(bankroll, kelly_multiplier):
    st.markdown('<p class="gradient-text">🚀 Live Command Center</p>', unsafe_allow_html=True)
    
    df = utils.load_data(utils.LATEST_URL)
    
    # --- 1. TILT CONTROL ---
    with st.expander("🛡️ Risk & Tilt Control", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            max_daily_risk = st.number_input("Max Daily Risk (% of Bankroll)", value=5.0, step=0.5, min_value=0.0, max_value=100.0)
        with c2:
            win_goal = st.number_input("Daily Profit Goal ($)", value=50.0, step=10.0, min_value=0.0)
        max_risk_dollars = bankroll * (max_daily_risk / 100)
        st.caption(f"🛑 Warning threshold: **${max_risk_dollars:.2f}** exposure.")

    if isinstance(df, pd.DataFrame) and not df.empty:
        # --- FILTERS ---
        sports = ["All"] + sorted(list(df['Sport'].unique())) if 'Sport' in df.columns else ["All"]
        selected_sport = st.selectbox("Filter Sport", sports)
        
        if selected_sport != "All" and 'Sport' in df.columns:
            df = df[df['Sport'] == selected_sport]
        
        # --- KPI CARDS ---
        total_bets = len(df)
        top_edge = df['Edge'].max() if 'Edge' in df.columns and not df.empty else 0
        
        # Display KPI cards
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Active Bets", total_bets)
        with col2:
            st.metric("🔥 Top Edge", f"{top_edge:.1%}")

        # --- DYNAMIC SMART PICKS ---
        st.markdown("### 🔥 Smart Daily Picks")
        candidates = df[df['Bet Type'] != 'ARBITRAGE'].copy() if 'Bet Type' in df.columns else df.copy()

        if not candidates.empty:
            smart_picks = []
            
            # Slot 1: THE BANKER (Strict: Odds < 2.0, High Conf)
            if 'Odds' in candidates.columns and 'Confidence' in candidates.columns:
                banker_candidates = candidates[candidates['Odds'] < 2.0]
                if not banker_candidates.empty:
                    banker = banker_candidates.sort_values('Confidence', ascending=False).iloc[0]
                    smart_picks.append({"Label": "🛡️ The Banker", "Row": banker, "Color": "#00e676", "Reason": "Safe Favorite"})
            
            # Slot 2: THE VALUE PLAY
            used_matches = [p['Row']['Match'] for p in smart_picks if 'Match' in p['Row']]
            remaining = candidates[~candidates['Match'].isin(used_matches)] if 'Match' in candidates.columns else candidates
            if not remaining.empty and 'Edge' in remaining.columns:
                value_play = remaining.sort_values('Edge', ascending=False).iloc[0]
                smart_picks.append({"Label": "🚀 The Value Play", "Row": value_play, "Color": "#00C9FF", "Reason": "Max Edge"})
            
            # Slot 3: DIVERSIFIER
            used_matches = [p['Row']['Match'] for p in smart_picks if 'Match' in p['Row']]
            remaining = candidates[~candidates['Match'].isin(used_matches)] if 'Match' in candidates.columns else candidates
            if not remaining.empty and 'Edge' in remaining.columns:
                div_pick = remaining.sort_values('Edge', ascending=False).iloc[0]
                smart_picks.append({"Label": "⚖️ Diversifier", "Row": div_pick, "Color": "#FFD700", "Reason": "Portfolio Balance"})

            if smart_picks:
                cols = st.columns(len(smart_picks))
                for idx, pick in enumerate(smart_picks):
                    row = pick['Row']
                    sport_icon = utils.get_team_emoji(row.get('Sport', 'Soccer'))
                    rec_stake = bankroll * float(row.get('Stake', 0.01)) * (kelly_multiplier / 0.25)
                    with cols[idx]:
                        st.markdown(f"""
                        <div style="background-color: #1e2130; border: 1px solid {pick['Color']}; border-radius: 12px; padding: 15px; text-align: center; height: 100%;">
                            <div style="color: {pick['Color']}; font-weight: bold; text-transform: uppercase; font-size: 0.8em; margin-bottom: 5px;">{pick['Label']}</div>
                            <div style="font-size: 1.0em; font-weight: bold; margin-bottom: 5px;">{sport_icon} {row.get('Match', 'Unknown Match')}</div>
                            <div style="color: white; font-weight: bold; font-size: 1.1em; margin-bottom: 10px;">{row.get('Bet', 'Unknown Bet')}</div>
                            <div style="display: flex; justify-content: center; gap: 15px; margin-bottom: 10px;">
                                <div><div style="font-size: 0.7em; color: #888;">ODDS</div><div style="font-weight: bold; color: #fff;">{row.get('Odds', 0):.2f}</div></div>
                                <div><div style="font-size: 0.7em; color: #888;">EDGE</div><div style="font-weight: bold; color: {pick['Color']};">{row.get('Edge', 0):.1%}</div></div>
                            </div>
                            <div style="margin-top: 10px; font-size: 0.9em;">Stake: <span style="color: {pick['Color']}; font-weight: bold;">${rec_stake:.2f}</span></div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No smart picks available.")

        # --- MAIN FEED (Only Active/Future Bets) ---
        st.markdown("---")
        st.markdown("### 📋 Active & Upcoming Bets")
        
        current_time = datetime.now(timezone.utc)
        # Filter for active/future bets using the function from utils
        active_mask = df.apply(lambda x: utils.is_active_bet(x), axis=1)
        active_bets = df[active_mask].copy()
        
        if not active_bets.empty:
            st.metric("🎯 Active Bets", len(active_bets))
            
            for i, row in active_bets.iterrows():
                sport_icon = utils.get_team_emoji(row.get('Sport', 'Soccer'))
                match_time = row.get('Formatted_Date', 'Time TBD')
                risk_badge = utils.get_risk_badge(row)
                bookie = row.get('Info', 'Best Price')
                if pd.isna(bookie) or bookie == "": 
                    bookie = "Best Price"
                
                stake_pct = float(row.get('Stake', 0.01))
                cash_stake = bankroll * stake_pct * (kelly_multiplier / 0.25)
                
                with st.container():
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"""
                        <div class="bet-card">
                            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                <span style="color:#8b92a5; font-size:0.8em;">{match_time} • {row.get('League', 'League')}</span>
                                {risk_badge}
                            </div>
                            <div style="font-size:1.3em; font-weight:700; margin-bottom:5px;">
                                {sport_icon} {row.get('Match', 'Unknown Match')}
                            </div>
                            <div style="display:flex; align-items:center; gap:10px;">
                                <span style="color:#00C9FF; font-weight:600;">{row.get('Bet', 'Unknown Bet')}</span>
                                <span style="color:#555;">|</span>
                                <span style="color:#8b92a5; font-size:0.9em;">{bookie}</span>
                            </div>
                            <div style="margin-top:15px; display:flex; gap:20px;">
                                <div><div style="font-size:0.7em; color:#888;">EDGE</div><div style="font-weight:bold; color:#00e676;">{row.get('Edge', 0):.1%}</div></div>
                                <div><div style="font-size:0.7em; color:#888;">CONF</div><div style="font-weight:bold;">{row.get('Confidence', 0):.1%}</div></div>
                                <div><div style="font-size:0.7em; color:#888;">STAKE</div><div style="font-weight:bold;">${cash_stake:.2f}</div></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with c2:
                        st.markdown(f"""<div style="height:100%; display:flex; align-items:center; justify-content:center;"><div class="odds-box">{row.get('Odds', 0):.2f}</div></div>""", unsafe_allow_html=True)
                        
                        with st.expander("✏️ Edit"):
                            if 'key' in row:
                                key = row['key']
                                is_in_slip = any(b.get('key') == key for b in st.session_state.bet_slip)
                                user_stake = st.number_input(
                                    "Stake Amount ($)", 
                                    min_value=0.0, 
                                    value=float(cash_stake), 
                                    step=1.0, 
                                    key=f"stake_{key}"
                                )
                                
                                if st.checkbox("Add to Slip", value=is_in_slip, key=f"add_{key}"):
                                    if not is_in_slip:
                                        row_data = row.to_dict()
                                        row_data['User_Stake'] = user_stake
                                        st.session_state.bet_slip.append(row_data)
                                        st.rerun()
                                else:
                                    if is_in_slip:
                                        st.session_state.bet_slip = [b for b in st.session_state.bet_slip if b.get('key') != key]
                                        st.rerun()

        else:
            st.info("No active or upcoming bets found. Check back soon for new recommendations!")

        # --- STRICT PARLAY BUILDER ---
        st.markdown("---")
        st.subheader("🧩 Smart Parlay Builder")
        
        # 1. Safe Candidates: Odds < 2.20 (Favorites Only)
        if 'Odds' in df.columns and 'Bet Type' in df.columns:
            safe_candidates = df[(df['Odds'] > 1.1) & (df['Odds'] < 2.20) & (df['Bet Type'] != 'ARBITRAGE')]
        else:
            safe_candidates = pd.DataFrame()
        
        # 2. Value Candidates: Any Odds (for Moonshots)
        if 'Odds' in df.columns and 'Bet Type' in df.columns:
            value_candidates = df[(df['Odds'] > 1.1) & (df['Bet Type'] != 'ARBITRAGE')]
        else:
            value_candidates = df[df['Odds'] > 1.1] if 'Odds' in df.columns else pd.DataFrame()
        
        if len(value_candidates) >= 2 and 'Edge' in value_candidates.columns:
            tab1, tab2, tab3, tab4 = st.tabs(["🛡️ Safe (2-Leg)", "🚀 Value (3-Leg)", "🎰 Lotto (4-Leg)", "☄️ Hail Mary (5-Leg)"])

            def render_parlay_card(pool, num_legs, title):
                # Limit pool size for performance
                if pool.empty or 'Edge' not in pool.columns:
                    st.info(f"No suitable bets for {title}.")
                    return
                    
                pool = pool.sort_values('Edge', ascending=False).head(12).to_dict('records')
                
                if len(pool) < num_legs:
                    st.info(f"Not enough suitable bets found for {title}.")
                    return

                best_combo = None
                best_score = -1
                
                for combo in combinations(pool, num_legs):
                    if len(set([c.get('Match', '') for c in combo])) == num_legs:
                        score = (np.prod([1 + c.get('Edge', 0) for c in combo])) - 1
                        if score > best_score:
                            best_score = score
                            best_combo = combo
                
                if best_combo:
                    tot_odds = np.prod([c.get('Odds', 1) for c in best_combo])
                    tot_prob = np.prod([c.get('Confidence', 0) for c in best_combo])
                    kelly_stake_pct = (best_score / (tot_odds - 1)) * kelly_multiplier if tot_odds > 1 else 0
                    kelly_stake_cash = bankroll * kelly_stake_pct
                    
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown(f"#### {title}")
                        for leg in best_combo:
                            st.markdown(f"• **{leg.get('Bet', 'Unknown')}** @ {leg.get('Odds', 0):.2f} ({leg.get('Match', 'Unknown Match')})")
                    with c2:
                        st.metric("Total Odds", f"{tot_odds:.2f}")
                        st.metric("Win Prob", f"{tot_prob:.1%}")
                        st.metric("Rec. Stake", f"${kelly_stake_cash:.2f}")
                    
                    user_stake = st.number_input(f"Wager ($) - {title}", min_value=0.0, value=float(max(5.0, kelly_stake_cash)), step=5.0, key=f"parlay_{title}")
                    st.success(f"💰 Potential Payout: **${user_stake * tot_odds:.2f}**")
                else:
                    st.info("Could not build a valid parlay.")

            # Pass specific pools to specific tabs
            with tab1: render_parlay_card(safe_candidates, 2, "Bankroll Builder")
            with tab2: render_parlay_card(value_candidates, 3, "Value Stack")
            with tab3: render_parlay_card(value_candidates, 4, "Lotto Ticket")
            with tab4: render_parlay_card(value_candidates, 5, "Hail Mary")

        else:
            st.info("Not enough value bets to build a parlay.")

    elif df == "NO_BETS_FOUND":
        st.success("✅ System Online. Market Scanned. No Value Found.")
    else:
        st.error("Connection Error. Check GitHub configuration.")

def render_market_map():
    st.markdown('<p class="gradient-text">🗺️ Market Map</p>', unsafe_allow_html=True)
    df = utils.load_data(utils.LATEST_URL)
    if isinstance(df, pd.DataFrame) and not df.empty:
        if 'Bet Type' in df.columns:
            df = df[df['Bet Type'] != 'ARBITRAGE']
        if 'Odds' in df.columns:
            df = df[df['Odds'] > 0]
        if not df.empty and 'Odds' in df.columns and 'Confidence' in df.columns:
            df['Implied'] = 1 / df['Odds']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Fair Value', line=dict(color='#444', dash='dash')))
            fig.add_trace(go.Scatter(
                x=df['Implied'], 
                y=df['Confidence'], 
                mode='markers', 
                marker=dict(
                    size=df['Edge']*150 + 10 if 'Edge' in df.columns else 10, 
                    color=df['Edge'] if 'Edge' in df.columns else 'blue',
                    colorscale='Viridis', 
                    showscale=True,
                    colorbar=dict(title="Edge")
                ),
                text=df['Match'] + '<br>' + df['Bet'] if 'Match' in df.columns and 'Bet' in df.columns else None,
                hoverinfo='text'
            ))
            fig.update_layout(
                template="plotly_dark", 
                height=600, 
                xaxis_title="Implied Probability", 
                yaxis_title="Model Probability",
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
        else: 
            st.info("No valid bets for Market Map.")
    else: 
        st.info("No data loaded.")

def render_bet_tracker(bankroll):
    st.markdown('<p class="gradient-text">🎟️ Bet Slip</p>', unsafe_allow_html=True)
    
    if not hasattr(st.session_state, 'bet_slip') or not st.session_state.bet_slip:
        st.info("Your bet slip is empty. Add bets from the Command Center!")
        return
        
    bankroll = st.number_input("Your Bankroll ($)", value=float(bankroll), min_value=0.0, step=0.01, format="%.2f", key="tracker_bankroll")
    
    slip_df = pd.DataFrame(st.session_state.bet_slip)
    total_stake = 0
    potential_return = 0
    
    for i, bet in slip_df.iterrows():
        with st.container():
            st.markdown(f"""
            <div class="bet-card">
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="font-weight:bold; color:#00C9FF;">{bet.get('Sport', '')}</span>
                    <span style="color:#00e676; font-weight:bold;">{bet.get('Odds', 0):.2f}</span>
                </div>
                <div style="font-size:1.1em; font-weight:700; margin-bottom:5px;">
                    {bet.get('Match', 'Unknown Match')}
                </div>
                <div style="color:white; margin-bottom:10px;">
                    {bet.get('Bet', 'Unknown Bet')}
                </div>
                <div style="display:flex; justify-content:space-between; background:rgba(0,0,0,0.2); padding:8px; border-radius:6px;">
                    <span>Stake</span>
                    <span style="color:#00e676; font-weight:bold;">${bet.get('User_Stake', 0):.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            total_stake += bet.get('User_Stake', 0)
            potential_return += bet.get('User_Stake', 0) * bet.get('Odds', 1)
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stake", f"${total_stake:.2f}")
    with col2:
        st.metric("Potential Return", f"${potential_return:.2f}")
    with col3:
        st.metric("Net Profit", f"${potential_return - total_stake:.2f}")
    
    if st.button("❌ Clear All Bets", type="primary", use_container_width=True):
        st.session_state.bet_slip = []
        st.rerun()
    
    if st.button("✅ Place Bets", type="secondary", use_container_width=True):
        st.success("🎉 Bets placed successfully! (This would connect to your bookmaker in production)")
        # In production, this would send bets to actual bookmakers
        st.session_state.bet_slip = []
        st.rerun()

def render_history():
    st.markdown('<p class="gradient-text">📊 Betting Performance</p>', unsafe_allow_html=True)
    
    # Load and prepare data
    history_df = utils.load_data(utils.HISTORY_URL)
    
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        st.info("No betting history yet. Place your first bet to start tracking performance!")
        return
    
    # --- CRITICAL FIX: Call the function that exists in the updated utils.py ---
    # This function now handles the Dec 8, 2025 deadline and Google score lookup
    history_df = utils.settle_bets_with_google_scores(history_df)

    # Performance stats
    stats = utils.get_performance_stats(history_df)
    sport_stats = stats.get('sport_stats', {})
    
    # Performance Metrics
    if stats["total_bets"] > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 Win Rate", f"{stats['win_rate']:.1%}", 
                     delta=f"{stats['win_rate'] - 0.5:+.0%}" if stats['win_rate'] > 0.5 else None)
        with col2:
            st.metric("💰 ROI", f"{stats['roi']:.1%}", 
                     delta=f"{stats['roi']:+.1%}" if stats['roi'] > 0 else None)
        with col3:
            st.metric("🎫 Total Bets", stats["total_bets"])
        with col4:
            st.metric("⚡ Avg Edge", f"{history_df['Edge'].mean():.1%}" if 'Edge' in history_df.columns and not history_df.empty else "0.0%")
    
    # Sport Performance Chart
    if sport_stats:
        st.markdown("### 🏆 Sport Performance")
        sport_df = pd.DataFrame([
            {"Sport": sport, "Win Rate": win_rate, "Bets": len(history_df[history_df['Sport'] == sport])} 
            for sport, win_rate in sport_stats.items()
        ]).sort_values('Win Rate', ascending=False)
        
        # Clean sport performance visualization
        fig = px.bar(
            sport_df, 
            x='Sport', 
            y='Win Rate',
            color='Win Rate',
            color_continuous_scale=['#ff4d4d', '#00e676'],
            text=sport_df.apply(lambda x: f"{x['Win Rate']:.0%} ({x['Bets']})", axis=1),
            title="Win Rate by Sport (Min. 5 Bets)"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis_title="",
            yaxis_title="Win Rate",
            yaxis=dict(tickformat='.0%'),
            height=400
        )
        fig.update_traces(
            marker_line_color='#2d3748',
            marker_line_width=1.5,
            opacity=0.9
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Clean Results Table
    st.markdown("### 📋 Recent Bets")
    
    # Prepare display data - CRITICAL: Ensure Date_Obj exists before sorting
    display_df = history_df.copy()
    if 'Date_Obj' not in display_df.columns:
         st.error("Critical error: 'Date_Obj' column not found in history data after settlement.")
         st.dataframe(display_df, use_container_width=True, height=400) # Show raw data as fallback
         return
    
    display_df = display_df.sort_values('Date_Obj', ascending=False)
    
    # Format result column properly with score context
    display_df['Result_Display'] = display_df.apply(
        lambda x: utils.format_result_with_score(x['Result'], x.get('Score', '')), 
        axis=1
    )
    
    # Format profit column
    display_df['Profit_Display'] = display_df.apply(
        lambda x: f"+${x['Profit']:.2f}" if x['Result'] == 'Win' else 
                 f"-${abs(x['Profit']):.2f}" if x['Result'] in ['Loss', 'Auto-Settled', 'Push'] else 
                 "Pending",
        axis=1
    )
    
    # Show actual score prominently
    display_df['Match_Score'] = display_df.apply(
        lambda x: f"{x.get('Score', 'N/A')}" if x.get('Score') and x.get('Score') not in ['N/A', 'nan', 'NaN', ''] else "Result Pending",
        axis=1
    )
    
    # Select columns to display
    cols_to_show = ['Formatted_Date', 'Sport', 'Match', 'Match_Score', 'Bet', 'Odds', 'Edge', 'Stake', 'Result_Display', 'Profit_Display']
    cols_to_show = [c for c in cols_to_show if c in display_df.columns]
    
    # Create clean display table with scores
    st.dataframe(
        display_df[cols_to_show].rename(columns={
            'Formatted_Date': 'Date',
            'Edge': 'Edge %',
            'Stake': 'Stake ($)',
            'Result_Display': 'Result',
            'Profit_Display': 'Profit',
            'Match_Score': 'Final Score'
        }).style.format({
            'Odds': '{:.2f}',
            'Edge %': '{:.1%}',
            'Stake ($)': '${:.2f}'
        }).applymap(
            lambda x: 'background-color: rgba(0, 230, 118, 0.2); color: #69f0ae' if 'WIN' in str(x).upper() else 
                     'background-color: rgba(255, 82, 82, 0.2); color: #ff8a80' if 'LOSS' in str(x).upper() or 'AUTO-SETTLED' in str(x).upper() else 
                     'background-color: rgba(255, 204, 0, 0.2); color: #ffcc80' if 'PENDING' in str(x).upper() else 
                     'background-color: rgba(158, 158, 158, 0.2); color: #bdbdbd' if 'PUSH' in str(x).upper() else 
                     'background-color: rgba(0, 201, 255, 0.15); color: #00C9FF' if 'ACTIVE' in str(x).upper() else '',
            subset=['Result']
        ).applymap(
            lambda x: 'font-weight: bold; color: #00C9FF' if x not in ['Result Pending', 'N/A'] else 'color: #888',
            subset=['Final Score']
        ),
        use_container_width=True,
        height=450
    )
    
    # Download button
    st.download_button(
        "📥 Export Full History", 
        history_df.to_csv(index=False).encode('utf-8'), 
        "betting_history.csv", 
        "text/csv",
        use_container_width=True
    )

def render_about():
    st.markdown('<p class="gradient-text">ℹ️ About</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🚀 Betting Co-Pilot Pro
    
    **Version 73.1 (Google Score Lookup Edition)** - The AI-powered betting assistant that combines quantitative models with professional risk management.
    
    ### 🔍 Core Features
    
    - **AI Edge Detection**: Advanced machine learning models identify market inefficiencies
    - **Professional Bankroll Management**: Quarter-Kelly staking with volatility adjustments
    - **Multi-Sport Coverage**: Soccer, NFL, NBA, MLB with specialized models for each
    - **Strict Parlay Builder**: Algorithmically constructed parlays with risk controls
    - **Google Score Lookup**: All settled bets show final scores from Google search results for complete transparency
    
    ### ⚙️ Technical Stack
    
    - **Engine**: Python, Pandas, Scikit-learn, XGBoost
    - **Frontend**: Streamlit, Plotly
    - **Data Sources**: The Odds API, Enhanced Google Score Lookup
    - **Infrastructure**: Streamlit Cloud, GitHub Actions
    
    ### ⚠️ Responsible Gambling
    
    > **GAMBLING INVOLVES SUBSTANTIAL RISK. ONLY BET WHAT YOU CAN AFFORD TO LOSE.**
    
    This tool is for informational purposes only. Past performance is not indicative of future results. Always gamble responsibly.
    
    ### 🤝 Contact & Support
    
    - **GitHub**: [github.com/jd0913/betting-copilot-pro](https://github.com/jd0913/betting-copilot-pro)
    - **Issues**: Report bugs and feature requests via GitHub Issues
    - **Updates**: New features shipped weekly
    
    **© 2025 Betting Co-Pilot Pro** - Professional gambling analytics for serious bettors
    """)
    
    # System status
    with st.expander("🔧 System Status"):
        st.markdown(f"""
        **Data Sources**:
        - Latest Bets: {'✅ Available' if isinstance(utils.load_data(utils.LATEST_URL), pd.DataFrame) else '❌ Offline'}
        - Betting History: {'✅ Available' if isinstance(utils.load_data(utils.HISTORY_URL), pd.DataFrame) else '❌ Offline'}
        
        **Configuration**:
        - GitHub Repo: `{utils.GITHUB_USERNAME}/{utils.GITHUB_REPO}`
        - Streamlit Cloud: ✅ Connected
        - Auto-Settlement: ✅ Active (Google Score Lookup)
        - Score Tracking: ✅ Enabled
        """)
