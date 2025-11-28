# app.py
# The "True Final Monolith" Cockpit v44.0
# CONTAINS: God Mode UI + Accuracy Report + Parlay Builder + Bet Tracker + Market Map

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from itertools import combinations

# ==============================================================================
# CONFIGURATION & SETUP
# ==============================================================================
st.set_page_config(
    page_title="Betting Co-Pilot Pro", 
    page_icon="🚀", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- GITHUB CONFIG ---
# REPLACE WITH YOUR ACTUAL USERNAME
GITHUB_USERNAME = "jd0913"
GITHUB_REPO = "betting-copilot-pro"

LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"

# ==============================================================================
# 🎨 VISUAL STYLING ENGINE (CSS)
# ==============================================================================
def inject_custom_css(font_choice):
    """Injects CSS to style the app based on the selected font."""
    
    fonts = {
        "Modern (Roboto)": "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap",
        "Tech (JetBrains Mono)": "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap",
        "Clean (Inter)": "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap",
        "Futuristic (Orbitron)": "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap"
    }
    
    font_family = {
        "Modern (Roboto)": "'Roboto', sans-serif",
        "Tech (JetBrains Mono)": "'JetBrains Mono', monospace",
        "Clean (Inter)": "'Inter', sans-serif",
        "Futuristic (Orbitron)": "'Orbitron', sans-serif"
    }

    font_url = fonts[font_choice]
    family = font_family[font_choice]

    st.markdown(f"""
    <style>
        @import url('{font_url}');
        
        html, body, [class*="css"] {{
            font-family: {family};
        }}
        
        /* Gradient Header */
        .gradient-text {{
            background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
            font-size: 3em;
            padding-bottom: 10px;
        }}

        /* Card Styling */
        .bet-card {{
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            transition: transform 0.2s;
        }}
        .bet-card:hover {{
            transform: scale(1.01);
            border-color: #00C9FF;
        }}
        
        /* Badges */
        .badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 5px;
            display: inline-block;
        }}
        .badge-green {{ background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
        .badge-red {{ background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
        .badge-gray {{ background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }}
        .badge-blue {{ background-color: #cce5ff; color: #004085; border: 1px solid #b8daff; }}
        .badge-yellow {{ background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }}
        
        /* Metrics */
        div[data-testid="stMetricValue"] {{
            font-size: 1.5rem;
            color: #00C9FF;
        }}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# DATA LOGIC
# ==============================================================================
@st.cache_data(ttl=600)
def load_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 404: return "FILE_NOT_FOUND"
        df = pd.read_csv(url)
        if df.empty: return "NO_BETS_FOUND"
        
        # Numeric conversion
        for col in ['Edge', 'Confidence', 'Odds', 'Stake', 'Profit']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Date Formatting
        if 'Date' in df.columns:
            df['Date_Obj'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Formatted_Date'] = df['Date_Obj'].dt.strftime('%a, %b %d • %I:%M %p')
            df['Formatted_Date'] = df['Formatted_Date'].fillna('Time TBD')
        else:
            df['Formatted_Date'] = 'Time TBD'
            
        return df
    except: return "FILE_NOT_FOUND"

def get_team_emoji(sport):
    if sport == "Soccer": return "⚽"
    if sport == "NFL": return "🏈"
    if sport == "NBA": return "🏀"
    if sport == "MLB": return "⚾"
    return "🏅"

def get_risk_profile(row):
    edge = row.get('Edge', 0); odds = row.get('Odds', 0); conf = row.get('Confidence', 0)
    bet_type = row.get('Bet Type', '')
    
    if bet_type == 'ARBITRAGE': return '<span class="badge badge-blue">💎 RISK FREE PROFIT</span>'
    if odds > 3.5 and edge > 0.15: return '<span class="badge badge-red">⚡ RISING STAR (High Risk)</span>'
    if conf > 0.60 and edge > 0.05: return '<span class="badge badge-green">⭐ ANCHOR BET</span>'
    if row.get('Bet') == 'Draw': return '<span class="badge badge-gray">⚖️ VALUE DRAW</span>'
    return '<span class="badge badge-gray">✅ STANDARD VALUE</span>'

def format_result_badge(result):
    if result == 'Win': return '<span class="badge badge-green">WIN</span>'
    elif result == 'Loss': return '<span class="badge badge-red">LOSS</span>'
    elif result == 'Push': return '<span class="badge badge-gray">PUSH</span>'
    elif result == 'Pending': return '<span class="badge badge-yellow">⏳ PENDING</span>'
    else: return f'<span class="badge badge-gray">{result}</span>'

# ==============================================================================
# PAGE 1: DASHBOARD (COMMAND CENTER)
# ==============================================================================
def dashboard_page(bankroll, kelly_multiplier):
    st.markdown('<p class="gradient-text">🚀 Live Command Center</p>', unsafe_allow_html=True)
    
    df = load_data(LATEST_URL)
    
    if isinstance(df, pd.DataFrame):
        # Filters
        sports = ["All"] + list(df['Sport'].unique()) if 'Sport' in df.columns else ["All"]
        selected_sport = st.sidebar.selectbox("Filter Sport", sports)
        if selected_sport != "All": df = df[df['Sport'] == selected_sport]
        
        st.markdown("### 📋 Actionable Recommendations")
        
        if not df.empty:
            df['key'] = df['Match'] + "_" + df['Bet']

        for i, row in df.iterrows():
            sport_icon = get_team_emoji(row.get('Sport', 'Soccer'))
            match_time = row.get('Formatted_Date', 'Time TBD')
            risk_badge = get_risk_profile(row)
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
                        if st.checkbox("Add to Slip", value=is_in_slip, key=key):
                            if not is_in_slip:
                                row_data = row.to_dict()
                                row_data['User_Stake'] = cash_stake
                                st.session_state.bet_slip.append(row_data)
                                st.rerun()
                        else:
                            if is_in_slip:
                                st.session_state.bet_slip = [b for b in st.session_state.bet_slip if b['key'] != key]
                                st.rerun()
        
        # --- SMART PARLAY BUILDER (RESTORED) ---
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
            else:
                st.info("Could not build a valid parlay from current bets.")
        else:
            st.info("Not enough value bets to build a parlay.")

    elif df == "NO_BETS_FOUND":
        st.success("✅ System Online. Market Scanned. No Value Found.")
    else:
        st.error("Connection Error. Check GitHub configuration.")

# ==============================================================================
# PAGE 2: MARKET MAP
# ==============================================================================
def market_map_page():
    st.markdown('<p class="gradient-text">🗺️ Market Map</p>', unsafe_allow_html=True)
    df = load_data(LATEST_URL)
    if isinstance(df, pd.DataFrame):
        if 'Bet Type' in df.columns: df = df[df['Bet Type'] != 'ARBITRAGE']
        df['Implied'] = 1 / df['Odds']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Fair Value', line=dict(color='#444', dash='dash')))
        fig.add_trace(go.Scatter(
            x=df['Implied'], y=df['Confidence'], mode='markers',
            marker=dict(size=df['Edge']*150 + 10, color=df['Edge'], colorscale='Viridis', showscale=True),
            text=df['Match'] + '<br>' + df['Bet'], hoverinfo='text'
        ))
        fig.update_layout(template="plotly_dark", height=600, xaxis_title="Implied Prob", yaxis_title="Model Prob")
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No data loaded.")

# ==============================================================================
# PAGE 3: BET TRACKER
# ==============================================================================
def bet_tracker_page(bankroll):
    st.markdown('<p class="gradient-text">🎟️ Bet Slip</p>', unsafe_allow_html=True)
    if st.session_state.bet_slip:
        slip_df = pd.DataFrame(st.session_state.bet_slip)
        total_stake = 0
        potential_return = 0
        
        for i, bet in slip_df.iterrows():
            st.markdown(f"""
            <div class="bet-ticket" style="border-left: 4px solid #00C9FF;">
                <div style="display:flex; justify-content:space-between;">
                    <div style="font-weight:bold;">{bet['Match']}</div>
                    <div style="color:#00e676;">{bet['Odds']:.2f}</div>
                </div>
                <div style="font-size:0.9em; color:#ccc;">{bet['Bet']}</div>
                <div style="margin-top:10px; font-size:0.8em; color:#888;">
                    Stake: <span style="color:white;">${bet.get('User_Stake', 0):.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            total_stake += bet.get('User_Stake', 0)
            potential_return += bet.get('User_Stake', 0) * bet['Odds']
            
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Total Stake", f"${total_stake:.2f}")
        c2.metric("Potential Return", f"${potential_return:.2f}")
        if st.button("Clear Slip"): st.session_state.bet_slip = []; st.rerun()
    else: st.info("Your bet slip is empty.")

# ==============================================================================
# PAGE 4: HISTORY (WITH ACCURACY REPORT)
# ==============================================================================
def history_page():
    st.markdown('<p class="gradient-text">📜 Performance Archive</p>', unsafe_allow_html=True)
    df = load_data(HISTORY_URL)
    
    if isinstance(df, pd.DataFrame):
        if 'Result' not in df.columns:
            st.info("No results settled yet.")
            st.dataframe(df)
            return

        # --- ACCURACY REPORT ---
        settled = df[df['Result'].isin(['Win', 'Loss'])]
        if not settled.empty:
            total_profit = settled['Profit'].sum()
            wins = len(settled[settled['Result'] == 'Win'])
            total_settled = len(settled)
            accuracy = wins / total_settled
            
            st.markdown("### 🏆 Accuracy Report")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Profit", f"{total_profit:.2f}u")
            c2.metric("Model Accuracy", f"{accuracy:.1%}", help="Win Rate")
            c3.metric("Total Bets Settled", total_settled)
            
            # Breakdown by Sport
            if 'Sport' in settled.columns:
                st.markdown("#### Accuracy by Sport")
                sport_stats = settled.groupby('Sport').apply(
                    lambda x: pd.Series({
                        'Accuracy': len(x[x['Result']=='Win']) / len(x),
                        'Profit': x['Profit'].sum(),
                        'Count': len(x)
                    })
                )
                st.dataframe(sport_stats.style.format({'Accuracy': '{:.1%}', 'Profit': '{:.2f}', 'Count': '{:.0f}'}))
            st.divider()

        # HTML Table with Badges
        st.write("### Full History")
        display_df = df.copy()
        display_df['Result'] = display_df['Result'].fillna('Pending')
        display_df['Status'] = display_df['Result'].apply(format_result_badge)
        
        cols = ['Formatted_Date', 'Sport', 'Match', 'Bet', 'Odds', 'Status', 'Profit']
        display_df = display_df.rename(columns={'Formatted_Date': 'Date'})
        cols = [c for c in cols if c in display_df.columns]
        
        st.write(display_df[cols].to_html(escape=False, index=False), unsafe_allow_html=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "history.csv", "text/csv")
        
    else: st.info("No history found.")

def about_page():
    st.title("📖 About")
    st.markdown("### Betting Co-Pilot Pro v44.0")
    st.markdown("Automated quantitative analysis engine running on GitHub Actions.")

# ==============================================================================
# NAVIGATION & GLOBAL SETTINGS
# ==============================================================================
if 'bet_slip' not in st.session_state: st.session_state.bet_slip = []

st.sidebar.title("Navigation")

# --- GLOBAL FONT SELECTOR ---
st.sidebar.header("🎨 Appearance")
font_choice = st.sidebar.selectbox("Font Style", ["Modern (Roboto)", "Tech (JetBrains Mono)", "Clean (Inter)", "Futuristic (Orbitron)"])
inject_custom_css(font_choice)

# --- GLOBAL BANKROLL ---
st.sidebar.header("💰 Bankroll")
bankroll = st.sidebar.number_input("Bankroll ($)", value=1000, step=100)
kelly_multiplier = st.sidebar.slider("Kelly Multiplier", 0.1, 1.0, 0.25, help="Recommended: 0.25")

st.sidebar.markdown("---")

page = st.sidebar.radio("Go To", ["Command Center", "Market Map", "Bet Tracker", "History", "About"])
if st.sidebar.button("🔄 Refresh Data"): st.cache_data.clear(); st.rerun()

if page == "Command Center": dashboard_page(bankroll, kelly_multiplier)
elif page == "Market Map": market_map_page()
elif page == "Bet Tracker": bet_tracker_page(bankroll)
elif page == "History": history_page()
elif page == "About": about_page()

st.sidebar.markdown("---")
st.sidebar.caption(f"Connected to: `{GITHUB_USERNAME}/{GITHUB_REPO}`")
