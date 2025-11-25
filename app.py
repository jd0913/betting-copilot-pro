# app.py
# The "God Mode" Cockpit v21.0
# Features: Bankroll Sim, Visual Polish, Edge Meters, Mobile Optimization

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# ==============================================================================
# CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Betting Co-Pilot Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# REPLACE WITH YOUR DETAILS
GITHUB_USERNAME = "jd0913"
GITHUB_REPO = "betting-copilot-pro"

LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

@st.cache_data(ttl=600)
def load_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 404: return "FILE_NOT_FOUND"
        df = pd.read_csv(url)
        if df.empty: return "NO_BETS_FOUND"
        
        # Numeric conversion
        cols = ['Edge', 'Confidence', 'Odds', 'Stake']
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except: return "FILE_NOT_FOUND"

def get_team_emoji(sport):
    if sport == "Soccer": return "⚽"
    if sport == "NFL": return "🏈"
    if sport == "NBA": return "🏀"
    return "🏅"

def color_edge(val):
    color = 'red'
    if val > 0.05: color = 'orange'
    if val > 0.10: color = 'green'
    if val > 0.20: color = 'blue' # Super value
    return f'color: {color}'

# ==============================================================================
# PAGES
# ==============================================================================

def dashboard_page():
    st.title("🚀 Live Command Center")
    
    # --- 1. Bankroll Management Sidebar ---
    st.sidebar.header("💰 Bankroll Manager")
    bankroll = st.sidebar.number_input("Current Bankroll ($)", value=1000, step=100)
    kelly_multiplier = st.sidebar.slider("Risk Appetite (Kelly Multiplier)", 0.1, 1.0, 0.25, help="1.0 is Full Kelly (High Risk). 0.25 is Quarter Kelly (Professional Standard).")
    
    df = load_data(LATEST_URL)
    
    if isinstance(df, pd.DataFrame):
        # --- 2. Global Filters ---
        sports = ["All"] + list(df['Sport'].unique()) if 'Sport' in df.columns else ["All"]
        selected_sport = st.sidebar.selectbox("Filter Sport", sports)
        if selected_sport != "All":
            df = df[df['Sport'] == selected_sport]

        # --- 3. KPI Row ---
        total_bets = len(df)
        avg_edge = df['Edge'].mean()
        # Calculate total potential profit for the day
        # Stake = Bankroll * Stake_Pct * Kelly_Mult
        # Profit = Stake * (Odds - 1)
        df['Calc_Stake_Cash'] = bankroll * df['Stake'] * (kelly_multiplier / 0.25) # Adjusting because backend assumes 0.25
        df['Potential_Profit'] = df['Calc_Stake_Cash'] * (df['Odds'] - 1)
        total_potential_profit = df['Potential_Profit'].sum()
        total_risk = df['Calc_Stake_Cash'].sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Active Opportunities", total_bets)
        col2.metric("Average Edge", f"{avg_edge:.2%}")
        col3.metric("Total Capital at Risk", f"${total_risk:.2f}")
        col4.metric("Proj. Daily Profit", f"${total_potential_profit:.2f}", delta="If all win")

        # --- 4. The "God Mode" Table ---
        st.subheader(f"📋 Actionable Recommendations ({selected_sport})")
        
        for i, row in df.iterrows():
            with st.container():
                # Create a card-like layout
                c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
                
                sport_icon = get_team_emoji(row.get('Sport', 'Soccer'))
                
                with c1:
                    st.markdown(f"**{sport_icon} {row['Match']}**")
                    st.caption(f"{row.get('League', 'Unknown League')} • {row['Bet']}")
                    if 'News Alert' in row and pd.notna(row['News Alert']):
                        st.warning(row['News Alert'])
                
                with c2:
                    st.metric("Odds", f"{row['Odds']:.2f}")
                
                with c3:
                    st.metric("Edge", f"{row['Edge']:.2%}")
                    st.progress(min(float(row['Edge']), 1.0)) # Visual Edge Meter
                
                with c4:
                    st.metric("Confidence", f"{row['Confidence']:.2%}")
                
                with c5:
                    # Dynamic Stake Calculation Display
                    rec_stake = row['Stake'] * (kelly_multiplier / 0.25)
                    cash_stake = bankroll * rec_stake
                    st.metric("Bet Size", f"${cash_stake:.2f}", delta=f"{rec_stake:.2%}")
                
                st.divider()

    elif df == "NO_BETS_FOUND":
        st.success("✅ System Online. Market Scanned. No Value Found.")
        st.info("The machine is disciplined. It only recommends bets when the math is in your favor.")
    else:
        st.error("Connection Error. Check GitHub configuration.")

def market_map_page():
    st.title("🗺️ Market Map Visualization")
    df = load_data(LATEST_URL)
    if isinstance(df, pd.DataFrame):
        df['Implied'] = 1 / df['Odds']
        
        fig = go.Figure()
        # The "No Value" Line
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Fair Value', line=dict(color='gray', dash='dash')))
        
        # The Bets
        fig.add_trace(go.Scatter(
            x=df['Implied'], 
            y=df['Confidence'],
            mode='markers',
            marker=dict(
                size=df['Edge']*100 + 10, # Size bubbles by Edge
                color=df['Edge'], 
                colorscale='RdYlGn', 
                showscale=True,
                colorbar=dict(title="Edge")
            ),
            text=df['Match'] + '<br>' + df['Bet'],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title="Market Inefficiency Map (Bubbles = Edge Size)",
            xaxis_title="Bookmaker Implied Probability",
            yaxis_title="Co-Pilot Calculated Probability",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bets ABOVE the dotted line are profitable. Larger/Greener bubbles = Better Bets.")

def history_page():
    st.title("📜 Performance Archive")
    df = load_data(HISTORY_URL)
    if isinstance(df, pd.DataFrame):
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full History", csv, "history.csv", "text/csv")
    else:
        st.info("No history archived yet.")

# ==============================================================================
# NAVIGATION
# ==============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go To", ["Command Center", "Market Map", "History"])

if st.sidebar.button("🔄 Force Refresh"):
    st.cache_data.clear()
    st.rerun()

if page == "Command Center": dashboard_page()
elif page == "Market Map": market_map_page()
elif page == "History": history_page()

st.sidebar.markdown("---")
st.sidebar.caption(f"Connected to: `{GITHUB_USERNAME}/{GITHUB_REPO}`")
