# app.py
# The "US Pro God Mode" Cockpit v32.1
# Combines Live US Odds/Arb Data with the Premium Visual Interface & Bet Tracking

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# ==============================================================================
# CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Betting Co-Pilot US Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# IMPORTANT: CONFIGURE YOUR GITHUB REPOSITORY DETAILS HERE
# ------------------------------------------------------------------------------
GITHUB_USERNAME = "jd0913"
GITHUB_REPO = "betting-copilot-pro"
# ------------------------------------------------------------------------------

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

def get_risk_profile(row):
    """Re-calculates the risk profile on the fly."""
    edge = row.get('Edge', 0)
    odds = row.get('Odds', 0)
    conf = row.get('Confidence', 0)
    
    if row.get('Bet Type') == 'ARBITRAGE': return "💎 RISK FREE PROFIT"
    if odds > 3.5 and edge > 0.15: return "⚡ Rising Star (High Risk/Reward)"
    if conf > 0.60 and edge > 0.05: return "⭐ High Confidence Anchor"
    if row.get('Bet') == 'Draw': return "⚖️ Value Draw"
    return "✅ Standard Value"

# ==============================================================================
# PAGES
# ==============================================================================

def dashboard_page():
    st.title("🚀 Live Command Center (US Pro)")
    
    df = load_data(LATEST_URL)
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("💰 Bankroll Strategy")
    bankroll = st.sidebar.number_input("Bankroll Size ($)", value=1000, step=100)
    kelly_multiplier = st.sidebar.slider("Kelly Multiplier", 0.1, 1.0, 0.25, help="Recommended: 0.25 (Quarter Kelly)")
    
    st.sidebar.header("🔍 Advanced Filters")
    
    if isinstance(df, pd.DataFrame):
        # --- 1. ARBITRAGE SECTION (The "Free Money" Alert) ---
        if 'Bet Type' in df.columns:
            arbs = df[df['Bet Type'] == 'ARBITRAGE']
            if not arbs.empty:
                st.error(f"🚨 {len(arbs)} ARBITRAGE OPPORTUNITIES DETECTED! GUARANTEED PROFIT.")
                for i, row in arbs.iterrows():
                    with st.container():
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"### 💎 {row['Match']}")
                            st.code(row['Info']) # Shows exactly which books to bet
                        with c2:
                            st.metric("Guaranteed Profit", f"{row['Edge']:.2%}")
                    st.divider()

        # --- 2. STANDARD VALUE BETS ---
        # Filter out arbs for the main list
        if 'Bet Type' in df.columns:
            value_df = df[df['Bet Type'] != 'ARBITRAGE'].copy()
        else:
            value_df = df.copy()

        # Sport Filter
        sports = ["All"] + list(value_df['Sport'].unique()) if 'Sport' in value_df.columns else ["All"]
        selected_sport = st.sidebar.selectbox("Filter Sport", sports)
        if selected_sport != "All":
            value_df = value_df[value_df['Sport'] == selected_sport]

        # Metric Sliders
        min_edge = st.sidebar.slider("Min Edge (%)", 0, 50, 5) / 100.0
        min_conf = st.sidebar.slider("Min Confidence (%)", 0, 100, 30) / 100.0
        
        # Apply Filters
        filtered_df = value_df[
            (value_df['Edge'] >= min_edge) & 
            (value_df['Confidence'] >= min_conf)
        ].copy()

        # --- KPI ROW ---
        total_bets = len(filtered_df)
        if total_bets > 0:
            avg_edge = filtered_df['Edge'].mean()
            # Calculate dynamic stake based on user bankroll input
            filtered_df['User_Stake_Cash'] = bankroll * filtered_df['Stake'] * (kelly_multiplier / 0.25)
            total_risk = filtered_df['User_Stake_Cash'].sum()
            proj_profit = (filtered_df['User_Stake_Cash'] * (filtered_df['Odds'] - 1)).sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Opportunities", total_bets)
            c2.metric("Avg Edge", f"{avg_edge:.2%}")
            c3.metric("Total Risk", f"${total_risk:.2f}")
            c4.metric("Proj. Profit", f"${proj_profit:.2f}", delta="Daily Potential")
        else:
            st.info("No bets match your current filters.")

        # --- THE "GOD MODE" CARD VIEW ---
        st.subheader(f"📋 Actionable Recommendations")
        
        # Create unique keys for bet slip
        if not filtered_df.empty:
            filtered_df['key'] = filtered_df['Match'] + "_" + filtered_df['Bet']

        for i, row in filtered_df.iterrows():
            profile = get_risk_profile(row)
            sport_icon = get_team_emoji(row.get('Sport', 'Soccer'))
            
            # Handle missing info gracefully
            bookie_info = row.get('Info', 'Check Books')
            if pd.isna(bookie_info): bookie_info = "Check Books"
            
            # Card Container
            with st.container():
                # Top Row: Match Info & Key Stats
                c1, c2, c3, c4, c5 = st.columns([2.5, 1, 1, 1, 1])
                
                with c1:
                    st.markdown(f"### {sport_icon} {row['Match']}")
                    st.caption(f"{row.get('League', 'Unknown')} • **{row['Bet']}**")
                    st.markdown(f"**{bookie_info}**") 
                
                with c2:
                    st.metric("Odds", f"{row['Odds']:.2f}")
                
                with c3:
                    st.metric("Edge", f"{row['Edge']:.2%}")
                    st.progress(min(float(row['Edge']), 1.0))
                
                with c4:
                    st.metric("Confidence", f"{row['Confidence']:.2%}")
                
                with c5:
                    rec_stake_pct = row['Stake'] * (kelly_multiplier / 0.25)
                    cash_val = bankroll * rec_stake_pct
                    st.metric("Bet Size", f"${cash_val:.2f}", delta=f"{rec_stake_pct:.2%}")

                # Deep Dive Expander & Bet Slip
                with st.expander(f"🔍 Deep Dive & Bet Slip: {row['Match']}"):
                    dd1, dd2 = st.columns(2)
                    with dd1:
                        st.markdown("**Analysis Breakdown:**")
                        st.write(f"- **Implied Probability:** {(1/row['Odds']):.2%}")
                        st.write(f"- **Model Probability:** {row['Confidence']:.2%}")
                        st.write(f"- **Risk Profile:** {profile}")
                    with dd2:
                        if 'News Alert' in row and pd.notna(row['News Alert']):
                            st.error(f"**News Alert:** {row['News Alert']}")
                        else:
                            st.success("No critical injury news detected.")
                        
                        # --- BET SLIP LOGIC ---
                        is_in_slip = any(bet['key'] == row['key'] for bet in st.session_state.bet_slip)
                        if st.checkbox("Add to my personal Bet Slip", value=is_in_slip, key=row['key']):
                            if not is_in_slip:
                                st.session_state.bet_slip.append(row.to_dict())
                                st.rerun()
                        else:
                            if is_in_slip:
                                st.session_state.bet_slip = [b for b in st.session_state.bet_slip if b['key'] != row['key']]
                                st.rerun()
                
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
        # Filter out arbs for the map
        if 'Bet Type' in df.columns:
            df = df[df['Bet Type'] != 'ARBITRAGE']
            
        df['Implied'] = 1 / df['Odds']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Fair Value', line=dict(color='gray', dash='dash')))
        
        fig.add_trace(go.Scatter(
            x=df['Implied'], 
            y=df['Confidence'],
            mode='markers',
            marker=dict(
                size=df['Edge']*100 + 10, 
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

def bet_tracker_page():
    st.title("🎟️ Personal Bet Slip & Tracker")
    
    # Add Bankroll Input here too for accurate calculation
    bankroll = st.number_input("Your Bankroll ($)", value=1000, step=100, key="tracker_bankroll")
    
    if 'bet_slip' not in st.session_state:
        st.session_state.bet_slip = []
        
    st.subheader("My Current Bet Slip")
    if st.session_state.bet_slip:
        slip_df = pd.DataFrame(st.session_state.bet_slip)
        
        # Display key columns
        display_cols = ['Match', 'Bet', 'Odds', 'Edge', 'Confidence']
        if 'Info' in slip_df.columns: display_cols.append('Info')
            
        st.dataframe(slip_df[display_cols].style.format({
            'Odds': '{:.2f}', 'Edge': '{:.2%}', 'Confidence': '{:.2%}'
        }))
        
        # Calculate Total Potential Return based on actual bankroll
        total_stake = 0
        potential_profit = 0
        
        for bet in st.session_state.bet_slip:
            # Default to 1% stake if not specified, or use the model's recommendation
            stake_pct = bet.get('Stake', 0.01) 
            cash_stake = bankroll * stake_pct
            total_stake += cash_stake
            potential_profit += cash_stake * (bet['Odds'] - 1)
            
        c1, c2 = st.columns(2)
        c1.metric("Total Stake Required", f"${total_stake:.2f}")
        c2.metric("Total Potential Profit", f"${potential_profit:.2f}")
        
        if st.button("Clear Bet Slip"):
            st.session_state.bet_slip = []
            st.rerun()
    else:
        st.info("Your bet slip is empty. Select bets from the Command Center to add them.")

def history_page():
    st.title("📜 Performance Archive")
    df = load_data(HISTORY_URL)
    if isinstance(df, pd.DataFrame):
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full History", csv, "history.csv", "text/csv")
    else:
        st.info("No history archived yet.")

def about_page():
    st.title("📖 About the Co-Pilot")
    st.markdown("""
    **v32.1 US Pro God Mode**
    
    This system runs on a **Portfolio Architecture**:
    1.  **Soccer Brain:** Elo Rating + Poisson Distribution + Shot Dominance + Genetic Evolution.
    2.  **NFL Brain:** Yards Per Play + Turnover Differential Regression.
    3.  **NBA Brain:** Four Factors Efficiency Model.
    
    **Key Metrics:**
    *   **Edge:** The mathematical advantage over the bookmaker.
    *   **Confidence:** The model's raw probability of winning.
    *   **Kelly Stake:** The mathematically optimal bet size to maximize growth and minimize ruin.
    """)

# ==============================================================================
# NAVIGATION
# ==============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go To", ["Command Center", "Market Map", "Bet Tracker", "History", "About"])

if st.sidebar.button("🔄 Force Refresh"):
    st.cache_data.clear()
    st.rerun()

# Initialize Session State for Bet Slip
if 'bet_slip' not in st.session_state:
    st.session_state.bet_slip = []

if page == "Command Center": dashboard_page()
elif page == "Market Map": market_map_page()
elif page == "Bet Tracker": bet_tracker_page()
elif page == "History": history_page()
elif page == "About": about_page()

st.sidebar.markdown("---")
st.sidebar.caption(f"Connected to: `{GITHUB_USERNAME}/{GITHUB_REPO}`")
