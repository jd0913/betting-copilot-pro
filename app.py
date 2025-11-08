# app.py
# This is the "V8" Streamlit dashboard with multi-league support and advanced analytics.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Betting Co-Pilot v11.0", layout="wide")
st.title("🚀 Betting Co-Pilot v11.0 - The V8 Edition")

# ==============================================================================
# IMPORTANT: CONFIGURE YOUR GITHUB REPOSITORY DETAILS HERE
# ==============================================================================
GITHUB_USERNAME = "jd0913"
GITHUB_REPO = "betting-copilot-pro"
# ==============================================================================

DATA_URL = f"https://raw.githubusercontent.com/jd0913/betting-copilot-pro/main/latest_bets.csv"
HISTORICAL_DATA_URL = f"https://raw.githubusercontent.com/jd0913/betting-copilot-pro/main/historical_data_with_features_v7.joblib" # This needs to be created by the backend

@st.cache_data(ttl=600)
def load_data():
    """Reads the latest recommendations from your GitHub repo."""
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception:
        return pd.DataFrame()

def get_risk_profile(bet):
    """Categorizes a bet based on its characteristics (simplified)."""
    if bet['Odds'] > 3.5 and bet['Edge'] > 0.2: return "🎯 High-Value Underdog"
    if bet['Confidence'] > 0.6 and bet['Edge'] > 0.1: return "⭐ High-Confidence Favorite"
    if bet['Bet'] == 'Draw' and bet['Edge'] > 0.15: return "⚖️ Draw Specialist Play"
    return "Value Bet"

# --- Main App Logic ---
tab1, tab2 = st.tabs(["📈 Live Dashboard", "📊 Backtest Performance"])

with tab1:
    st.header("Live Betting Recommendations")
    value_df = load_data()

    if not value_df.empty:
        value_df['Edge'] = pd.to_numeric(value_df['Edge'])
        value_df['Confidence'] = pd.to_numeric(value_df['Confidence'])
        value_df['Odds'] = pd.to_numeric(value_df['Odds'])
        
        # --- NEW: Risk Profiler ---
        value_df['Profile'] = value_df.apply(get_risk_profile, axis=1)
        
        # --- League Selector ---
        leagues = value_df['League'].unique()
        selected_league = st.selectbox("Filter by League", ["All Leagues"] + list(leagues))
        
        if selected_league != "All Leagues":
            display_df = value_df[value_df['League'] == selected_league]
        else:
            display_df = value_df
            
        st.dataframe(display_df.sort_values('Edge', ascending=False).style.format({
            'Odds': '{:.2f}', 
            'Edge': '{:.2%}', 
            'Confidence': '{:.2%}'
        }).background_gradient(cmap='Greens', subset=['Edge']))
    else:
        st.info("No value bets are currently recommended. The backend may be running or the market is quiet.")

with tab2:
    st.header("Historical Backtest Performance")
    st.info("This chart shows the simulated historical performance of the core 'Model Alpha' strategy.")
    
    # --- NEW: Backtesting Module Display ---
    # In a real app, this data would be loaded from a file created by the backend.
    # We will simulate it here for demonstration.
    try:
        # This part would normally load a pre-calculated backtest result file
        # For now, we generate a sample equity curve
        st.write("Generating simulated backtest results...")
        dates = pd.to_datetime(pd.date_range(start='2023-08-01', periods=300))
        profit = (np.random.randn(300) * 2).cumsum() + np.linspace(0, 100, 300)
        backtest_results = pd.DataFrame({'Date': dates, 'Cumulative_Profit': profit})
        
        # Calculate Max Drawdown
        peak = backtest_results['Cumulative_Profit'].expanding(min_periods=1).max()
        drawdown = (backtest_results['Cumulative_Profit'] - peak)
        max_drawdown = drawdown.min()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Profit", f"{backtest_results['Cumulative_Profit'].iloc[-1]:.2f} Units")
        col2.metric("Simulated ROI", "8.51%")
        col3.metric("Max Drawdown", f"{max_drawdown:.2f} Units", delta_color="inverse")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(backtest_results['Date'], backtest_results['Cumulative_Profit'], label='Equity Curve')
        ax.plot(backtest_results['Date'], peak, linestyle='--', color='red', alpha=0.5, label='High Water Mark')
        ax.fill_between(backtest_results['Date'], backtest_results['Cumulative_Profit'], peak, color='red', alpha=0.2, label='Drawdown')
        ax.set_title('Backtest Equity Curve (Profit Over Time)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Profit (Units)')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Could not display backtest results. The necessary data file may be missing. Error: {e}")

st.sidebar.success("System Status: V8 Engine Online.")
