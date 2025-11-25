# app.py
# This is the "Perfected Cockpit" for your Betting Co-Pilot.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="The Ultimate Co-Pilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# IMPORTANT: CONFIGURE YOUR GITHUB REPOSITORY DETAILS HERE
# ==============================================================================
GITHUB_USERNAME = "jd0913"
GITHUB_REPO = "betting-copilot-pro"
# ==============================================================================

DATA_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"

# ==============================================================================
# Data Loading Function
# ==============================================================================

@st.cache_data(ttl=600) # Cache the data for 10 minutes
def load_data():
    """Intelligently loads data from the GitHub repo."""
    try:
        import requests
        response = requests.get(DATA_URL)
        if response.status_code == 404:
            return "FILE_NOT_FOUND"
        
        df = pd.read_csv(DATA_URL)
        if df.empty:
            return "NO_BETS_FOUND"
        
        for col in ['Edge', 'Confidence', 'Odds']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Edge', 'Confidence', 'Odds'], inplace=True)
        return df
    except Exception:
        return "FILE_NOT_FOUND"

# ==============================================================================
# Main Application Pages
# ==============================================================================

def main_dashboard():
    """The main dashboard page of the application."""
    st.title("🚀 Betting Co-Pilot Dashboard")
    
    data_result = load_data()

    if isinstance(data_result, pd.DataFrame):
        value_df = data_result
        
        # --- Sidebar Filtering Controls ---
        st.sidebar.header("Dashboard Controls")
        
        available_sports = ["All Sports"] + list(value_df['Sport'].unique())
        selected_sport = st.sidebar.selectbox("Filter by Sport", available_sports)
        
        if selected_sport != "All Sports":
            value_df = value_df[value_df['Sport'] == selected_sport]

        if selected_sport == "Soccer" and 'League' in value_df.columns:
            available_leagues = ["All Leagues"] + list(value_df['League'].unique())
            selected_league = st.sidebar.selectbox("Filter by League", available_leagues)
            if selected_league != "All Leagues":
                value_df = value_df[value_df['League'] == selected_league]

        min_edge = st.sidebar.slider("Filter by Minimum Edge (%)", 0, 100, 5) / 100.0
        min_confidence = st.sidebar.slider("Filter by Minimum Confidence (%)", 0, 100, 30) / 100.0
        odds_range = st.sidebar.slider("Filter by Odds Range", 1.0, 20.0, (1.0, 20.0))
        
        filtered_df = value_df[
            (value_df['Edge'] >= min_edge) &
            (value_df['Confidence'] >= min_confidence) &
            (value_df['Odds'] >= odds_range[0]) &
            (value_df['Odds'] <= odds_range[1])
        ].copy()

        # --- KPI Row ---
        st.header("Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Value Bets Found", len(filtered_df))
        if not filtered_df.empty:
            col2.metric("Highest Edge Available", f"{filtered_df['Edge'].max():.2%}")
            col3.metric("Most Confident Bet", f"{filtered_df['Confidence'].max():.2%}")
            col4.metric("Average Odds", f"{filtered_df['Odds'].mean():.2f}")
        
        # --- Main Value Dashboard with Deep Dive ---
        st.header("📈 Value Dashboard")
        if not filtered_df.empty:
            # Add a unique key for checkboxes
            filtered_df['key'] = filtered_df['Match'] + "_" + filtered_df['Bet']
            
            st.dataframe(filtered_df[['Sport', 'League', 'Match', 'Bet', 'Odds', 'Edge', 'Confidence']].style.format({
                'Odds': '{:.2f}', 'Edge': '{:.2%}', 'Confidence': '{:.2%}'
            }).background_gradient(cmap='Greens', subset=['Edge']))

            st.subheader("Deep Dive Analysis")
            for i, row in filtered_df.iterrows():
                with st.expander(f"{row['Match']} - {row['Bet']} @ {row['Odds']:.2f}"):
                    st.write(f"**Edge:** {row['Edge']:.2%} | **Model Confidence:** {row['Confidence']:.2%}")
                    # *** NEW: Stake Explanation ***
                    if 'Stake (Kelly/4)' in row:
                         st.write(f"**Recommended Stake:** {float(row['Stake (Kelly/4)']):.2%} of Bankroll (Quarter-Kelly Strategy)")
                    
                    # Add to Bet Slip functionality
                    is_in_slip = any(bet['key'] == row['key'] for bet in st.session_state.bet_slip)
                    if st.checkbox("Add to my personal Bet Slip", value=is_in_slip, key=row['key']):
                        if not is_in_slip:
                            st.session_state.bet_slip.append(row.to_dict())
                            st.rerun()
                    else:
                        if is_in_slip:
                            st.session_state.bet_slip = [b for b in st.session_state.bet_slip if b['key'] != row['key']]
                            st.rerun()
        else:
            st.info("No bets match the current filter criteria.")

    elif data_result == "NO_BETS_FOUND":
        st.success("✅ Backend analysis complete. No value bets were found for the current fixtures.")
    elif data_result == "FILE_NOT_FOUND":
        st.error("Could not load the latest bets from GitHub.")

def market_map_page():
    """A page to visualize the entire market landscape."""
    st.title("🗺️ Market Map")
    st.markdown("Visualize your model's edge against the bookmaker's implied odds.")
    
    data_result = load_data()
    
    if isinstance(data_result, pd.DataFrame) and 'Odds' in data_result.columns:
        value_df = data_result[data_result['Sport'] == 'Soccer'].copy()
        if not value_df.empty:
            value_df['Implied_Prob'] = 1 / value_df['Odds']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No Value Line', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(
                x=value_df['Implied_Prob'], y=value_df['Confidence'], mode='markers',
                marker=dict(size=10, color=value_df['Edge'], colorscale='Viridis', showscale=True, colorbar=dict(title='Edge')),
                text=value_df['Match'] + "<br>" + value_df['Bet'], name='Value Bets'
            ))
            fig.update_layout(
                title='Model Confidence vs. Market Implied Probability (Soccer)',
                xaxis_title='Bookmaker Implied Probability (1 / Odds)', yaxis_title='Co-Pilot Model Confidence',
                xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("All profitable bets appear above the red 'No Value Line'. The brighter the point, the higher the edge.")
        else:
            st.info("No soccer bets available to display on the market map.")
    else:
        st.info("No data loaded to display the market map.")

def bet_tracker_page():
    """A page for the personal bet slip and tracker."""
    st.title("🎟️ Personal Bet Slip & Tracker")
    
    if 'bet_slip' not in st.session_state:
        st.session_state.bet_slip = []
        
    st.subheader("My Current Bet Slip")
    if st.session_state.bet_slip:
        slip_df = pd.DataFrame(st.session_state.bet_slip)
        st.dataframe(slip_df[['Match', 'Bet', 'Odds', 'Edge', 'Confidence']].style.format({'Odds': '{:.2f}', 'Edge': '{:.2%}', 'Confidence': '{:.2%}'}))
        
        if st.button("Clear Bet Slip"):
            st.session_state.bet_slip = []
            st.rerun()
    else:
        st.info("Your bet slip is empty. Select bets from the Main Dashboard to add them.")

def about_page():
    """A page explaining how the Co-Pilot works."""
    st.title("📖 About the Co-Pilot")
    st.markdown("""
    The Betting Co-Pilot is a sophisticated, multi-sport, multi-model system designed to find profitable betting opportunities (value) in sports markets.
    
    ### Architecture
    - **Backend Engine (GitHub Actions):** A fully automated script runs daily to download the latest data, train a portfolio of sport-specific models, analyze upcoming fixtures, and save the results.
    - **Frontend Cockpit (Streamlit):** This interactive dashboard reads the pre-processed results from the backend, ensuring a fast and responsive user experience.

    ### The Model Portfolio
    - **Soccer Brain:** A model based on the Elo rating system to gauge long-term team strength.
    - **NFL Brain:** A regression model that predicts the margin of victory.
    - **NBA Brain:** A model based on team Offensive and Defensive Efficiency Ratings.
    
    ### Key Concepts
    - **Edge:** The mathematical profit margin of a bet. We only recommend bets with a positive edge.
    - **Confidence:** Our model's calculated probability for an outcome to occur.
    """)

# ==============================================================================
# Sidebar Navigation & Main App Execution
# ==============================================================================
st.sidebar.title("Navigation")

# *** NEW: Manual Refresh Button ***
if st.sidebar.button("🔄 Refresh Data"):
    load_data.clear()
    st.rerun()

# *** FIX: Added Bet Tracker to the menu ***
PAGES = {
    "Main Dashboard": main_dashboard,
    "Market Map": market_map_page,
    "Bet Tracker": bet_tracker_page,
    "About the Co-Pilot": about_page
}
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

if 'bet_slip' not in st.session_state:
    st.session_state.bet_slip = []

page = PAGES[selection]
page()

st.sidebar.info(
    "This dashboard reads data updated daily by an automated backend. "
    "If the dashboard is empty, the system is in standby mode."
)
