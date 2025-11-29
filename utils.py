# utils.py
# Shared functions for data loading, styling, and logic.
# v49.0 - Date Fix

import streamlit as st
import pandas as pd
import requests

# --- CONFIGURATION ---
# REPLACE WITH YOUR USERNAME
GITHUB_USERNAME = "jd0913"
GITHUB_REPO = "betting-copilot-pro"

LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"

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
            # Force conversion to datetime, handle errors
            df['Date_Obj'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
            
            # Format: Thu, Jan 1, 2026 at 03:00 PM
            # We use a lambda to handle NaT (Not a Time) values gracefully
            df['Formatted_Date'] = df['Date_Obj'].apply(
                lambda x: x.strftime('%a, %b %d • %I:%M %p') if pd.notnull(x) else 'Time TBD'
            )
        else:
            df['Formatted_Date'] = 'Time TBD'
            
        return df
    except: return "FILE_NOT_FOUND"

def inject_custom_css(font_choice="Clean (Inter)"):
    """Injects the 'Vegas Dark' design system."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .gradient-text {
            background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-weight: 800; font-size: 3em; padding-bottom: 10px;
        }
        .bet-card {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px; padding: 16px; margin-bottom: 12px; transition: transform 0.2s;
        }
        .bet-card:hover { transform: translateY(-2px); border-color: #00C9FF; }
        .odds-box {
            background-color: #262a3b; color: #00e676; font-weight: 700; font-size: 1.1em;
            padding: 8px 16px; border-radius: 8px; text-align: center; border: 1px solid #00e676;
        }
        .badge {
            padding: 4px 8px; border-radius: 6px; font-size: 0.7em; font-weight: 800;
            text-transform: uppercase; display: inline-block; margin-right: 5px;
        }
        .badge-arb { background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%); color: #000; }
        .badge-high { background-color: #ff4b4b; color: white; }
        .badge-safe { background-color: #00e676; color: #000; }
        .badge-std { background-color: #31333F; color: #ccc; border: 1px solid #555; }
        .res-win { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
        .res-loss { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
        .res-push { background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
        .res-pending { color: #ffc107; font-weight: bold; font-style: italic; }
        div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #00C9FF; }
    </style>
    """, unsafe_allow_html=True)

def get_team_emoji(sport):
    if sport == "Soccer": return "⚽"
    if sport == "NFL": return "🏈"
    if sport == "NBA": return "🏀"
    if sport == "MLB": return "⚾"
    return "🏅"

def get_risk_badge(row):
    edge = row.get('Edge', 0); odds = row.get('Odds', 0); conf = row.get('Confidence', 0)
    bet_type = row.get('Bet Type', '')
    if bet_type == 'ARBITRAGE': return '<span class="badge badge-arb">💎 ARBITRAGE</span>'
    if odds > 3.5 and edge > 0.15: return '<span class="badge badge-high">⚡ HIGH RISK</span>'
    if conf > 0.60 and edge > 0.05: return '<span class="badge badge-safe">⭐ ANCHOR</span>'
    return '<span class="badge badge-std">VALUE</span>'

def format_result_badge(result):
    if result == 'Win': return '<span class="res-win">WIN</span>'
    elif result == 'Loss': return '<span class="res-loss">LOSS</span>'
    elif result == 'Push': return '<span class="res-push">PUSH</span>'
    elif result == 'Pending': return '<span class="res-pending">⏳ PENDING</span>'
    else: return f'<span>{result}</span>'
