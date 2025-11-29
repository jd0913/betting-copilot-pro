# utils.py
# Shared functions for data loading, styling, performance, and UI helpers
# v68.0 — Fully fixed, robust, with live performance metrics

import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================
GITHUB_USERNAME = "jd0913"
GITHUB_REPO = "betting-copilot-pro"

LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"

# ==============================================================================
# 1. DATA LOADING (Robust + Caching)
# ==============================================================================
@st.cache_data(ttl=300, show_spinner=False)  # 5-minute cache
def load_data(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            return "FILE_NOT_FOUND"
        if response.status_code != 200:
            return "CONNECTION_ERROR"
        
        df = pd.read_csv(url)
        
        if df.empty:
            return "NO_BETS_FOUND"

        # Safe numeric conversion
        numeric_cols = ['Edge', 'Confidence', 'Odds', 'Stake', 'Profit']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Date formatting
        if 'Date' in df.columns:
            df['Date_Obj'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
            df['Formatted_Date'] = df['Date_Obj'].dt.strftime('%a, %b %d • %I:%M %p')
            df['Formatted_Date'] = df['Formatted_Date'].fillna('Time TBD')
        else:
            df['Formatted_Date'] = 'Time TBD'

        return df

    except Exception as e:
        st.error(f"Data load failed: {str(e)}")
        return "CONNECTION_ERROR"

# Force-clear cache (used by refresh button in app.py)
def clear_cache():
    load_data.clear()

# ==============================================================================
# 2. PERFORMANCE METRICS (Win Rate, ROI, etc.)
# ==============================================================================
def get_performance_stats(history_df):
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return {
            "win_rate": 0.0,
            "roi": 0.0,
            "total_bets": 0,
            "total_profit": 0.0,
            "total_staked": 0.0,
            "sport_stats": {}
        }

    # Only settled bets
    settled = history_df[history_df['Result'].isin(['Win', 'Loss'])].copy()
    if settled.empty:
        return {
            "win_rate": 0.0,
            "roi": 0.0,
            "total_bets": 0,
            "total_profit": 0.0,
            "total_staked": 0.0,
            "sport_stats": {}
        }

    wins = len(settled[settled['Result'] == 'Win'])
    total = len(settled)
    profit = settled['Profit'].sum()
    staked = settled['Stake'].sum()

    win_rate = wins / total if total > 0 else 0.0
    roi = (profit / staked) if staked > 0 else 0.0

    # Per-sport breakdown
    sport_stats = {}
    if 'Sport' in settled.columns:
        for sport in settled['Sport'].unique():
            s = settled[settled['Sport'] == sport]
            if len(s) > 0:
                sport_stats[sport] = {
                    "win_rate": len(s[s['Result'] == 'Win']) / len(s),
                    "profit": s['Profit'].sum(),
                    "bets": len(s)
                }

    return {
        "win_rate": win_rate,
        "roi": roi,
        "total_bets": total,
        "total_profit": profit,
        "total_staked": staked,
        "sport_stats": sport_stats
    }

# ==============================================================================
# 3. UI HELPERS & STYLING
# ==============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        .gradient-text {
            background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3em;
            padding-bottom: 10px;
            text-align: center;
        }
        
        .bet-card {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            transition: all 0.2s;
        }
        .bet-card:hover {
            transform: translateY(-2px);
            border-color: #00C9FF;
            box-shadow: 0 4px 12px rgba(0, 201, 255, 0.15);
        }
        
        .odds-box {
            background-color: #262a3b;
            color: #00e676;
            font-weight: 700;
            font-size: 1.2em;
            padding: 8px 16px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #00e676;
        }
        
        .badge {
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 0.75em;
            font-weight: 800;
            text-transform: uppercase;
            display: inline-block;
            margin-right: 6px;
        }
        .badge-arb { background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%); color: #000; }
        .badge-high { background-color: #ff4b4b; color: white; }
        .badge-safe { background-color: #00e676; color: #000; }
        .badge-std { background-color: #31333F; color: #ccc; border: 1px solid #555; }
        
        .res-win { background-color: #d4edda; color: #155724; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
        .res-loss { background-color: #f8d7da; color: #721c24; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
        .res-push { background-color: #e2e3e5; color: #383d41; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
        .res-pending { color: #ffc107; font-weight: bold; font-style: italic; background: rgba(255,193,7,0.2); padding: 4px 8px; border-radius: 6px; }
        
        div[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #00C9FF; }
        .stButton>button { background: linear-gradient(45deg, #00C9FF, #92FE9D); border: none; color: black; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

def get_team_emoji(sport):
    emojis = {
        "Soccer": "⚽",
        "NFL": "NFL",
        "NBA": "NBA",
        "MLB": "MLB",
        "Tennis": "Tennis",
        "Cricket": "Cricket"
    }
    return emojis.get(sport, "Sports")

def get_risk_badge(row):
    if pd.isna(row.get('Edge')) or pd.isna(row.get('Odds')):
        return '<span class="badge badge-std">VALUE</span>'
    
    edge = row['Edge']
    odds = row['Odds']
    conf = row.get('Confidence', 0.5)
    bet_type = row.get('Bet Type', '')

    if bet_type == 'ARBITRAGE':
        return '<span class="badge badge-arb">Arbitrage</span>'
    if odds > 3.5 and edge > 0.15:
        return '<span class="badge badge-high">High Risk</span>'
    if conf > 0.65 and edge > 0.08:
        return '<span class="badge badge-safe">Anchor</span>'
    return '<span class="badge badge-std">Value</span>'

def format_result_badge(result):
    if pd.isna(result):
        return '<span class="res-pending">Pending</span>'
    result = str(result).strip()
    if result == 'Win':
        return '<span class="res-win">WIN</span>'
    elif result == 'Loss':
        return '<span class="res-loss">LOSS</span>'
    elif result == 'Push':
        return '<span class="res-push">PUSH</span>'
    else:
        return '<span class="res-pending">Pending</span>'
