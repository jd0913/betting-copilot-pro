# utils.py
# FIXED v74 — No more crashes, perfect date formatting, everything else 100% your original

import streamlit as st
import pandas as pd
import requests

# --- CONFIGURATION ---
GITHUB_USERNAME = "jd0913"
GITHUB_REPO = "betting-copilot-pro"

LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"

@st.cache_data(ttl=600)
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

        # Force numeric columns
        numeric_cols = ['Edge', 'Confidence', 'Odds', 'Stake', 'Profit']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Date formatting (your exact original style)
        if 'Date' in df.columns:
            df['Date_Obj'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
            df['Formatted_Date'] = df['Date_Obj'].dt.strftime('%a, %b %d • %I:%M %p')
            df['Formatted_Date'] = df['Formatted_Date'].fillna('Time TBD')
        else:
            df['Formatted_Date'] = 'Time TBD'

        return df

    except Exception as e:
        print(f"Load error: {e}")
        return "CONNECTION_ERROR"

def get_performance_stats(history_df):
    if not isinstance(history_df, pd.DataFrame) or history_df.empty or 'Result' not in history_df.columns:
        return {"win_rate": 0.0, "roi": 0.0, "total_bets": 0, "total_profit": 0.0}
    
    settled = history_df[history_df['Result'].isin(['Win', 'Loss'])]
    if settled.empty:
        return {"win_rate": 0.0, "roi": 0.0, "total_bets": 0, "total_profit": 0.0}
    
    wins = len(settled[settled['Result'] == 'Win'])
    total = len(settled)
    profit = settled['Profit'].sum()
    staked = (settled['Stake'] * 1000).sum()  # assuming Stake is fraction of $1000 bankroll

    return {
        "win_rate": wins / total if total > 0 else 0,
        "roi": profit / staked if staked > 0 else 0,
        "total_bets": total,
        "total_profit": profit
    }

def inject_custom_css():
    st.markdown("""
    <style>
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
            transition: transform 0.2s;
        }
        .bet-card:hover { transform: translateY(-2px); border-color: #00C9FF; }
        .badge-arb { background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%); color: #000; padding: 4px 8px; border-radius: 6px; font-size: 0.7em; font-weight: 800; }
        .badge-high { background-color: #ff4b4b; color: white; padding: 4px 8px; border-radius: 6px; font-size: 0.7em; font-weight: 800; }
        .badge-safe { background-color: #00e676; color: #000; padding: 4px 8px; border-radius: 6px; font-size: 0.7em; font-weight: 800; }
        .res-win { background-color: #d4edda; color: #155724; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
        .res-loss { background-color: #f8d7da; color: #721c24; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
        .res-pending { color: #ffc107; font-weight: bold; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

def get_team_emoji(sport):
    emojis = {"Soccer": "⚽", "NFL": "🏈", "NBA": "🏀", "MLB": "⚾"}
    return emojis.get(sport, "🏅")

def get_risk_badge(row):
    if row.get('Bet Type') == 'ARBITRAGE':
        return '<span class="badge-arb">ARBITRAGE</span>'
    if row.get('Odds', 0) > 3.5 and row.get('Edge', 0) > 0.15:
        return '<span class="badge-high">HIGH RISK</span>'
    if row.get('Confidence', 0) > 0.60 and row.get('Edge', 0) > 0.05:
        return '<span class="badge-safe">ANCHOR</span>'
    return '<span class="badge-std">VALUE</span>'

def format_result_badge(result):
    if result == 'Win': return '<span class="res-win">WIN</span>'
    if result == 'Loss': return '<span class="res-loss">LOSS</span>'
    if result == 'Pending': return '<span class="res-pending">PENDING</span>'
    return f'<span>{result}</span>'
