# utils.py
# Shared functions for data loading, styling, and logic.
# v62.0 - Added Performance Calculation Logic

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

def get_performance_stats(history_df):
    """Calculates live performance metrics from history."""
    if not isinstance(history_df, pd.DataFrame) or 'Result' not in history_df.columns:
        return {"win_rate": 0.0, "roi": 0.0, "total_bets": 0, "sport_stats": {}}
    
    settled = history_df[history_df['Result'].isin(['Win', 'Loss'])]
    if settled.empty:
        return {"win_rate": 0.0, "roi": 0.0, "total_bets": 0, "sport_stats": {}}
    
    wins = len(settled[settled['Result'] == 'Win'])
    total = len(settled)
    profit = settled['Profit'].sum()
    total_staked = settled['Stake'].sum() if 'Stake' in settled.columns else total # Approx
    
    win_rate = wins / total
    roi = profit / total_staked if total_staked > 0 else 0.0
    
    # Per Sport Stats
    sport_stats = {}
    if 'Sport' in settled.columns:
        for sport in settled['Sport'].unique():
            s_df = settled[settled['Sport'] == sport]
            s_wins = len(s_df[s_df['Result'] == 'Win'])
            s_total = len(s_df)
            if s_total > 0:
                sport_stats[sport] = s_wins / s_total
    
    return {"win_rate": win_rate, "roi": roi, "total_bets": total, "sport_stats": sport_stats}

def inject_custom_css(font_choice="Clean (Inter)"):
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
