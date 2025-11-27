# utils.py
# Shared functions for data loading, styling, and logic.

import streamlit as st
import pandas as pd
import requests

# --- CONFIG ---
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
            df['Date_Obj'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Formatted_Date'] = df['Date_Obj'].dt.strftime('%a, %b %d • %I:%M %p')
            df['Formatted_Date'] = df['Formatted_Date'].fillna('Time TBD')
        else:
            df['Formatted_Date'] = 'Time TBD'
            
        return df
    except: return "FILE_NOT_FOUND"

def inject_custom_css(font_choice):
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
        html, body, [class*="css"] {{ font-family: {family}; }}
        .gradient-text {{
            background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-weight: bold; font-size: 3em; padding-bottom: 10px;
        }}
        .bet-card {{
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px; padding: 15px; margin-bottom: 10px; transition: transform 0.2s;
        }}
        .bet-card:hover {{ transform: scale(1.01); border-color: #00C9FF; }}
        .badge {{
            padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; margin-right: 5px; display: inline-block;
        }}
        .badge-green {{ background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
        .badge-red {{ background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
        .badge-gray {{ background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }}
        .badge-blue {{ background-color: #cce5ff; color: #004085; border: 1px solid #b8daff; }}
        .badge-yellow {{ background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }}
        div[data-testid="stMetricValue"] {{ font-size: 1.5rem; color: #00C9FF; }}
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
    if bet_type == 'ARBITRAGE': return '<span class="badge badge-blue">💎 RISK FREE PROFIT</span>'
    if odds > 3.5 and edge > 0.15: return '<span class="badge badge-red">⚡ RISING STAR</span>'
    if conf > 0.60 and edge > 0.05: return '<span class="badge badge-green">⭐ ANCHOR BET</span>'
    if row.get('Bet') == 'Draw': return '<span class="badge badge-gray">⚖️ VALUE DRAW</span>'
    return '<span class="badge badge-gray">✅ STANDARD VALUE</span>'

def format_result_badge(result):
    if result == 'Win': return '<span class="badge badge-green">WIN</span>'
    elif result == 'Loss': return '<span class="badge badge-red">LOSS</span>'
    elif result == 'Push': return '<span class="badge badge-gray">PUSH</span>'
    elif result == 'Pending': return '<span class="badge badge-yellow">⏳ PENDING</span>'
    else: return f'<span class="badge badge-gray">{result}</span>'
