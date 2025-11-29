# utils.py — v67.0 FINAL
import streamlit as st
import pandas as pd
import requests
import hashlib

GITHUB_USERNAME = "jd0913"
GITHUB_REPO = "betting-copilot-pro"
LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"

def _load_csv_safely(url):
    try:
        df = pd.read_csv(url)
        if df.empty:
            return pd.DataFrame()
        numeric_cols = ['Edge', 'Confidence', 'Odds', 'Stake', 'Profit']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if 'Date' in df.columns:
            df['Date_Obj'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Formatted_Date'] = df['Date_Obj'].dt.strftime('%a, %b %d • %I:%M %p').fillna('Time TBD')
        else:
            df['Formatted_Date'] = 'Time TBD'
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner="Updating live odds...")
def get_latest_bets():
    return _load_csv_safely(LATEST_URL)

@st.cache_data(ttl=600, show_spinner="Loading history...")
def get_history():
    df = _load_csv_safely(HISTORY_URL)
    if not df.empty:
        df['Profit'] = df['Profit'].fillna(0.0)
        df['Stake'] = df['Stake'].fillna(1.0)
    return df

def get_performance_stats(history_df):
    if history_df.empty or 'Result' not in history_df.columns:
        return {"win_rate": 0.0, "roi": 0.0, "total_bets": 0, "profit": 0.0, "sport_stats": {}}
    settled = history_df[history_df['Result'].isin(['Win', 'Loss'])].copy()
    if settled.empty:
        return {"win_rate": 0.0, "roi": 0.0, "total_bets": 0, "profit": 0.0, "sport_stats": {}}
    wins = len(settled[settled['Result'] == 'Win'])
    total = len(settled)
    profit = settled['Profit'].sum()
    total_staked = settled['Stake'].sum() or total
    win_rate = wins / total if total > 0 else 0.0
    roi = profit / total_staked if total_staked > 0 else 0.0
    sport_stats = {}
    if 'Sport' in settled.columns:
        for sport in settled['Sport'].unique():
            s = settled[settled['Sport'] == sport]
            sport_stats[sport] = len(s[s['Result'] == 'Win']) / len(s) if len(s) > 0 else 0.0
    return {
        "win_rate": round(win_rate, 3),
        "roi": round(roi, 4),
        "total_bets": total,
        "profit": round(profit, 2),
        "sport_stats": sport_stats
    }

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .gradient-text { background: linear-gradient(90deg, #00C9FF, #92FE9D); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 3.2em; }
        .bet-card, .bet-ticket { background: rgba(30, 33, 48, 0.6); border-radius: 12px; padding: 16px; margin: 10px 0; border: 1px solid #333; }
        .odds-box { background: #00e676; color: black; font-weight: 800; padding: 10px 20px; border-radius: 8px; font-size: 1.4em; }
        .badge-arb { background: linear-gradient(90deg, #00C9FF, #92FE9D); color: black; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
        .badge-high { background: #ff4444; color: white; padding: 4px 10px; border-radius: 6px; }
        .badge-safe { background: #00e676; color: black; padding: 4px 10px; border-radius: 6px; }
    </style>
    """, unsafe_allow_html=True)

def get_team_emoji(sport):
    return {"Soccer": "Soccer", "NFL": "Football", "NBA": "Basketball", "MLB": "Baseball"}.get(sport, "Trophy")

def get_risk_badge(row):
    if row.get('Bet Type') == 'ARBITRAGE':
        return '<span class="badge-arb">ARBITRAGE</span>'
    if row.get('Odds', 0) > 3.5 and row.get('Edge', 0) > 0.15:
        return '<span class="badge-high">HIGH RISK</span>'
    if row.get('Confidence', 0) > 0.60 and row.get('Edge', 0) > 0.05:
        return '<span class="badge-safe">ANCHOR</span>'
    return '<span style="background:#31333F;color:#ccc;padding:4px 10px;border-radius:6px;">VALUE</span>'
