# utils.py
# Shared functions for data loading, styling, and logic.
# v63.0 - FIXED: URL formatting, CSS imports, error handling

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone
import os

# --- CONFIGURATION (SECURE METHOD) ---
# Get GitHub credentials from Streamlit Secrets or fallback to defaults
GITHUB_USERNAME = st.secrets.get("github_username", "jd0913")
GITHUB_REPO = st.secrets.get("github_repo", "betting-copilot-pro")

# FIX: Removed extra spaces in URLs that were breaking data loading
LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"

@st.cache_data(ttl=600)
def load_data(url):
    """Safely load data from GitHub with proper error handling"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Check if content is actually CSV data
        if 'text/plain' not in response.headers.get('Content-Type', '') and 'text/csv' not in response.headers.get('Content-Type', ''):
            st.warning(f"Unexpected content type: {response.headers.get('Content-Type')}")
            return "FILE_NOT_FOUND"
            
        # Read CSV with proper error handling
        df = pd.read_csv(url)
        
        if df.empty:
            st.info("No bets found - data file is empty")
            return "NO_BETS_FOUND"
        
        # Numeric conversion with error handling
        numeric_cols = ['Edge', 'Confidence', 'Odds', 'Stake', 'Profit']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Date Formatting with UTC handling
        if 'Date' in df.columns:
            # Handle different date formats
            df['Date_Obj'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
            # Filter out NaT values
            df = df[df['Date_Obj'].notna()]
            # Format dates in local timezone
            df['Formatted_Date'] = df['Date_Obj'].dt.strftime('%a, %b %d • %I:%M %p')
        else:
            df['Formatted_Date'] = 'Time TBD'
            
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error loading data: {str(e)}")
        return "FILE_NOT_FOUND"
    except pd.errors.EmptyDataError:
        st.warning("Data file is empty")
        return "NO_BETS_FOUND"
    except pd.errors.ParserError as e:
        st.error(f"CSV parsing error: {str(e)}")
        return "FILE_NOT_FOUND"
    except Exception as e:
        st.exception(f"Unexpected error loading data: {str(e)}")
        return "FILE_NOT_FOUND"

def get_performance_stats(history_df):
    """Calculates live performance metrics from history with safety checks"""
    if not isinstance(history_df, pd.DataFrame):
        return {"win_rate": 0.0, "roi": 0.0, "total_bets": 0, "sport_stats": {}}
    
    # Filter only settled bets with valid results
    settled_mask = history_df['Result'].isin(['Win', 'Loss', 'Push']) if 'Result' in history_df.columns else pd.Series(False, index=history_df.index)
    settled = history_df[settled_mask]
    
    if settled.empty:
        return {"win_rate": 0.0, "roi": 0.0, "total_bets": 0, "sport_stats": {}}
    
    # Calculate win rate (excluding pushes)
    win_mask = settled['Result'] == 'Win'
    loss_mask = settled['Result'] == 'Loss'
    wins = win_mask.sum()
    losses = loss_mask.sum()
    total_decided = wins + losses
    
    win_rate = wins / total_decided if total_decided > 0 else 0.0
    
    # Calculate ROI with safety checks
    if 'Profit' in settled.columns and 'Stake' in settled.columns:
        total_profit = settled['Profit'].sum()
        total_staked = settled['Stake'].sum()
        roi = total_profit / total_staked if total_staked > 0 else 0.0
    else:
        roi = 0.0
        total_staked = 0
    
    # Per Sport Stats
    sport_stats = {}
    if 'Sport' in settled.columns and total_decided > 0:
        for sport in settled['Sport'].unique():
            s_df = settled[settled['Sport'] == sport]
            s_wins = (s_df['Result'] == 'Win').sum()
            s_losses = (s_df['Result'] == 'Loss').sum()
            s_total = s_wins + s_losses
            
            if s_total > 0:
                sport_stats[sport] = s_wins / s_total
    
    return {
        "win_rate": win_rate,
        "roi": roi,
        "total_bets": len(settled),
        "total_staked": total_staked,
        "total_profit": total_profit if 'Profit' in settled.columns else 0.0,
        "sport_stats": sport_stats
    }

def inject_custom_css(font_choice="Clean (Inter)"):
    """Inject custom CSS with fixed font import and improved styling"""
    # FIX: Removed space in font URL that was breaking CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        /* Base styling */
        html, body, [class*="css"] { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
            color: #e0e0e0;
        }
        
        /* Header styling */
        h1, h2, h3, h4, h5, h6 {
            color: white;
            font-weight: 700;
        }
        
        /* Card styling */
        .bet-card {
            background: linear-gradient(145deg, #1a1c23, #16181d);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .bet-card:hover {
            transform: translateY(-3px);
            border-color: #00C9FF;
            box-shadow: 0 6px 12px rgba(0, 201, 255, 0.15);
        }
        
        /* Odds box styling */
        .odds-box {
            background: linear-gradient(90deg, #1a1c23, #1e2230);
            color: #00e676;
            font-weight: 700;
            font-size: 1.2em;
            padding: 10px 18px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid rgba(0, 230, 118, 0.3);
            margin: 8px 0;
        }
        
        /* Badge styling */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 0.75em;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 2px 5px 2px 0;
        }
        .badge-arb {
            background: linear-gradient(90deg, #00C9FF, #92FE9D);
            color: #000;
            border: none;
        }
        .badge-high {
            background: linear-gradient(90deg, #ff4d4d, #ff8c8c);
            color: white;
        }
        .badge-safe {
            background: linear-gradient(90deg, #00e676, #4dffbd);
            color: #000;
        }
        .badge-std {
            background: rgba(49, 51, 63, 0.8);
            color: #aaa;
            border: 1px solid #444;
        }
        
        /* Result badges */
        .res-win {
            background: rgba(27, 94, 32, 0.2);
            color: #69f0ae;
            border: 1px solid rgba(105, 240, 174, 0.3);
            padding: 3px 10px;
            border-radius: 8px;
            font-weight: 600;
        }
        .res-loss {
            background: rgba(183, 28, 28, 0.2);
            color: #ff8a80;
            border: 1px solid rgba(255, 138, 128, 0.3);
            padding: 3px 10px;
            border-radius: 8px;
            font-weight: 600;
        }
        .res-push {
            background: rgba(84, 84, 84, 0.2);
            color: #bdbdbd;
            border: 1px solid rgba(189, 189, 189, 0.3);
            padding: 3px 10px;
            border-radius: 8px;
            font-weight: 600;
        }
        .res-pending {
            color: #ffcc80;
            font-weight: 600;
            font-style: normal;
            padding: 3px 0;
        }
        
        /* Gradient text for headers */
        .gradient-text {
            background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800;
            font-size: 2.8em;
            padding-bottom: 10px;
            line-height: 1.2;
        }
        
        /* Metric styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            background: linear-gradient(90deg, #00C9FF, #92FE9D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #1a1c23, #1e2230);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
            border-radius: 10px;
            padding: 8px 20px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            border-color: #00C9FF;
            background: linear-gradient(90deg, #1c1e28, #222638);
            box-shadow: 0 0 15px rgba(0, 201, 255, 0.2);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1117, #131621);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
    """, unsafe_allow_html=True)

def get_team_emoji(sport):
    """Return appropriate emoji for sport with fallback"""
    emoji_map = {
        "Soccer": "⚽",
        "NFL": "🏈",
        "NBA": "🏀",
        "MLB": "⚾",
        "Tennis": "🎾",
        "MMA": "🥊",
        "Boxing": "🥊",
        "Hockey": "🏒",
        "Golf": "⛳"
    }
    return emoji_map.get(sport, "🏅")

def get_risk_badge(row):
    """Generate risk badge HTML with proper edge/confidence handling"""
    try:
        edge = float(row.get('Edge', 0))
        odds = float(row.get('Odds', 0))
        conf = float(row.get('Confidence', 0))
        bet_type = row.get('Bet_Type', row.get('Bet Type', ''))
        
        if bet_type == 'ARBITRAGE':
            return '<span class="badge badge-arb">💎 ARB</span>'
        if odds > 3.5 and edge > 0.15:
            return '<span class="badge badge-high">⚡ HIGH</span>'
        if conf > 0.60 and edge > 0.07:
            return '<span class="badge badge-safe">⭐ VALUE</span>'
        if edge > 0.02:
            return '<span class="badge badge-std">EDGE</span>'
        return '<span class="badge badge-std">STANDARD</span>'
    except (ValueError, TypeError):
        return '<span class="badge badge-std">N/A</span>'

def format_result_badge(result):
    """Format result badge with proper HTML escaping"""
    if not result or pd.isna(result):
        return '<span class="res-pending">❓ UNKNOWN</span>'
    
    result = str(result).strip()
    if result.lower() in ['win', 'won']:
        return '<span class="res-win">✓ WIN</span>'
    elif result.lower() in ['loss', 'lost']:
        return '<span class="res-loss">✗ LOSS</span>'
    elif result.lower() in ['push', 'tie']:
        return '<span class="res-push">→ PUSH</span>'
    elif result.lower() in ['pending', 'open', '']:
        return '<span class="res-pending">⏳ PENDING</span>'
    else:
        return f'<span class="res-pending">{result.upper()}</span>'

def format_currency(value):
    """Format numbers as currency with proper signs"""
    if pd.isna(value) or value == 0:
        return "$0.00"
    sign = "+" if value > 0 else "-"
    return f"{sign}${abs(value):,.2f}"
