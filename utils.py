# utils.py
# Shared functions for data loading, styling, and logic.
# v71.0 (Final Score Settlement + Active Bets Fix)
# FIX: Get final scores from The Odds API
# FIX: Properly settle all bets before Dec 8, 2025
# FIX: Show active/future bets in Command Center

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import numpy as np
import logging
import os
import config
from fuzzywuzzy import process
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BettingUtils")

# --- CONFIGURATION ---
# Get GitHub credentials from Streamlit Secrets or config
GITHUB_USERNAME = st.secrets.get("github_username", config.GITHUB_USERNAME if hasattr(config, 'GITHUB_USERNAME') else "jd0913")
GITHUB_REPO = st.secrets.get("github_repo", config.GITHUB_REPO if hasattr(config, 'GITHUB_REPO') else "betting-copilot-pro")

# Fixed URL formatting
LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"

@st.cache_data(ttl=600)
def load_data(url):
    """Safely load data from GitHub with proper error handling"""
    try:
        logger.info(f"Loading data from: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('Content-Type', '')
        if 'csv' not in content_type and 'text/plain' not in content_type and 'text/csv' not in content_type:
            logger.warning(f"Unexpected content type: {content_type}")
            return "FILE_NOT_FOUND"
        
        # Read CSV with proper error handling
        df = pd.read_csv(url)
        
        if df.empty:
            logger.info("Data file is empty")
            return "NO_BETS_FOUND"
        
        # Numeric conversion with error handling
        numeric_cols = ['Edge', 'Confidence', 'Odds', 'Stake', 'Profit']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Date Formatting with UTC handling
        if 'Date' in df.columns:
            try:
                # Try to parse date with flexible formats
                df['Date_Obj'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
            except:
                # Fallback for different date formats
                df['Date_Obj'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce', utc=True)
            
            # Filter out NaT values and format dates
            df = df[df['Date_Obj'].notna()].copy()
            df['Formatted_Date'] = df['Date_Obj'].dt.strftime('%a, %b %d • %I:%M %p')
        else:
            df['Formatted_Date'] = 'Time TBD'
            df['Date_Obj'] = pd.NaT
        
        # FIX: Handle Score column properly
        if 'Score' in df.columns:
            df['Score'] = df['Score'].fillna('N/A').astype(str)
            # Clean up any 'nan' string values
            df['Score'] = df['Score'].apply(lambda x: 'N/A' if x.lower() in ['nan', ''] else x)
        else:
            df['Score'] = 'N/A'
        
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error loading {url}: {str(e)}")
        return "FILE_NOT_FOUND"
    except pd.errors.EmptyDataError:
        logger.warning("Data file is empty")
        return "NO_BETS_FOUND"
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {str(e)}")
        return "FILE_NOT_FOUND"
    except Exception as e:
        logger.exception(f"Unexpected error loading {url}: {str(e)}")
        return "FILE_NOT_FOUND"

def fetch_odds_api_scores(sport_key, days=7):
    """Fetch historical scores from The Odds API for a specific sport"""
    api_key = config.API_CONFIG.get("THE_ODDS_API_KEY", "").strip()
    if not api_key or "PASTE_YOUR" in api_key:
        logger.warning("Odds API key not configured")
        return pd.DataFrame()
    
    try:
        # Get past results (yesterday to days ago)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
        params = {
            'api_key': api_key,
            'daysFrom': days,
            'dateFormat': 'iso'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        # Process scores
        scores = []
        for game in data:
            if not game.get('scores') or not game.get('completed'):
                continue
                
            # Get final scores
            home_score = None
            away_score = None
            for score in game['scores']:
                if score['name'] == game['home_team']:
                    home_score = score['score']
                elif score['name'] == game['away_team']:
                    away_score = score['score']
            
            if home_score is not None and away_score is not None:
                scores.append({
                    'Match': f"{game['home_team']} vs {game['away_team']}",
                    'Date_Obj': pd.to_datetime(game['commence_time'], utc=True),
                    'HomeScore': home_score,
                    'AwayScore': away_score,
                    'Sport': sport_key
                })
        
        return pd.DataFrame(scores)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching scores from Odds API: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        logger.exception(f"Unexpected error fetching scores: {str(e)}")
        return pd.DataFrame()

def settle_with_odds_api_scores(history_df):
    """Settle bets using scores from The Odds API"""
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return history_df
    
    current_time = datetime.now(timezone.utc)
    
    # Define sports to check (Soccer, NFL, NBA, MLB)
    sports_to_check = {
        'soccer_epl': 'soccer_epl',
        'NFL': 'americanfootball_nfl',
        'NBA': 'basketball_nba',
        'MLB': 'baseball_mlb'
    }
    
    # Fetch scores for each sport
    all_scores = []
    for sport_name, odds_sport_key in sports_to_check.items():
        sport_scores = fetch_odds_api_scores(odds_sport_key, days=14)  # Get last 14 days of scores
        if not sport_scores.empty:
            all_scores.append(sport_scores)
    
    if not all_scores:
        logger.warning("No scores retrieved from Odds API")
        return history_df
    
    scores_df = pd.concat(all_scores, ignore_index=True)
    
    # Settle each bet
    for idx, row in history_df.iterrows():
        try:
            match_date = pd.to_datetime(row['Date_Obj'], utc=True)
        except (ValueError, TypeError):
            continue
        
        match_name = row['Match']
        predicted_bet = row['Bet']
        current_result = row.get('Result', 'Pending')
        
        # Only settle past matches (before Dec 8, 2025 @ 10pm UTC)
        settlement_deadline = pd.to_datetime("2025-12-08 22:00:00", utc=True)
        if match_date > settlement_deadline:
            continue  # Skip future matches
        
        # Skip if already settled with a proper score
        if current_result in ['Win', 'Loss', 'Push'] and row.get('Score', 'N/A') != 'N/A' and row.get('Score', 'N/A') != 'nan':
            continue
        
        # Find matching score
        if not scores_df.empty:
            # Try exact match first
            exact_match = scores_df[scores_df['Match'] == match_name]
            if not exact_match.empty:
                score_row = exact_match.iloc[0]
                home_score = score_row['HomeScore']
                away_score = score_row['AwayScore']
                actual_score = f"{home_score} - {away_score}"
                
                # Determine actual outcome
                if home_score > away_score:
                    actual_result = 'Home Win'
                elif away_score > home_score:
                    actual_result = 'Away Win'
                else:
                    actual_result = 'Draw'
                
                # Determine if bet won
                if (predicted_bet == 'Home Win' and actual_result == 'Home Win') or \
                   (predicted_bet == 'Away Win' and actual_result == 'Away Win') or \
                   (predicted_bet == 'Draw' and actual_result == 'Draw'):
                    history_df.at[idx, 'Result'] = 'Win'
                    history_df.at[idx, 'Profit'] = row['Stake'] * (row['Odds'] - 1)
                else:
                    history_df.at[idx, 'Result'] = 'Loss'
                    history_df.at[idx, 'Profit'] = -row['Stake']
                
                history_df.at[idx, 'Score'] = actual_score
                continue
        
        # If no score found but match is old, mark as Auto-Settled (assume loss)
        if match_date < current_time - timedelta(days=3) and current_result == 'Pending':
            history_df.at[idx, 'Result'] = 'Auto-Settled'
            history_df.at[idx, 'Profit'] = -row.get('Stake', 0)
            history_df.at[idx, 'Score'] = 'N/A'
    
    return history_df

def get_performance_stats(history_df):
    """Calculates live performance metrics from history with safety checks"""
    if not isinstance(history_df, pd.DataFrame):
        return {"win_rate": 0.0, "roi": 0.0, "total_bets": 0, "sport_stats": {}}
    
    # Filter only settled bets with valid results
    settled_mask = history_df['Result'].isin(['Win', 'Loss', 'Push', 'Auto-Settled'])
    settled = history_df[settled_mask]
    
    if settled.empty:
        return {"win_rate": 0.0, "roi": 0.0, "total_bets": 0, "sport_stats": {}}
    
    # Calculate win rate (excluding pushes)
    win_mask = settled['Result'] == 'Win'
    loss_mask = settled['Result'].isin(['Loss', 'Auto-Settled'])
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
            s_losses = (s_df['Result'].isin(['Loss', 'Auto-Settled'])).sum()
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

def inject_custom_css():
    """Inject modern CSS with proper font imports and styling"""
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
        
        /* Dataframe styling */
        .dataframe {
            background-color: #1a1c23 !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #e0e0e0 !important;
        }
        .dataframe th {
            background-color: #1e2130 !important;
            color: #8b92a5 !important;
            font-weight: 600 !important;
            text-align: center !important;
        }
        .dataframe td {
            text-align: center !important;
            background-color: #16181d !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        .dataframe tr:hover {
            background-color: rgba(0, 201, 255, 0.05) !important;
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
        
        /* Clean result badges */
        .res-win {
            color: #69f0ae;
            font-weight: 600;
            background: rgba(27, 94, 32, 0.1);
            padding: 2px 8px;
            border-radius: 6px;
            display: inline-block;
        }
        .res-loss {
            color: #ff8a80;
            font-weight: 600;
            background: rgba(183, 28, 28, 0.1);
            padding: 2px 8px;
            border-radius: 6px;
            display: inline-block;
        }
        .res-push {
            color: #bdbdbd;
            font-weight: 600;
            background: rgba(84, 84, 84, 0.1);
            padding: 2px 8px;
            border-radius: 6px;
            display: inline-block;
        }
        .res-pending {
            color: #ffcc80;
            font-weight: 600;
            background: rgba(255, 204, 0, 0.1);
            padding: 2px 8px;
            border-radius: 6px;
            display: inline-block;
        }
        .res-active {
            color: #00C9FF;
            font-weight: 600;
            background: rgba(0, 201, 255, 0.1);
            padding: 2px 8px;
            border-radius: 6px;
            display: inline-block;
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
        
        /* Active bet status */
        .status-active {
            color: #00C9FF;
            font-weight: bold;
        }
        .status-pending {
            color: #ffcc80;
            font-style: italic;
        }
        
        /* Final score styling */
        .final-score {
            font-weight: bold;
            color: #00C9FF;
        }
        .score-nan {
            color: #888;
            font-style: italic;
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
    """Generate clean risk badge text with proper edge/confidence handling"""
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

def format_result_with_score(result, score):
    """Format result with score information for display"""
    # Handle missing or invalid scores
    if not score or score == 'N/A' or score.lower() == 'nan':
        if result in ['Win', 'Loss', 'Push', 'Auto-Settled']:
            return f'<span class="res-{result.lower()}">{result.upper()}</span>'
        return '<span class="res-pending">⏳ PENDING</span>'
    
    # Format based on result
    if result == 'Win':
        return f'<span class="res-win">✅ WIN ({score})</span>'
    elif result == 'Loss':
        return f'<span class="res-loss">❌ LOSS ({score})</span>'
    elif result == 'Push':
        return f'<span class="res-push">⚖️ PUSH ({score})</span>'
    elif result == 'Auto-Settled':
        return f'<span class="res-loss">🔄 AUTO-SETTLED ({score})</span>'
    else:
        return f'<span class="res-pending">{result.upper()} ({score})</span>'

def is_active_bet(row):
    """Determine if a bet is for a future match (active)"""
    try:
        match_date = pd.to_datetime(row['Date_Obj'], utc=True)
        current_time = datetime.now(timezone.utc)
        return match_date > current_time - timedelta(hours=2)  # Consider matches in last 2 hours as active
    except (ValueError, TypeError):
        return True

def get_active_bets(latest_df):
    """Get only active/future bets from latest recommendations"""
    if not isinstance(latest_df, pd.DataFrame) or latest_df.empty:
        return pd.DataFrame()
    
    current_time = datetime.now(timezone.utc)
    
    # Filter for active/future bets
    active_mask = latest_df.apply(is_active_bet, axis=1)
    active_bets = latest_df[active_mask].copy()
    
    # Sort by date
    if 'Date_Obj' in active_bets.columns:
        active_bets = active_bets.sort_values('Date_Obj', ascending=True)
    
    return active_bets

def format_currency(value):
    """Format numbers as currency with proper signs"""
    if pd.isna(value) or abs(value) < 0.01:
        return "$0.00"
    sign = "+" if value > 0 else "-"
    return f"{sign}${abs(value):,.2f}"

def get_bet_status_color(result):
    """Return color codes for different bet statuses"""
    if 'Win' in result:
        return "#69f0ae", "#00c853"
    elif 'Loss' in result or 'Auto-Settled' in result:
        return "#ff8a80", "#ff1744"
    elif 'Push' in result:
        return "#bdbdbd", "#757575"
    else:
        return "#ffcc80", "#ffa000"
