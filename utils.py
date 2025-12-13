# utils.py
# Shared functions for data loading, styling, and logic.
# v82.1 - REMOVED: Config.py dependency
# ADDED: Google-based score processing and schema validation (Based on uploaded file structure)
# NOTE: This version aligns with the Google-only betting_engine.py and backend_runner.py

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import numpy as np
import logging
import os
import time
from bs4 import BeautifulSoup
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BettingUtils")

# HARDCODED GITHUB REPO (Replaces config)
GITHUB_USERNAME = "jd0913" # Replace with your actual username
GITHUB_REPO = "betting-copilot-pro" # Replace with your actual repo name

# Fixed URL formatting
LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"

# Use the schema defined in backend_runner.py
# We'll define it here locally as it's needed for loading data
BET_SCHEMA = {
    'Date': 'datetime64[ns, UTC]',
    'Date_Generated': 'datetime64[ns, UTC]',
    'Sport': 'string',
    'League': 'string',
    'Match': 'string',
    'Bet_Type': 'string', # Renamed from 'Bet Type' for consistency
    'Bet': 'string',
    'Odds': 'float64',
    'Edge': 'float64',
    'Confidence': 'float64',
    'Stake': 'float64',
    'Info': 'string',
    'Result': 'category',
    'Profit': 'float64',
    'Score': 'string'
}

@st.cache_data(ttl=600)
def load_data(url):
    """Safely load data from GitHub with proper schema validation"""
    try:
        logger.info(f"Loading data from: {url}")
        response = requests.get(url, timeout=10) # Use a reasonable timeout
        response.raise_for_status()
        
        # Read CSV with proper error handling
        df = pd.read_csv(url)
        
        if df.empty:
            logger.info("Data file is empty")
            return "NO_BETS_FOUND"
        
        # Validate and enforce schema
        df = validate_bet_schema(df)
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

def validate_bet_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe matches expected schema with proper dtypes"""
    for col, dtype in BET_SCHEMA.items():
        if col not in df.columns:
            if "datetime" in dtype:
                df[col] = pd.NaT
            elif "float" in dtype:
                df[col] = 0.0
            elif "category" in dtype:
                df[col] = pd.Categorical([])
            else:
                df[col] = ""
    
    for col, dtype in BET_SCHEMA.items():
        try:
            if "datetime" in dtype:
                df[col] = pd.to_datetime(df[col], utc=True)
            elif "category" in dtype:
                df[col] = pd.Categorical(df[col])
            else:
                df[col] = df[col].astype(dtype)
        except Exception as e:
            logger.warning(f"Schema conversion failed for {col}: {str(e)}")
            if "float" in dtype:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    return df[list(BET_SCHEMA.keys())] # Enforce column order

def get_performance_stats(history_df):
    """Calculates live performance metrics from history with safety checks"""
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
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
        .res-auto-settled {
            color: #ff8a80;
            font-weight: 600;
            background: rgba(183, 28, 28, 0.15); /* Slightly different for auto-settled */
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
        
        /* Final score styling */
        .final-score {
            font-weight: bold;
            color: #00C9FF;
        }
        .score-nan {
            color: #888;
            font-style: italic;
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
        
        /* Bet card styling */
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

def format_result_badge(result):
    """Format result badge with proper HTML escaping"""
    if not result or pd.isna(result):
        return '<span class="res-pending">❓ UNKNOWN</span>'
    
    result = str(result).strip()
    if result.lower() in ['win', 'won']:
        return '<span class="res-win">✅ WIN</span>'
    elif result.lower() in ['loss', 'lost', 'auto-settled']:
        return '<span class="res-loss">❌ LOSS</span>' # Or use res-auto-settled for auto-settled
    elif result.lower() in ['push', 'tie']:
        return '<span class="res-push">→ PUSH</span>'
    elif result.lower() in ['pending', 'open', '']:
        return '<span class="res-pending">⏳ PENDING</span>'
    else:
        return f'<span class="res-pending">{result.upper()}</span>'

def format_currency(value):
    """Format numbers as currency with proper signs"""
    if pd.isna(value) or abs(value) < 0.01:
        return "$0.00"
    sign = "+" if value > 0 else "-"
    return f"{sign}${abs(value):,.2f}"

def get_bet_status_color(result):
    """Return color codes for different bet statuses"""
    result = str(result).lower()
    if 'win' in result:
        return "#69f0ae", "#00c853"  # Green colors
    elif 'loss' in result or 'auto-settled' in result:
        return "#ff8a80", "#ff1744"  # Red colors
    elif 'push' in result:
        return "#bdbdbd", "#757575"  # Gray colors
    else:
        return "#ffcc80", "#ffa000"  # Amber colors for pending

# --- GOOGLE-ONLY SPECIFIC FUNCTIONS (Added for alignment with betting_engine.py) ---
def get_score_from_google(match_name, match_date_str):
    """
    Scrape Google for the final score of a match.
    Args:
        match_name (str): e.g., "Arsenal vs Chelsea"
        match_date_str (str): Date in 'YYYY-MM-DD' format
    Returns:
        tuple: (home_score, away_score) or (None, None) if not found or pending.
    """
    # Construct search query including date for accuracy
    search_query = f"{match_name} {match_date_str} score"
    encoded_query = search_query.replace(' ', '+')
    url = f"https://www.google.com/search?q={encoded_query}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for Google's sports score box or similar structured element
        # Common classes for Google sports scores (these might change over time):
        score_element = (
            soup.find('div', class_='imso_mh__scr-sep') or
            soup.find('div', class_='imso_mh__s-t') or
            soup.find('div', class_='imso_mh__s-m') or
            soup.find('div', class_='imso_mh__s-sc') or
            soup.find('span', class_='imso_mh__t1-s') or
            soup.find('span', class_='imso_mh__t2-s')
        )
        
        if score_element:
            score_text = score_element.get_text().strip()
            # Look for patterns like "2 - 1", "3:0", "1-0"
            score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)', score_text)
            if score_match:
                home_score = int(score_match.group(1))
                away_score = int(score_match.group(2))
                print(f"   > Google score found for {match_name} ({match_date_str}): {home_score} - {away_score}")
                return home_score, away_score
        
        # If not found in structured elements, search in all divs/text
        all_divs = soup.find_all('div')
        for div in all_divs:
             text = div.get_text().strip()
             if ' - ' in text or ':' in text:
                 score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)', text)
                 if score_match:
                     home_score = int(score_match.group(1))
                     away_score = int(score_match.group(2))
                     print(f"   > Google score found (fallback) for {match_name} ({match_date_str}): {home_score} - {away_score}")
                     return home_score, away_score
        
        print(f"   > No score found on Google for {match_name} ({match_date_str})")
        return None, None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"   > Error fetching score from Google for {match_name} ({match_date_str}): {e}")
        return None, None
    except Exception as e:
        logger.error(f"   > Unexpected error parsing Google score for {match_name} ({match_date_str}): {e}")
        return None, None

def scrape_google_scores(date_str, league="", limit=10):
    """
    Scrape Google for historical scores for a specific date and optionally league.
    Args:
        date_str (str): Date in 'YYYY-MM-DD' format
        league (str): Optional league name to narrow search
        limit (int): Max number of matches to return
    Returns:
        list: List of dictionaries [{'Match': 'Team1 vs Team2', 'Score': 'X-Y', 'League': '...'}, ...]
    """
    search_query = f"{date_str} {league} scores"
    encoded_query = search_query.replace(' ', '+')
    url = f"https://www.google.com/search?q={encoded_query}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        logger.info(f"Scraping Google for scores: {date_str}, League: {league}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find score containers (depends on Google's layout)
        score_containers = soup.find_all('div', class_=lambda x: x and ('imso_mh' in x.lower() or 'score' in x.lower()))
        
        scores = []
        for container in score_containers:
            # Extract match info and score
            home_team_elem = container.find(class_=lambda x: x and 'imso_mh__tm-txt' in x) # First team found
            score_elem = container.find(class_=lambda x: x and ('imso_mh__scr-sep' in x or 'imso_mh__s-t' in x))
            away_team_elem = container.find_all(class_=lambda x: x and 'imso_mh__tm-txt' in x)[-1] if len(container.find_all(class_=lambda x: x and 'imso_mh__tm-txt' in x)) > 1 else None
            
            if home_team_elem and away_team_elem and score_elem:
                match_name = f"{home_team_elem.get_text(strip=True)} vs {away_team_elem.get_text(strip=True)}"
                score = score_elem.get_text(strip=True)
                scores.append({'Match': match_name, 'Score': score, 'League': league, 'Date': date_str})
                
                if len(scores) >= limit:
                    break
        
        logger.info(f"Found {len(scores)} scores from Google for {date_str}.")
        return scores
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error scraping scores for {date_str}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error parsing scores for {date_str}: {e}")
        return []

def format_result_with_score(result, score):
    """Format result with score information for display (compatible with views.py)"""
    # Handle missing or invalid scores
    if not isinstance(score, str) or score == 'N/A' or score.lower() == 'nan':
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
