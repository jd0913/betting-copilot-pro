# utils.py
# Updated version for Google-only approach
# v73.0 - Removed Streamlit Secrets dependency, uses Google for all data

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

# Configuration (using GitHub URLs directly)
GITHUB_USERNAME = "jd0913"
GITHUB_REPO = "betting-copilot-pro"

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
                df['Date_Obj'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
            except:
                df['Date_Obj'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce', utc=True)
            
            df = df[df['Date_Obj'].notna()].copy()
            df['Formatted_Date'] = df['Date_Obj'].dt.strftime('%a, %b %d • %I:%M %p')
        else:
            df['Formatted_Date'] = 'Time TBD'
            df['Date_Obj'] = pd.NaT
        
        # Handle Score column properly
        if 'Score' in df.columns:
            df['Score'] = df['Score'].fillna('N/A').astype(str)
            df['Score'] = df['Score'].apply(lambda x: 'N/A' if x.lower() in ['nan', ''] else x)
        else:
            df['Score'] = 'N/A'
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {url}: {str(e)}")
        return "FILE_NOT_FOUND"

def get_google_score(match_name):
    """
    Get match score directly from Google search results with enhanced parsing
    Args:
        match_name (str): Match name like "Arsenal vs Chelsea"
    Returns:
        str: Score in format "X - Y" or None if not found
    """
    try:
        # Format search query
        search_query = f"{match_name} score"
        encoded_query = search_query.replace(' ', '+')
        url = f"https://www.google.com/search?q={encoded_query}"
        
        # Add headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for score elements (Google uses various class names)
        score_elements = [
            # Primary score elements
            soup.find('div', class_='imso_mh__scr-sep'),
            soup.find('div', class_='imso_mh__s-t'),
            soup.find('div', class_='imso_mh__s-m'),
            # Alternative elements
            soup.find('div', class_='imso_mh__s-sc'),
            soup.find('span', class_='imso_mh__t1-s'),
            soup.find('span', class_='imso_mh__t2-s'),
            # Generic score containers
            soup.find('div', string=re.compile(r'\d+\s*[-:]\s*\d+')),
        ]
        
        # Try to extract score from elements
        for element in score_elements:
            if element:
                score_text = element.get_text().strip()
                # Look for patterns like "2 - 1", "3:0", "1-0"
                score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)', score_text)
                if score_match:
                    home_score = score_match.group(1)
                    away_score = score_match.group(2)
                    return f"{home_score} - {away_score}"
        
        # If not found in structured elements, search in all divs
        all_divs = soup.find_all('div')
        for div in all_divs:
            text = div.get_text().strip()
            if ' - ' in text:
                score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)', text)
                if score_match:
                    home_score = score_match.group(1)
                    away_score = score_match.group(2)
                    return f"{home_score} - {away_score}"
        
        # Try different search patterns
        alternative_queries = [
            f"{match_name} final score",
            f"{match_name} result",
            f"{match_name} live score",
            f"{match_name} today score"
        ]
        
        for alt_query in alternative_queries:
            encoded_alt_query = alt_query.replace(' ', '+')
            alt_url = f"https://www.google.com/search?q={encoded_alt_query}"
            alt_response = requests.get(alt_url, headers=headers, timeout=10)
            if alt_response.status_code == 200:
                alt_soup = BeautifulSoup(alt_response.text, 'html.parser')
                
                # Search for scores in alternative results
                alt_score_elements = alt_soup.find_all(string=re.compile(r'\d+\s*[-:]\s*\d+'))
                for score_text in alt_score_elements:
                    score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)', score_text)
                    if score_match:
                        home_score = score_match.group(1)
                        away_score = score_match.group(2)
                        return f"{home_score} - {away_score}"
        
        return None
        
    except Exception as e:
        logger.warning(f"Google score lookup failed for {match_name}: {str(e)}")
        return None

def determine_match_result(home_score, away_score):
    """Determine match result based on scores"""
    if home_score > away_score:
        return 'Home Win'
    elif away_score > home_score:
        return 'Away Win'
    else:
        return 'Draw'

def settle_bets_with_google_scores(history_df):
    """
    Auto-settle bets using Google score lookup for past matches
    """
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return history_df
    
    current_time = datetime.now(timezone.utc)
    settled_count = 0
    
    for idx, row in history_df.iterrows():
        # Skip if already settled with a valid result
        if row.get('Result', 'Pending') in ['Win', 'Loss', 'Push']:
            continue
            
        # Only process matches that are at least 2 hours old
        try:
            match_time = pd.to_datetime(row['Date_Obj'], utc=True)
            if match_time > current_time - timedelta(hours=2):
                continue  # Skip recent/active matches
        except:
            continue
        
        # Get match score from Google
        match_name = row['Match']
        score = get_google_score(match_name)
        
        if score and ' - ' in score:
            try:
                home_score_str, away_score_str = score.split(' - ')
                home_score = int(home_score_str.strip())
                away_score = int(away_score_str.strip())
                
                # Determine actual result
                actual_result = determine_match_result(home_score, away_score)
                
                # Determine if bet won
                bet_type = row['Bet']
                if (bet_type == 'Home Win' and actual_result == 'Home Win') or \
                   (bet_type == 'Away Win' and actual_result == 'Away Win') or \
                   (bet_type == 'Draw' and actual_result == 'Draw'):
                    history_df.at[idx, 'Result'] = 'Win'
                    history_df.at[idx, 'Profit'] = row['Stake'] * (row['Odds'] - 1)
                else:
                    history_df.at[idx, 'Result'] = 'Loss'
                    history_df.at[idx, 'Profit'] = -row['Stake']
                
                history_df.at[idx, 'Score'] = score
                settled_count += 1
                
                logger.info(f"Settled bet {match_name} as {actual_result} with score {score}")
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing score {score} for {match_name}: {str(e)}")
                # Mark as Auto-Settled if score parsing fails
                history_df.at[idx, 'Result'] = 'Auto-Settled'
                history_df.at[idx, 'Profit'] = -row.get('Stake', 0)
                history_df.at[idx, 'Score'] = 'N/A'
        else:
            # If no score found, mark as Auto-Settled after 3 days (conservative approach)
            if match_time < current_time - timedelta(days=3):
                history_df.at[idx, 'Result'] = 'Auto-Settled'
                history_df.at[idx, 'Profit'] = -row.get('Stake', 0)
                history_df.at[idx, 'Score'] = 'N/A'
    
    logger.info(f"Settled {settled_count} bets using Google score lookup")
    return history_df

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

def format_result_with_score(result, score):
    """Format result with score information for display"""
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

def is_active_bet(row):
    """Determine if a bet is for a future match (active)"""
    try:
        current_time = datetime.now(timezone.utc)
        match_date = pd.to_datetime(row['Date_Obj'], utc=True)
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
