# utils.py
# Shared functions for data loading, styling, and logic.
# v83.3 (Google Score Lookup & Settlement Engine - Enhanced Past Match Lookup)
# FIX: Added settle_bets_with_google_scores function
# FIX: Properly handles Date_Obj column creation/validation
# FIX: Updated deadline logic to use current date
# FIX: Enhanced Google score lookup for past matches

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
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "jd0913") # Use env var or default
GITHUB_REPO = os.getenv("GITHUB_REPO", "betting-copilot-pro") # Use env var or default

# Fixed URL formatting
LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"

@st.cache_data(ttl=600)
def load_data(url):
    """Safely load data from GitHub with proper schema validation and Date_Obj creation"""
    try:
        logger.info(f"Loading data from: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('Content-Type', '')
        if 'csv' not in content_type and 'text/plain' not in content_type:
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
        
        # Date Formatting with UTC handling and critical Date_Obj creation
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
        
        # Handle Score column properly to prevent 'nan' values
        if 'Score' in df.columns:
            df['Score'] = df['Score'].fillna('N/A').astype(str)
            # Replace any 'nan' string values with 'N/A'
            df['Score'] = df['Score'].apply(lambda x: 'N/A' if str(x).lower() == 'nan' else x)
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

def get_google_score(match_name, match_date_str):
    """
    Scrape Google for the final score of a match.
    Args:
        match_name (str): e.g., "Arsenal vs Chelsea"
        match_date_str (str): Date in 'YYYY-MM-DD' format
    Returns:
        str: Score in format "X - Y" or None if not found.
    """
    # Construct search query including date for accuracy
    search_query = f"{match_name} {match_date_str} final score"
    encoded_query = search_query.replace(' ', '+')
    url = f"https://www.google.com/search?q={encoded_query}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        logger.info(f"Scraping Google for score: {match_name} ({match_date_str})")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # List of potential selectors for Google sports score boxes
        # These might change over time, so we try multiple
        potential_selectors = [
            # Primary selectors (most likely)
            'div[data-hveid] div[jsname="vZSTIf"]', # General result box content
            'div[data-hveid] div[jsname="sTFXNd"]', # Another common result box
            'div[data-hveid] div[jsname="XcVN5d"]', # Another possible structure
            # Specific sports box selectors (might be nested)
            'div[data-attrid="kc:/sports/sports_team_record:score"]', # Old style?
            'div[data-attrid="kc:/sports:sport"]', # General sports section
            # Alternative: Look for specific class names often used for scores
            'div[class*="imso_mh"]', # Google's sports module classes
            'div[class*="imso_mh__scr"]', # Score-specific classes
            'div[class*="imso_mh__s-"]', # Score elements
            'span[class*="imso_mh"]', # Score spans
            'span[class*="imso_mh__t"]', # Team score spans
        ]
        
        score_text = ""
        
        # Try each selector combination
        for selector in potential_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text().strip()
                    # Look for patterns like "2 - 1", "3:0", "1-0", "2-1 FT", etc.
                    score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)(?:\s*FT|\s*F)?', text)
                    if score_match:
                        home_score = score_match.group(1)
                        away_score = score_match.group(2)
                        logger.info(f"   > Google score found with selector '{selector}': {home_score} - {away_score}")
                        return f"{home_score} - {away_score}"
                    # Also check for the text content of the element itself
                    score_text += " " + text
        
        # If no score found in structured elements, search in all divs/text content
        # This is a fallback, less reliable
        if score_text:
             # Look for patterns like "2 - 1", "3:0", "1-0", "2-1 FT", etc. in the collected text
            score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)(?:\s*FT|\s*F)?', score_text)
            if score_match:
                home_score = score_match.group(1)
                away_score = score_match.group(2)
                logger.info(f"   > Google score found in fallback text: {home_score} - {away_score}")
                return f"{home_score} - {away_score}"

        # Another fallback: Find all divs that might contain scores
        all_divs = soup.find_all('div')
        for div in all_divs:
            div_text = div.get_text().strip()
            # Look for patterns like "2 - 1", "3:0", "1-0", "2-1 FT", etc.
            score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)(?:\s*FT|\s*F)?', div_text)
            if score_match:
                home_score = score_match.group(1)
                away_score = score_match.group(2)
                logger.info(f"   > Google score found (fallback div search): {home_score} - {away_score}")
                return f"{home_score} - {away_score}"
        
        # Final fallback: Check spans as well
        all_spans = soup.find_all('span')
        for span in all_spans:
            span_text = span.get_text().strip()
            score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)(?:\s*FT|\s*F)?', span_text)
            if score_match:
                home_score = score_match.group(1)
                away_score = score_match.group(2)
                logger.info(f"   > Google score found (fallback span search): {home_score} - {away_score}")
                return f"{home_score} - {away_score}"
                
        logger.info(f"   > No score found on Google for {match_name} ({match_date_str}) after trying multiple selectors and text searches.")
        return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"   > Network error fetching score from Google for {match_name} ({match_date_str}): {e}")
        return None
    except Exception as e:
        logger.error(f"   > Unexpected error parsing Google score for {match_name} ({match_date_str}): {e}")
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
    Auto-settle bets using Google score lookup for past matches.
    CRITICAL FIX: Uses current date for settlement decisions, removing the Dec 8, 2025 deadline.
    """
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        logger.warning("History dataframe is empty or invalid for settlement.")
        return history_df

    # Ensure Date_Obj column exists (critical for the KeyError fix)
    if 'Date_Obj' not in history_df.columns:
        logger.error("CRITICAL: 'Date_Obj' column missing from history dataframe. Attempting recovery...")
        if 'Date' in history_df.columns:
            try:
                history_df['Date_Obj'] = pd.to_datetime(history_df['Date'], utc=True)
                logger.info("'Date_Obj' column created from 'Date' column.")
            except Exception as e:
                logger.error(f"Failed to create 'Date_Obj' from 'Date': {str(e)}. Settlement skipped.")
                return history_df
        else:
            logger.error("Neither 'Date_Obj' nor 'Date' column found. Cannot perform settlement.")
            return history_df

    current_time = datetime.now(timezone.utc)
    settled_count = 0
    
    # Create a copy to avoid SettingWithCopyWarning
    df = history_df.copy()
    
    for idx, row in df.iterrows():
        try:
            # Get match details - ensure Date_Obj is parsed correctly
            match_time = pd.to_datetime(row['Date_Obj'], utc=True)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse Date_Obj for row {idx}. Skipping settlement.")
            continue # Skip this row if date parsing fails
            
        match_name = row['Match']
        predicted_bet = row['Bet']
        current_result = row.get('Result', 'Pending')
        current_score = row.get('Score', 'N/A')
        
        # CRITICAL FIX: Check if the match has passed AND the result is still pending
        # This replaces the old deadline logic
        if match_time < current_time and current_result == 'Pending':
            # Fetch score from Google for matches that have passed
            score_str = get_google_score(match_name, match_time.strftime('%Y-%m-%d'))
            
            if score_str and ' - ' in score_str:
                try:
                    home_score_str, away_score_str = score_str.split(' - ')
                    home_score = int(home_score_str.strip())
                    away_score = int(away_score_str.strip())
                    
                    # Determine actual result
                    actual_result = determine_match_result(home_score, away_score)
                    
                    # Determine if bet won based on prediction vs actual result
                    bet_won = (
                        (predicted_bet == 'Home Win' and actual_result == 'Home Win') or
                        (predicted_bet == 'Away Win' and actual_result == 'Away Win') or
                        (predicted_bet == 'Draw' and actual_result == 'Draw')
                    )
                    
                    if bet_won:
                        df.at[idx, 'Result'] = 'Win'
                        df.at[idx, 'Profit'] = row['Stake'] * (row['Odds'] - 1)
                    else:
                        df.at[idx, 'Result'] = 'Loss'
                        df.at[idx, 'Profit'] = -row['Stake']
                    
                    df.at[idx, 'Score'] = score_str
                    settled_count += 1
                    logger.info(f"   > Settled bet for {match_name} as {actual_result} with score {score_str}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"   > Error processing Google score '{score_str}' for {match_name}: {str(e)}")
                    # Mark as Auto-Settled if score parsing fails
                    df.at[idx, 'Result'] = 'Auto-Settled'
                    df.at[idx, 'Profit'] = -row.get('Stake', 0)
                    df.at[idx, 'Score'] = 'N/A (Parse Error)'
                    settled_count += 1
            else:
                # If no score found for a match that has passed, mark as Auto-Settled (conservative approach)
                df.at[idx, 'Result'] = 'Auto-Settled'
                df.at[idx, 'Profit'] = -row.get('Stake', 0)
                df.at[idx, 'Score'] = 'N/A (Score Not Found)'
                settled_count += 1
                logger.info(f"   > Auto-settled (score not found) bet for {match_name} ({match_time.date()})")
    
    if settled_count > 0:
        logger.info(f"   > Google-based settlement complete. {settled_count} bets settled.")
    
    # Return the modified copy
    return df

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
        .res-auto-settled {
            color: #ff8a80; /* Same color as loss */
            font-weight: 600;
            background: rgba(183, 28, 28, 0.15); /* Slightly different bg */
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
            return '💎 ARB'
        if odds > 3.5 and edge > 0.15:
            return '⚡ HIGH'
        if conf > 0.60 and edge > 0.07:
            return '⭐ VALUE'
        if edge > 0.02:
            return 'EDGE'
        return 'STANDARD'
    except (ValueError, TypeError):
        return 'N/A'

def format_edge_text(edge):
    """Format edge percentage with proper coloring and text"""
    if edge > 0.1:
        return f"🔥 {edge:.1%}"
    elif edge > 0.05:
        return f"🎯 {edge:.1%}"
    elif edge > 0.02:
        return f"✅ {edge:.1%}"
    else:
        return f"{edge:.1%}"

def is_active_bet(row):
    """Determine if a bet is for a future match (active)"""
    try:
        current_time = datetime.now(timezone.utc)
        match_time = pd.to_datetime(row['Date_Obj'], utc=True)
        # Consider matches in last 2 hours as still active/ongoing
        return match_time > current_time - timedelta(hours=2)
    except (ValueError, TypeError):
        return True # Default to active if date parsing fails

def format_result_with_score(result, score):
    """Format result with score information for display"""
    # Handle missing or invalid scores
    if not isinstance(score, str) or score in ['N/A', 'nan', 'NaN', '']:
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
        return f'<span class="res-auto-settled">🔄 AUTO-SETTLED ({score})</span>'
    else:
        return f'<span class="res-pending">{result.upper()} ({score})</span>'

def format_currency(value):
    """Format numbers as currency with proper signs"""
    if pd.isna(value) or abs(value) < 0.01:
        return "$0.00"
    sign = "+" if value > 0 else "-"
    return f"{sign}${abs(value):,.2f}"
