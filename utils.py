# utils-2.py
# Shared functions for data loading, styling, and logic.
# v85.2 (API-ONLY Score Lookup - Fuzzy Match Fix)
# FIX: Lowered fuzzy match threshold from 90 to 80 to improve score matching.

import streamlit as st
import pandas as pd
import requests
import io
from datetime import datetime, timedelta, timezone
import numpy as np
import logging
import os
import time
from bs4 import BeautifulSoup 
import re
from fuzzywuzzy import process # For team name matching
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BettingUtils")

# --- API and Configuration ---
# WARNING: Hardcoding API keys is generally discouraged. Use environment variables/secrets for production.
ODDS_API_KEY = "0c5a163c2e9a8c4b6a5d33c56747ecf1" 
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports" 
# Fallback to GitHub defaults
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "jd0913") 
GITHUB_REPO = os.getenv("GITHUB_REPO", "betting-copilot-pro") 

# Fixed URL formatting
LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"

# --- Shared Utility Functions ---

@st.cache_data(ttl=600)
def load_data(url):
    """Safely load data from GitHub with proper schema validation and Date_Obj creation"""
    try:
        logger.info(f"Loading data from: {url}")
        
        # Use a single request and read from content (avoids double network call)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Read CSV from the string content
        df = pd.read_csv(io.StringIO(response.text))
        
        if df.empty:
            logger.info("Data loaded but DataFrame is empty.")
            return pd.DataFrame()
        
        # Ensure Date_Obj is a consistent UTC datetime
        if 'Date' in df.columns:
            df['Date_Obj'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
            df.dropna(subset=['Date_Obj'], inplace=True)
            df['Date'] = df['Date_Obj'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Ensure Score, Result, and Profit columns exist for consistency
        for col, default in zip(['Score', 'Result', 'Profit'], ['', 'Pending', 0.0]):
            if col not in df.columns:
                df[col] = default
                
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error loading data from {url}: {e}")
    except Exception as e:
        logger.error(f"Error processing data from {url}: {e}")
        
    return pd.DataFrame()

# --- API Score Lookup Function (Primary and ONLY method) ---

def api_score_lookup(match_date, team_a, team_b, sport):
    """
    Primary score lookup using The Odds API for structured, reliable data.
    Returns (score_string, result) or (None, 'Pending').
    """
    if not ODDS_API_KEY:
        logger.warning("ODDS_API_KEY not set. Cannot settle bets without it.")
        return None, "API_KEY_MISSING"

    # Map our sport names to API's sport keys (Customize this map for your API)
    # The Odds API example keys:
    sport_map = {
        'soccer': 'soccer_epl',
        'nfl': 'americanfootball_nfl',
        'nba': 'basketball_nba',
        'mlb': 'baseball_mlb',
        'nhl': 'icehockey_nhl',
        'mma': 'mma_mixed_martial_arts',
        'tennis': 'tennis_atp_us_open' 
    }
    api_sport_key = sport_map.get(sport.lower(), None)
    if not api_sport_key:
        logger.warning(f"No API key mapping for sport: {sport}. Skipping lookup.")
        return None, "SPORT_NOT_MAPPED"

    endpoint = f"{ODDS_API_BASE_URL}/{api_sport_key}/scores/"
    params = {
        'apiKey': ODDS_API_KEY,
        'daysFrom': 3, # Check for scores in the last 3 days
        'all': 'true' # Include completed matches
    }
    
    session = requests.Session()
    
    try:
        logger.info(f"Attempting API lookup for {team_a} vs {team_b} ({sport})...")
        response = session.get(endpoint, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Match the game by date and team names
        for game in data:
            if game['completed'] is True:
                teams = [game['home_team'], game['away_team']]
                
                # Use fuzzy matching for team names
                match_a = process.extractOne(team_a, teams)
                match_b = process.extractOne(team_b, teams)

                # CRITICAL FIX: Lowered confidence threshold to 80
                if match_a[1] > 80 and match_b[1] > 80:
                    
                    # Extract final scores
                    scores = game.get('scores', [])
                    
                    # Look up scores by team name
                    home_score = next((int(s['score']) for s in scores if s['name'] == game['home_team']), 0)
                    away_score = next((int(s['score']) for s in scores if s['name'] == game['away_team']), 0)
                    
                    score_str = f"{home_score}-{away_score}"
                    
                    logger.info(f"API SUCCESS: Score found for {game['home_team']} vs {game['away_team']}: {score_str}")
                    return score_str, "API_SUCCESS"
                    
        logger.info("API lookup found no completed match.")
        return None, "Pending"

    except requests.exceptions.HTTPError as e:
        logger.error(f"API HTTP Error ({e.response.status_code}). Check API Key/usage limits.")
        return None, f"API_ERROR_{e.response.status_code}"
    except requests.exceptions.RequestException as e:
        logger.error(f"API Request Failed (Network/Timeout): {e}")
        return None, "API_NETWORK_ERROR"
    except Exception as e:
        logger.error(f"Error processing API response: {e}")
        return None, "API_PARSE_ERROR"
        
    return None, "Pending"


# --- Main Settlement Function (API-Only) ---

def settle_bets_with_api_scores(df):
    """Settles all pending bets in the DataFrame using the API ONLY."""
    if df.empty:
        return df

    unsettled_df = df[df['Result'] == 'Pending'].copy()
    
    if unsettled_df.empty:
        logger.info("No pending bets to settle.")
        return df

    logger.info(f"Starting API-ONLY settlement process for {len(unsettled_df)} pending bets...")

    for index, row in tqdm(unsettled_df.iterrows(), total=len(unsettled_df), desc="Settling Bets"):
        
        # Only check matches that are past their scheduled time + 1 hour buffer
        current_time_utc = datetime.now(timezone.utc)
        match_time = row['Date_Obj']
        
        if match_time > current_time_utc + timedelta(hours=1):
            continue # Skip future matches
            
        # Ensure match has a valid format (Home vs Away)
        if ' vs ' not in row['Match']:
             logger.warning(f"Skipping match '{row['Match']}': Invalid ' vs ' format.")
             continue
             
        home_team, away_team = row['Match'].split(' vs ', 1)
        sport = row['Sport']
        
        # 1. PRIMARY: API LOOKUP
        score_str, result_status = api_score_lookup(
            match_date=match_time.date(),
            team_a=home_team, 
            team_b=away_team,
            sport=sport
        )
        
        # 2. SETTLEMENT LOGIC (Only runs if a score was found)
        if score_str and result_status == 'API_SUCCESS':
            
            try:
                # Assuming score_str is in format "HomeScore-AwayScore" (e.g., "2-1")
                home_score, away_score = map(int, score_str.split('-'))
            except ValueError:
                logger.error(f"Invalid score format '{score_str}' for {row['Match']}. Skipping settlement.")
                continue

            # --- Score-based Settlement Logic (Example for Moneyline/Total Goals) ---
            bet_type = row['Bet_Type']
            
            # Simple settlement logic (customize this part heavily for different bet types)
            if 'Moneyline' in bet_type:
                # Basic Moneyline 
                winning_team = home_team if home_score > away_score else away_team if away_score > home_score else 'Draw'
                bet_on = row['Bet'].replace(' Moneyline', '') # Get the team bet on
                
                if winning_team == bet_on:
                    result = 'Win'
                    profit = row['Stake'] * (row['Odds'] - 1)
                elif winning_team == 'Draw':
                    result = 'Push'
                    profit = 0.0
                else:
                    result = 'Loss'
                    profit = -row['Stake']
                    
            elif 'Total Goals' in bet_type or 'Total Points' in bet_type:
                # Basic Total Goals/Points (e.g., "Total Goals Over 2.5")
                total = home_score + away_score
                
                if 'Over' in row['Bet']:
                    target = float(row['Bet'].split(' Over ')[1])
                    if total > target:
                        result = 'Win'
                        profit = row['Stake'] * (row['Odds'] - 1)
                    elif total == target:
                        result = 'Push'
                        profit = 0.0
                    else:
                        result = 'Loss'
                        profit = -row['Stake']
                elif 'Under' in row['Bet']:
                    target = float(row['Bet'].split(' Under ')[1])
                    if total < target:
                        result = 'Win'
                        profit = row['Stake'] * (row['Odds'] - 1)
                    elif total == target:
                        result = 'Push'
                        profit = 0.0
                    else:
                        result = 'Loss'
                        profit = -row['Stake']
                else:
                    result = 'Loss'
                    profit = -row['Stake'] 

            # 3. UPDATE DATAFRAME
            df.loc[index, 'Score'] = score_str
            df.loc[index, 'Result'] = result
            df.loc[index, 'Profit'] = profit

            logger.info(f"✅ Settled {row['Match']} ({result}): {score_str}. Profit: ${profit:.2f}")
        
        elif result_status != 'Pending':
            # Log API error for visibility without changing the bet status
            logger.warning(f"⚠️ API Status for {row['Match']}: {result_status}. Score not settled.")

    return df

# --- Existing Utility Functions (Below) ---

def inject_custom_css():
    """Injects custom CSS for better styling in Streamlit."""
    st.markdown("""
        <style>
        .gradient-text {
            font-size: 2.5em;
            font-weight: bold;
            background: -webkit-linear-gradient(45deg, #FF5733, #FFC300);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        /* Results Styling */
        .res-win { color: #38c172; font-weight: bold; } /* Tailwind Green */
        .res-loss { color: #e3342f; font-weight: bold; } /* Tailwind Red */
        .res-push { color: #ffed4a; font-weight: bold; } /* Tailwind Yellow */
        .res-pending { color: #6cb2eb; font-weight: bold; } /* Tailwind Blue */
        .res-auto-settled { color: #4dc0b5; font-weight: bold; } /* Tailwind Teal */
        </style>
        """, unsafe_allow_html=True)

def is_active_bet(row):
    """Determine if a bet is for a future match (active) or recently completed (last 2 hours)"""
    try:
        current_time = datetime.now(timezone.utc)
        match_time = pd.to_datetime(row['Date_Obj'], utc=True)
        # Consider matches in last 2 hours as still active/ongoing
        return match_time > current_time - timedelta(hours=2)
    except (ValueError, TypeError):
        return True # Default to active if date parsing fails

def get_active_bets(latest_df):
    """Get only active/future bets from latest recommendations"""
    if not isinstance(latest_df, pd.DataFrame) or latest_df.empty:
        return pd.DataFrame()
    
    # Filter for active/future bets
    active_mask = latest_df.apply(is_active_bet, axis=1)
    active_bets = latest_df[active_mask].copy()
    
    # Sort by date
    if 'Date_Obj' in active_bets.columns:
        active_bets = active_bets.sort_values(by='Date_Obj', ascending=True)
        
    return active_bets

def format_result_with_score(result, score):
    """Format result with score information for display"""
    # Handle missing or invalid scores
    if not isinstance(score, str) or score in ['N/A', 'nan', 'NaN', ''] or not result:
        if result and result in ['Win', 'Loss', 'Push', 'Auto-Settled']:
            # Fallback for settled bets without a score string (shouldn't happen with API)
            return f'<span class=\"res-{result.lower()}\">{result.upper()}</span>'
        return '<span class=\"res-pending\">⏳ PENDING</span>'
    
    # Format based on result
    if result == 'Win':
        return f'<span class=\"res-win\">✅ WIN ({score})</span>'
    elif result == 'Loss':
        return f'<span class=\"res-loss\">❌ LOSS ({score})</span>'
    elif result == 'Push':
        return f'<span class=\"res-push\">⚖️ PUSH ({score})</span>'
    elif result == 'Auto-Settled':
        return f'<span class=\"res-auto-settled\">🔄 AUTO-SETTLED ({score})</span>'
    else:
        return f'<span class=\"res-pending\">{result.upper()} ({score})</span>'

def format_currency(value):
    """Format numbers as currency with proper signs"""
    try:
        if pd.isna(value):
            return "$0.00"
        return f"${value:,.2f}"
    except (ValueError, TypeError):
        return "$0.00"

def format_odds(value):
    """Format odds to two decimal places."""
    try:
        if pd.isna(value):
            return "N/A"
        return f"{value:.2f}"
    except (ValueError, TypeError):
        return "N/A"
