# utils.py
# Shared functions for data loading, styling, and logic.
# v70.0 (Final Settlement & Score Display Fix)
# FIX: Now correctly determines wins/losses using actual scores
# FIX: Shows final scores prominently in history table
# FIX: Properly displays settled results instead of "Pending"

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import numpy as np
import logging
import os
import config
from fuzzywuzzy import process

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BettingUtils")

# --- CONFIGURATION (SECURE METHOD) ---
# Get GitHub credentials from Streamlit Secrets or config
GITHUB_USERNAME = st.secrets.get("github_username", config.GITHUB_USERNAME if hasattr(config, 'GITHUB_USERNAME') else "jd0913")
GITHUB_REPO = st.secrets.get("github_repo", config.GITHUB_REPO if hasattr(config, 'GITHUB_REPO') else "betting-copilot-pro")

# Fixed URL formatting (removed extra spaces)
LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"
SCORES_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/match_scores.csv"

@st.cache_data(ttl=600)
def load_data(url):
    """Safely load data from GitHub with proper error handling"""
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

def get_actual_scores():
    """Get actual scores from a dedicated scores file"""
    try:
        # Try to load scores data
        scores_df = load_data(SCORES_URL)
        if not isinstance(scores_df, pd.DataFrame) or scores_df.empty:
            logger.warning("No scores data available")
            return pd.DataFrame()
        
        # Ensure proper date formatting
        if 'Date' in scores_df.columns:
            scores_df['Date_Obj'] = pd.to_datetime(scores_df['Date'], errors='coerce', utc=True)
        
        return scores_df
    except Exception as e:
        logger.error(f"Error loading scores: {str(e)}")
        return pd.DataFrame()

def determine_match_result(home_score, away_score):
    """Determine match result based on scores"""
    if home_score > away_score:
        return 'Home Win'
    elif away_score > home_score:
        return 'Away Win'
    else:
        return 'Draw'

def settle_bets_with_scores(history_df):
    """Auto-settle bets using actual scores when available"""
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return history_df
    
    # Get actual scores data
    scores_df = get_actual_scores()
    
    current_time = datetime.now(timezone.utc)
    settled_count = 0
    corrected_count = 0
    
    for idx, row in history_df.iterrows():
        # Skip if already settled with a proper score
        if row.get('Result') in ['Win', 'Loss', 'Push'] and row.get('Score') and row.get('Score') != 'N/A' and row.get('Score') != '':
            continue
            
        # Get match details
        try:
            match_date = pd.to_datetime(row['Date_Obj'], utc=True)
        except (ValueError, TypeError):
            continue
            
        match_name = row['Match']
        predicted_bet = row['Bet']
        actual_score = row.get('Score', '')
        
        # Settle past matches (2+ hours old)
        if match_date < current_time - timedelta(hours=2):
            # If we have scores data, look up the actual score
            if not scores_df.empty and 'Match' in scores_df.columns and 'HomeScore' in scores_df.columns and 'AwayScore' in scores_df.columns:
                # Find the matching game using fuzzy matching
                home_team, away_team = match_name.split(' vs ')
                matching_score = None
                
                # Try exact match first
                exact_match = scores_df[scores_df['Match'] == match_name]
                if not exact_match.empty:
                    matching_score = exact_match.iloc[0]
                else:
                    # Try fuzzy match
                    match_scores = []
                    for _, score_row in scores_df.iterrows():
                        similarity = process.extractOne(match_name, [score_row['Match']])[1]
                        if similarity > 85:  # 85% similarity threshold
                            match_scores.append((score_row, similarity))
                    
                    if match_scores:
                        # Get the best match
                        matching_score = sorted(match_scores, key=lambda x: x[1], reverse=True)[0][0]
                
                if matching_score is not None:
                    home_score = int(matching_score['HomeScore'])
                    away_score = int(matching_score['AwayScore'])
                    actual_score = f"{home_score} - {away_score}"
                    
                    # Determine actual outcome
                    actual_result = determine_match_result(home_score, away_score)
                    
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
                    settled_count += 1
                    corrected_count += 1
                    continue
            
            # If score is available in the history row itself, use it
            if actual_score and actual_score != 'N/A' and actual_score != '' and ' - ' in actual_score:
                try:
                    home_score_str, away_score_str = actual_score.split(' - ')
                    home_score = int(home_score_str.strip())
                    away_score = int(away_score_str.strip())
                    
                    # Determine actual outcome
                    actual_result = determine_match_result(home_score, away_score)
                    
                    # Determine if bet won
                    if (predicted_bet == 'Home Win' and actual_result == 'Home Win') or \
                       (predicted_bet == 'Away Win' and actual_result == 'Away Win') or \
                       (predicted_bet == 'Draw' and actual_result == 'Draw'):
                        history_df.at[idx, 'Result'] = 'Win'
                        history_df.at[idx, 'Profit'] = row['Stake'] * (row['Odds'] - 1)
                    else:
                        history_df.at[idx, 'Result'] = 'Loss'
                        history_df.at[idx, 'Profit'] = -row['Stake']
                    
                    settled_count += 1
                    corrected_count += 1
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing score for {match_name}: {str(e)}")
            
            # If no score but match is old, mark as Auto-Settled
            elif row.get('Result', 'Pending') == 'Pending':
                history_df.at[idx, 'Result'] = 'Auto-Settled'
                history_df.at[idx, 'Profit'] = -row['Stake']  # Conservative: assume loss
                history_df.at[idx, 'Score'] = 'N/A'  # No score available
                settled_count += 1
    
    if settled_count > 0:
        logger.info(f"Auto-settled {settled_count} past bets")
    if corrected_count > 0:
        logger.info(f"Corrected {corrected_count} results using actual scores")
    
    return history_df

def format_result_with_score(result, score):
    """Format result with score information for display"""
    # Handle missing or invalid scores
    if not score or score == 'N/A' or score == '' or result == 'Pending':
        if result == 'Win':
            return "✅ WIN (Score Pending)"
        elif result == 'Loss':
            return "❌ LOSS (Score Pending)"
        return "⏳ PENDING"
    
    # Format based on result
    if result == 'Win':
        return f"✅ WIN ({score})"
    elif result == 'Loss':
        return f"❌ LOSS ({score})"
    elif result == 'Push':
        return f"⚖️ PUSH ({score})"
    elif result == 'Auto-Settled':
        return f"🔄 AUTO-SETTLED ({score})"
    else:
        return f"{result} ({score})"

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
        
        /* Expanders */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 10px;
            font-weight: 600;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 35px;
            white-space: pre-wrap;
            background-color: #1e2130;
            border-radius: 8px 8px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #00C9FF;
            color: #000;
        }
        
        /* Form elements */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background-color: #16181d;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
        }
        
        /* Selectbox */
        .stSelectbox > div > div > div {
            background-color: #16181d;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
        }
        
        /* Expander content */
        .streamlit-expanderContent {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 0 0 8px 8px;
            padding: 15px;
        }
        
        /* Final score column styling */
        .final-score {
            font-weight: bold;
            color: #00C9FF;
        }
        .score-pending {
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
