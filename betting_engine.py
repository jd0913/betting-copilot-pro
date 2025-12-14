# betting_engine.py
# The Core Logic: AI Models, Google Data Fetching, Settlement, Feature Engineering
# v72.1 (Google-Only Odds & Score Lookup)
# FIX: Added Google live odds scraping
# FIX: Properly settles all bets before Dec 8, 2025
# FIX: Shows active/future bets in Command Center

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_val_predict
from scipy.stats import poisson
import joblib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta, timezone
import os
import json
import random
from fuzzywuzzy import process
from textblob import TextBlob
import logging
import re # Added for parsing
import config # We'll handle the missing config later

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BettingEngine")

# --- SECURE API SESSION (With retries and timeouts) ---
def create_api_session():
    """Create resilient session with retry logic"""
    session = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    return session

API_SESSION = create_api_session()

# --- CONFIGURATION (Removed import config, using environment variables) ---
# Get GitHub credentials from environment or use defaults
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "jd0913") # Replace with your username
GITHUB_REPO = os.getenv("GITHUB_REPO", "betting-copilot-pro") # Replace with your repo name

# Fixed URL formatting
LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"

# --- CORE MATH MODULES (Preserved) ---
def calculate_pythagorean_expectation(points_for, points_against, exponent=1.83):
    if points_for == 0 and points_against == 0: return 0.5
    if points_for < 0 or points_against < 0: return 0.5
    numerator = points_for ** exponent
    denominator = numerator + (points_against ** exponent)
    return numerator / denominator if denominator > 0 else 0.5

def zero_inflated_poisson(k, lam, pi=0.05):
    if k == 0:
        return pi + (1 - pi) * poisson.pmf(0, lam)
    else:
        return (1 - pi) * poisson.pmf(k, lam)

# --- GENETIC AI & MODELS (Preserved) ---
GENOME_FILE = 'model_genome.json'

def load_or_initialize_genome():
    if os.path.exists(GENOME_FILE):
        with open(GENOME_FILE, 'r') as f: return json.load(f)
    return {'generation': 0, 'best_score': 10.0, 'xgb_n_estimators': 200, 'xgb_max_depth': 3, 'xgb_learning_rate': 0.1, 'rf_n_estimators': 200, 'rf_max_depth': 10, 'nn_hidden_layer_size': 64, 'nn_alpha': 0.0001}

def mutate_genome(genome):
    mutant = genome.copy(); mutation_rate = 0.3
    if random.random() < mutation_rate: mutant['xgb_n_estimators'] = int(genome['xgb_n_estimators'] * random.uniform(0.8, 1.2))
    if random.random() < mutation_rate: mutant['xgb_learning_rate'] = genome['xgb_learning_rate'] * random.uniform(0.8, 1.2)
    if random.random() < mutation_rate: mutant['rf_n_estimators'] = int(genome['rf_n_estimators'] * random.uniform(0.8, 1.2))
    if random.random() < mutation_rate: mutant['nn_hidden_layer_size'] = int(genome['nn_hidden_layer_size'] * random.uniform(0.8, 1.2))
    return mutant

def build_ensemble_from_genome(genome):
    # Validate parameters before building
    n_est_xgb = max(50, min(500, int(genome.get('xgb_n_estimators', 200))))
    max_depth_xgb = max(2, min(10, int(genome.get('xgb_max_depth', 3))))
    lr_xgb = max(0.01, min(0.3, float(genome.get('xgb_learning_rate', 0.1))))
    
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False,
        n_estimators=n_est_xgb, max_depth=max_depth_xgb, learning_rate=lr_xgb,
        random_state=42, n_jobs=-1, tree_method='hist'
    )
    rf_clf = RandomForestClassifier(
        n_estimators=max(50, min(300, int(genome.get('rf_n_estimators', 200)))),
        max_depth=max(3, min(15, int(genome.get('rf_max_depth', 10)))),
        random_state=42, n_jobs=-1
    )
    nn_clf = MLPClassifier(
        hidden_layer_sizes=(max(32, min(128, int(genome.get('nn_hidden_layer_size', 64)))), max(16, min(64, int(genome.get('nn_hidden_layer_size', 64)) // 2)),
        alpha=max(1e-5, min(0.1, float(genome.get('nn_alpha', 0.0001)))),
        activation='relu', solver='adam', max_iter=500, random_state=42,
        early_stopping=True
    )
    lr_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    return VotingClassifier(estimators=[('xgb', xgb_clf), ('rf', rf_clf), ('nn', nn_clf), ('lr', lr_clf)], voting='soft', n_jobs=-1)

def train_meta_model(X, y, primary_model):
    try:
        preds = cross_val_predict(primary_model, X, y, cv=3)
        meta_y = (preds == y).astype(int)
        meta_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        meta_clf.fit(X, meta_y)
        return meta_clf
    except Exception as e:
        print(f"Error training meta-model: {e}")
        return None

def evolve_and_train(X, y):
    print("   🧬 Initiating Evolutionary Cycle...")
    current_genome = load_or_initialize_genome(); mutant_genome = mutate_genome(current_genome); tscv = TimeSeriesSplit(n_splits=3)
    champ_model = build_ensemble_from_genome(current_genome); champ_scores = cross_val_score(champ_model, X, y, cv=tscv, scoring='neg_log_loss'); champ_fitness = -champ_scores.mean()
    mutant_model = build_ensemble_from_genome(mutant_genome); mutant_scores = cross_val_score(mutant_model, X, y, cv=tscv, scoring='neg_log_loss'); mutant_fitness = -mutant_scores.mean()
    
    if mutant_fitness < champ_fitness:
        print("      > 🚀 EVOLUTION! Mutant wins.")
        mutant_genome['best_score'] = mutant_fitness; mutant_genome['generation'] = current_genome['generation'] + 1; winner_genome = mutant_genome
        with open(GENOME_FILE, 'w') as f: json.dump(winner_genome, f)
    else:
        print("      > 💀 Champion remains.")
        winner_genome = current_genome
        
    final_model = build_ensemble_from_genome(winner_genome)
    meta_model = train_meta_model(X, y, final_model)
    calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=3)
    calibrated_model.fit(X, y)
    return calibrated_model, meta_model

# --- GOOGLE DATA FETCHING & PROCESSING (NEW CORE FUNCTIONALITY) ---
def get_live_odds_from_google(sport_key):
    """
    Scrape Google for live odds for upcoming matches in a specific sport.
    This is a complex task as Google doesn't provide structured odds data.
    We'll search for upcoming matches and try to extract any displayed odds.
    """
    # Map sport keys to search-friendly terms
    sport_search_terms = {
        'soccer_epl': 'Premier League',
        'soccer_spain_la_liga': 'La Liga',
        'soccer_germany_bundesliga': 'Bundesliga',
        'soccer_italy_serie_a': 'Serie A',
        'soccer_france_ligue_one': 'Ligue 1',
        'soccer_uefa_champs_league': 'Champions League',
        'americanfootball_nfl': 'NFL',
        'basketball_nba': 'NBA',
        'baseball_mlb': 'MLB'
    }
    
    search_term = sport_search_terms.get(sport_key, sport_key.replace('_', ' ').title())
    query = f"{search_term} live odds today"
    encoded_query = query.replace(' ', '+')
    url = f"https://www.google.com/search?q={encoded_query}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        logger.info(f"Scraping Google for live odds: {sport_key}")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for sports-specific elements that might contain odds
        # Google's layout changes frequently, so we need to be flexible
        odds_data = []
        
        # Example: Look for elements with common class names for sports data
        # These selectors are hypothetical and might need frequent updates
        # Common Google sports classes for odds:
        # imso_mh__scr-sep (score separator)
        # imso_mh__s-t (score text)
        # imso_mh__s-m (score minute)
        # imso_mh__s-sc (score section)
        # imso_mh__t1-s (team 1 score)
        # imso_mh__t2-s (team 2 score)
        # imso_mh__b-ol (odds list) - Less common, might be for specific odds
        
        # This is a very basic and potentially fragile approach
        # Odds might appear in text near match names, like "Team A vs Team B 2.10 | 3.50 | 2.80"
        # Or in specific containers if Google displays them
        
        # Look for any text containing decimal numbers that could be odds
        # Pattern: 1.x to 15.x (typical odds range)
        text_content = soup.get_text()
        # Find sequences like "Team A vs Team B" followed by potential odds
        match_odds_pattern = r'([A-Za-z\s]+)\s+vs\s+([A-Za-z\s]+).*?(\d+\.\d{2})\s*\|\s*(\d+\.\d{2})\s*\|\s*(\d+\.\d{2})'
        matches = re.finditer(match_odds_pattern, text_content, re.IGNORECASE)
        
        for match in matches:
            home_team = match.group(1).strip()
            away_team = match.group(2).strip()
            home_odds = float(match.group(3))
            draw_odds = float(match.group(4))
            away_odds = float(match.group(5))
            
            # Calculate implied probabilities
            implied_prob_h = 1 / home_odds
            implied_prob_d = 1 / draw_odds
            implied_prob_a = 1 / away_odds
            implied_prob_total = implied_prob_h + implied_prob_d + implied_prob_a
            
            # Only add if odds seem valid (implied prob < 1.2 to account for bookmaker margin)
            if implied_prob_total < 1.2:
                odds_data.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'bookmakers': [
                        {
                            'title': 'Google Scraper',
                            'markets': [
                                {
                                    'key': 'h2h',
                                    'outcomes': [
                                        {'name': home_team, 'price': home_odds},
                                        {'name': 'Draw', 'price': draw_odds},
                                        {'name': away_team, 'price': away_odds}
                                    ]
                                }
                            ]
                        }
                    ],
                    'commence_time': datetime.now(timezone.utc).isoformat() # Use current time as approximation
                })
                logger.info(f"   > Found odds via Google: {home_team} vs {away_team} - {home_odds:.2f}, {draw_odds:.2f}, {away_odds:.2f}")
        
        # Alternative: Look for specific Google sports containers if available
        # This requires inspecting Google's current HTML structure for the specific query
        # For now, we'll return what we found from the text search
        return odds_data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error scraping live odds for {sport_key}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error parsing Google odds for {sport_key}: {str(e)}")
        return []

def find_arbitrage_from_google(game, sport_type):
    """Find arbitrage using Google-scraped odds"""
    # This function is complex to implement reliably with Google scraping
    # because odds might come from different sources on the page.
    # For now, we'll just extract the best odds found for each outcome.
    best_home = {'price': 0, 'book': 'Google'}
    best_away = {'price': 0, 'book': 'Google'}
    best_draw = {'price': 0, 'book': 'Google'}
    
    for bookmaker in game['bookmakers']:
        for market in bookmaker['markets']:
            if market['key'] == 'h2h':
                for outcome in market['outcomes']:
                    price = outcome['price']
                    name = outcome['name']
                    if name == game['home_team'] and price > best_home['price']:
                        best_home = {'price': price, 'book': bookmaker['title']}
                    elif name == game['away_team'] and price > best_away['price']:
                        best_away = {'price': price, 'book': bookmaker['title']}
                    elif name == 'Draw' and price > best_draw['price']:
                        best_draw = {'price': price, 'book': bookmaker['title']}
    
    implied_prob = 0; arb_info = None
    if sport_type == 'Soccer':
        if best_home['price'] > 0 and best_away['price'] > 0 and best_draw['price'] > 0:
            implied_prob = (1/best_home['price']) + (1/best_away['price']) + (1/best_draw['price'])
            if implied_prob < 0.99: # 1% margin for error/arbitrage
                arb_info = f"Home: {best_home['book']} ({best_home['price']:.2f}) | Draw: {best_draw['book']} ({best_draw['price']:.2f}) | Away: {best_away['book']} ({best_away['price']:.2f})"
    else: # Other sports have 2 outcomes
        if best_home['price'] > 0 and best_away['price'] > 0:
            implied_prob = (1/best_home['price']) + (1/best_away['price'])
            if implied_prob < 0.99:
                arb_info = f"Home: {best_home['book']} ({best_home['price']:.2f}) | Away: {best_away['book']} ({best_away['price']:.2f})"
    
    if arb_info: return (1 - implied_prob) / implied_prob, arb_info, best_home, best_draw, best_away
    return 0, None, best_home, best_draw, best_away

def fuzzy_match_team(team_name, team_list):
    if not team_list: return None
    match, score = process.extractOne(team_name, team_list)
    if score >= 80: return match
    return None

def get_news_alert(team1, team2):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        query = f'"{team1}" OR "{team2}" injury OR doubt OR out'
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=nws"
        res = requests.get(url, headers=headers)
        res.raise_for_status() 
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.find_all('div', {'role': 'heading'})
        for h in headlines[:3]:
            if any(keyword in h.text.lower() for keyword in ['injury', 'doubt', 'out', 'miss', 'sidelined']):
                return f"⚠️ News: {h.text}"
    except requests.exceptions.RequestException as e: 
        print(f"Error scraping news for {team1}/{team2}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during news parsing: {e}")
        return None
    return None

# --- GOOGLE SCORE SETTLEMENT (Updated to handle Dec 8 deadline) ---
def settle_bets():
    """Auto-settle bets using Google score lookup for past matches"""
    print("--- ⚖️ Running Settlement Engine (Google-based) ---")
    history_file = 'betting_history.csv'
    if not os.path.exists(history_file): 
        print("No history file found for settlement.")
        return
    
    try:
        df = pd.read_csv(history_file, parse_dates=['Date'])
    except Exception as e:
        print(f"Failed to read history file: {str(e)}")
        return
    
    # Initialize columns if missing
    for col in ['Result', 'Profit', 'Score']:
        if col not in df.columns:
            df[col] = 'Pending' if col == 'Result' else 0.0 if col == 'Profit' else ''
    
    # Set deadline for forced settlement
    settlement_deadline = pd.to_datetime("2025-12-08 22:00:00", utc=True) # Mon Dec 8, 2025 @ 10pm UTC
    current_time = datetime.now(timezone.utc)
    settled_count = 0
    
    for idx, row in df.iterrows():
        try:
            match_time = pd.to_datetime(row['Date'], utc=True)
        except (ValueError, TypeError):
            continue # Skip if date parsing fails
            
        match_name = row['Match']
        predicted_bet = row['Bet']
        current_result = row.get('Result', 'Pending')
        current_score = row.get('Score', 'N/A')
        
        # Skip if already properly settled
        if current_result in ['Win', 'Loss', 'Push'] and current_score != 'N/A' and current_score != 'nan':
            continue
            
        # --- FIX: Force settlement for matches before Dec 8, 2025 @ 10pm ---
        if match_time <= settlement_deadline:
            # Get score from Google
            home_score, away_score = get_score_from_google(match_name, match_time.strftime('%Y-%m-%d'))
            
            if home_score is not None and away_score is not None:
                # Determine actual result
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
                    df.at[idx, 'Result'] = 'Win'
                    df.at[idx, 'Profit'] = row['Stake'] * (row['Odds'] - 1)
                else:
                    df.at[idx, 'Result'] = 'Loss'
                    df.at[idx, 'Profit'] = -row['Stake']
                
                df.at[idx, 'Score'] = f"{home_score} - {away_score}"
                settled_count += 1
                logger.info(f"   > Settled bet for {match_name} as {actual_result} with score {home_score}-{away_score}")
            else:
                # If score not found, mark as Auto-Settled (assume loss) to prevent perpetual 'Pending'
                df.at[idx, 'Result'] = 'Auto-Settled'
                df.at[idx, 'Profit'] = -row.get('Stake', 0)
                df.at[idx, 'Score'] = 'N/A (Score Not Found)'
                settled_count += 1
                logger.info(f"   > Auto-settled (score not found) bet for {match_name} ({match_time.date()})")
        
        # For matches *after* the deadline, only settle if they are old enough to have results
        elif match_time < current_time - timedelta(hours=2): # At least 2 hours old
            # Get score from Google
            home_score, away_score = get_score_from_google(match_name, match_time.strftime('%Y-%m-%d'))
            
            if home_score is not None and away_score is not None:
                # Determine actual result
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
                    df.at[idx, 'Result'] = 'Win'
                    df.at[idx, 'Profit'] = row['Stake'] * (row['Odds'] - 1)
                else:
                    df.at[idx, 'Result'] = 'Loss'
                    df.at[idx, 'Profit'] = -row['Stake']
                
                df.at[idx, 'Score'] = f"{home_score} - {away_score}"
                settled_count += 1
                logger.info(f"   > Settled bet for {match_name} as {actual_result} with score {home_score}-{away_score}")
            else:
                # If score not found after 3 days, mark as Auto-Settled
                if match_time < current_time - timedelta(days=3):
                    df.at[idx, 'Result'] = 'Auto-Settled'
                    df.at[idx, 'Profit'] = -row.get('Stake', 0)
                    df.at[idx, 'Score'] = 'N/A (Score Not Found)'
                    settled_count += 1
                    logger.info(f"   > Auto-settled (score not found) bet for {match_name} ({match_time.date()})")
            
    # Save the updated history
    try:
        df.to_csv(history_file, index=False)
        logger.info(f"Settlement complete. {settled_count} bets settled using Google score lookup.")
    except Exception as e:
        logger.error(f"Failed to save settled history: {str(e)}")

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
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        logger.info(f"Scraping Google for score: {match_name} ({match_date_str})")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for Google's sports score box or similar structured element
        score_elements = [
            soup.find('div', class_='imso_mh__scr-sep'),
            soup.find('div', class_='imso_mh__s-t'),
            soup.find('div', class_='imso_mh__s-m'),
            soup.find('div', class_='imso_mh__s-sc'),
            soup.find('span', class_='imso_mh__t1-s'),
            soup.find('span', class_='imso_mh__t2-s'),
        ]
        
        for element in score_elements:
            if element:
                score_text = element.get_text().strip()
                # Look for patterns like "2 - 1", "3:0", "1-0"
                score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)', score_text)
                if score_match:
                    home_score = int(score_match.group(1))
                    away_score = int(score_match.group(2))
                    logger.info(f"   > Google score found: {home_score} - {away_score}")
                    return home_score, away_score
        
        # If not found in structured elements, search in all divs
        all_divs = soup.find_all('div')
        for div in all_divs:
            text = div.get_text().strip()
            if ' - ' in text:
                score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)', text)
                if score_match:
                    home_score = int(score_match.group(1))
                    away_score = int(score_match.group(2))
                    logger.info(f"   > Google score found (fallback): {home_score} - {away_score}")
                    return home_score, away_score
        
        logger.info(f"   > No score found on Google for {match_name} ({match_date_str})")
        return None, None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"   > Network error fetching score from Google for {match_name} ({match_date_str}): {e}")
        return None, None
    except Exception as e:
        logger.error(f"   > Unexpected error parsing Google score for {match_name} ({match_date_str}): {e}")
        return None, None

# --- FEATURE ENGINEERING (Preserved) ---
def calculate_soccer_features(df):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique(); elo_ratings = {team: 1500 for team in teams}; k_factor = 20; home_elos, away_elos = [], []; team_variance = {team: [] for team in teams}
    team_goals_for = {team: 0 for team in teams}
    team_goals_against = {team: 0 for team in teams}
    home_pyth, away_pyth = [], []

    for i, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        
        # Elo
        r_h, r_a = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500)
        home_elos.append(r_h); away_elos.append(r_a)
        e_h = 1 / (1 + 10**((r_a - r_h) / 400)); s_h = 1 if row['FTR'] == 'H' else 0 if row['FTR'] == 'A' else 0.5
        error = (s_h - e_h)**2; team_variance[h].append(error); team_variance[a].append(error)
        elo_ratings[h] += k_factor * (s_h - e_h); elo_ratings[a] += k_factor * ((1-s_h) - (1-e_h))
        
        # Pythagorean
        h_py = calculate_pythagorean_expectation(team_goals_for[h], team_goals_against[h])
        a_py = calculate_pythagorean_expectation(team_goals_for[a], team_goals_against[a])
        home_pyth.append(h_py); away_pyth.append(a_py)
        
        # Update Goals
        team_goals_for[h] += row['FTHG']; team_goals_against[h] += row['FTAG']
        team_goals_for[a] += row['FTAG']; team_goals_against[a] += row['FTHG']

    df['HomeElo'], df['AwayElo'] = home_elos, away_elos
    df['HomePyth'], df['AwayPyth'] = home_pyth, away_pyth
    volatility_map = {t: np.std(v[-10:]) if len(v) > 10 else 0.25 for t, v in team_variance.items()}
    
    return df, elo_ratings, volatility_map, team_goals_for, team_goals_against

def train_league_brain(div_code):
    seasons = ['2324', '2223', '2122']; 
    try: 
        df = pd.concat([pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{s}/{div_code}.csv', parse_dates=['Date'], dayfirst=True, on_bad_lines='skip', encoding='latin1') for s in seasons]).sort_values('Date').reset_index(drop=True)
    except requests.exceptions.RequestException as e: 
        print(f"Error fetching historical data for league {div_code}: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during data loading for {div_code}: {e}")
        return None, None
        
    if df.empty: return None, None
    
    df, elo_ratings, volatility_map, gf, ga = calculate_soccer_features(df)
    
    h_stats = df[['Date', 'HomeTeam', 'FTR']].rename(columns={'HomeTeam': 'Team'}); h_stats['Points'] = h_stats['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    a_stats = df[['Date', 'AwayTeam', 'FTR']].rename(columns={'AwayTeam': 'Team'}); a_stats['Points'] = a_stats['FTR'].map({'A': 3, 'D': 1, 'H': 0})
    all_stats = pd.concat([h_stats, a_stats]).sort_values('Date'); all_stats['Form_EWMA'] = all_stats.groupby('Team')['Points'].transform(lambda x: x.ewm(span=5, adjust=False).mean().shift(1))
    df = pd.merge(df, all_stats[['Date', 'Team', 'Form_EWMA']], left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Form_EWMA': 'HomeForm'})
    df = pd.merge(df, all_stats[['Date', 'Team', 'Form_EWMA']], left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Form_EWMA': 'AwayForm'})
    
    df.dropna(subset=['B365H', 'B365D', 'B365A', 'HomeForm', 'AwayForm'], inplace=True)
    df['elo_diff'] = df['HomeElo'] - df['AwayElo']
    df['form_diff'] = df['HomeForm'] - df['AwayForm']
    df['pyth_diff'] = df['HomePyth'] - df['AwayPyth']
    
    features = ['elo_diff', 'form_diff', 'pyth_diff']
    X, y = df[features], df['FTR']
    le = LabelEncoder(); y_encoded = le.fit_transform(y); scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    
    living_model, meta_model = evolve_and_train(X_scaled, y_encoded)
    
    avg_goals_home = df['FTHG'].mean(); avg_goals_away = df['FTAG'].mean()
    home_strength = df.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'}).rename(columns={'FTHG': 'h_gf_avg', 'FTAG': 'h_ga_avg'})
    away_strength = df.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'}).rename(columns={'FTAG': 'a_gf_avg', 'FTHG': 'a_ga_avg'})
    team_strengths = pd.concat([home_strength, away_strength], axis=1).fillna(1)
    team_strengths['attack'] = (team_strengths['h_gf_avg'] / avg_goals_home + team_strengths['a_gf_avg'] / avg_goals_away) / 2
    team_strengths['defence'] = (team_strengths['h_ga_avg'] / avg_goals_away + team_strengths['a_ga_avg'] / avg_goals_home) / 2
    
    return {'model': living_model, 'meta_model': meta_model, 'le': le, 'scaler': scaler, 'elo_ratings': elo_ratings, 'volatility': volatility_map, 'team_strengths': team_strengths, 'avgs': (avg_goals_home, avg_goals_away), 'gf': gf, 'ga': ga}, df

# --- UPDATED SPORT MODULES (Using Google Odds Scraping) ---
def run_global_soccer_module():
    print("--- Running Global Soccer Module (Google Odds & Scores) ---")
    bets = []
    LEAGUE_MAP = {'soccer_epl': 'E0', 'soccer_spain_la_liga': 'SP1', 'soccer_germany_bundesliga': 'D1', 'soccer_italy_serie_a': 'I1', 'soccer_france_ligue_one': 'F1', 'soccer_uefa_champs_league': 'UCL'}
    for sport_key, div_code in LEAGUE_MAP.items():
        print(f"   > Scanning {sport_key} (Google)...") # Changed log message
        
        # NEW: Fetch live odds from Google
        odds_data = get_live_odds_from_google(sport_key)
        
        brain = None; historical_df = None
        if div_code != 'UCL': brain, historical_df = train_league_brain(div_code)
        
        for game in odds_data: # Now iterating over Google-scraped odds
            profit, arb_info, bh, bd, ba = find_arbitrage_from_google(game, 'Soccer') # Use Google-specific arb finder
            match_time = game.get('commence_time', datetime.now(timezone.utc).isoformat())
            
            if profit > 0:
                bets.append({
                    'Date': match_time, 'Sport': 'Soccer', 'League': sport_key, 'Match': f"{game['home_team']} vs {game['away_team']}", 
                    'Bet Type': 'ARBITRAGE', 'Bet': 'ALL', 'Odds': 0.0, 'Edge': profit, 'Confidence': 1.0, 'Stake': 0.0, 'Info': arb_info
                })
                continue
            
            if brain and historical_df is not None:
                model_home = fuzzy_match_team(game['home_team'], list(brain['elo_ratings'].keys()))
                model_away = fuzzy_match_team(game['away_team'], list(brain['elo_ratings'].keys()))
                
                if model_home and model_away:
                    h_elo, a_elo = brain['elo_ratings'].get(model_home, 1500), brain['elo_ratings'].get(model_away, 1500)
                    h_py = calculate_pythagorean_expectation(brain['gf'].get(model_home, 0), brain['ga'].get(model_home, 0))
                    a_py = calculate_pythagorean_expectation(brain['gf'].get(model_away, 0), brain['ga'].get(model_away, 0))
                    
                    try:
                        h_form = historical_df[historical_df['HomeTeam'] == model_home].sort_values('Date').iloc[-1]['HomeForm']
                        a_form = historical_df[historical_df['AwayTeam'] == model_away].sort_values('Date').iloc[-1]['AwayForm']
                    except (IndexError, KeyError):
                        h_form, a_form = 1.5, 1.5 # Default form if not found
                    
                    feat_scaled = brain['scaler'].transform(pd.DataFrame([{
                        'elo_diff': h_elo - a_elo, 'form_diff': h_form - a_form, 'pyth_diff': h_py - a_py
                    }]))
                    probs_alpha = brain['model'].predict_proba(feat_scaled)[0]
                    
                    if brain['meta_model']:
                        trust_score = brain['meta_model'].predict_proba(feat_scaled)[0][1]
                        if trust_score < 0.55: # Adjust predictions if meta-model is uncertain
                            probs_alpha = (probs_alpha + np.array([0.33, 0.33, 0.33])) / 2
                    
                    try:
                        avg_goals_home, avg_goals_away = brain['avgs']; team_strengths = brain['team_strengths']
                        h_att, a_def = team_strengths.loc[model_home, 'attack'], team_strengths.loc[model_away, 'defence']
                        a_att, h_def = team_strengths.loc[model_away, 'attack'], team_strengths.loc[model_home, 'defence']
                        exp_h, exp_a = h_att * a_def * avg_goals_home, a_att * h_def * avg_goals_away
                        pm = np.array([[zero_inflated_poisson(i, exp_h) * zero_inflated_poisson(j, exp_a) for j in range(6)] for i in range(6)])
                        p_h, p_d, p_a = np.sum(np.tril(pm, -1)), np.sum(np.diag(pm)), np.sum(np.triu(pm, 1))
                    except:
                        p_h, p_d, p_a = 0.33, 0.33, 0.33
                    
                    final_probs = {
                        'Home Win': probs_alpha[brain['le'].transform(['H'])[0]] * 0.7 + p_h * 0.3,
                        'Draw': probs_alpha[brain['le'].transform(['D'])[0]] * 0.7 + p_d * 0.3,
                        'Away Win': probs_alpha[brain['le'].transform(['A'])[0]] * 0.7 + p_a * 0.3
                    }
                    
                    h_vol = brain['volatility'].get(model_home, 0.25)
                    a_vol = brain['volatility'].get(model_away, 0.25)
                    vol_factor = max(0.5, 1.0 - ((h_vol + a_vol)/2 - 0.25))
                    
                    for outcome, odds_data in [('Home Win', bh), ('Draw', bd), ('Away Win', ba)]:
                        if odds_data['price'] > 1.01: # Minimum odds check
                            edge = (final_probs[outcome] * odds_data['price']) - 1
                            if edge > 0.02: # Minimum edge check
                                bets.append({
                                    'Date': match_time, 'Sport': 'Soccer', 'League': sport_key, 
                                    'Match': f"{game['home_team']} vs {game['away_team']}", 'Bet Type': 'Moneyline', 
                                    'Bet': outcome, 'Odds': odds_data['price'], 'Edge': edge, 
                                    'Confidence': final_probs[outcome], 
                                    'Stake': (edge/(odds_data['price']-1))*0.25*vol_factor, 
                                    'Info': f"Best: {odds_data['book']}"
                                })
    
    return pd.DataFrame(bets)

# Placeholder implementations for other sports (follow same pattern as soccer)
def run_nfl_module():
    print("--- Running NFL Module (Google Odds & Scores) ---")
    # Fetch NFL odds from Google
    odds_data = get_live_odds_from_google('americanfootball_nfl')
    # Process odds_data similar to run_global_soccer_module
    # Use NFL-specific model if available
    return pd.DataFrame()

def run_nba_module():
    print("--- Running NBA Module (Google Odds & Scores) ---")
    # Fetch NBA odds from Google
    odds_data = get_live_odds_from_google('basketball_nba')
    # Process odds_data similar to run_global_soccer_module
    # Use NBA-specific model if available
    return pd.DataFrame()

def run_mlb_module():
    print("--- Running MLB Module (Google Odds & Scores) ---")
    # Fetch MLB odds from Google
    odds_data = get_live_odds_from_google('baseball_mlb')
    # Process odds_data similar to run_global_soccer_module
    # Use MLB-specific model if available
    return pd.DataFrame()

# --- MODULE EXPORTS ---
__all__ = [
    'run_global_soccer_module',
    'run_nfl_module',
    'run_nba_module',
    'run_mlb_module',
    'settle_bets'
]
