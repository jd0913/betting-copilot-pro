# betting_engine.py
# The Core Logic: AI Models, Google Data Fetching, Settlement, Feature Engineering
# v80.0 - REMOVAL: The Odds API & football-data.co.uk dependencies
# IMPLEMENTATION: Dynamic Google scraping for live odds, historical scores, and settlements
# FEATURES: Live data, Smart scraping, Dynamic adaptation, AI-driven analysis

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from scipy.stats import poisson
import joblib
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta, timezone
import os
import json
import random
from fuzzywuzzy import process
import logging
import config
from urllib.parse import quote_plus

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BettingEngine")

# ==============================================================================
# SECURE API SESSION (With retries and timeouts for Google)
# ==============================================================================
def create_api_session():
    """Create resilient session for Google scraping"""
    session = requests.Session()
    # Google is sensitive to bots, so we'll use lighter retry logic
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    retries = Retry(
        total=2,  # Lower retries for Google
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    return session

API_SESSION = create_api_session()

# ==============================================================================
# 1. SMART MATH MODULES (Preserved)
# ==============================================================================
def calculate_pythagorean_expectation(points_for, points_against, exponent=1.83):
    if points_for == 0 and points_against == 0: return 0.5
    if points_for < 0 or points_against < 0: return 0.5
    numerator = points_for ** exponent
    denominator = numerator + (points_against ** exponent)
    return numerator / denominator if denominator > 0 else 0.5

def zero_inflated_poisson(k, lam, pi=0.05):
    if lam <= 0 or pi < 0 or pi > 1:
        logger.warning(f"Invalid ZI Poisson params: k={k}, lam={lam}, pi={pi}")
        return poisson.pmf(k, max(lam, 0.1))
    if k == 0:
        return pi + (1 - pi) * poisson.pmf(0, lam)
    return (1 - pi) * poisson.pmf(k, lam)

# ==============================================================================
# 2. GENETIC AI & MODELS (Preserved, Adapted for Google Data)
# ==============================================================================
GENOME_FILE = 'model_genome.json'

def load_or_initialize_genome():
    if os.path.exists(GENOME_FILE):
        try:
            with open(GENOME_FILE, 'r') as f: return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.error("Genome file corrupted, initializing new one.")
    return {
        'generation': 0, 'best_score': 10.0, 'xgb_n_estimators': 200, 'xgb_max_depth': 3, 'xgb_learning_rate': 0.1,
        'rf_n_estimators': 200, 'rf_max_depth': 10, 'nn_hidden_layer_size': 64, 'nn_alpha': 0.0001
    }

def mutate_genome(genome):
    mutant = genome.copy(); mutation_rate = 0.3
    if random.random() < mutation_rate: mutant['xgb_n_estimators'] = int(genome['xgb_n_estimators'] * random.uniform(0.8, 1.2))
    if random.random() < mutation_rate: mutant['xgb_learning_rate'] = genome['xgb_learning_rate'] * random.uniform(0.8, 1.2)
    if random.random() < mutation_rate: mutant['rf_n_estimators'] = int(genome['rf_n_estimators'] * random.uniform(0.8, 1.2))
    if random.random() < mutation_rate: mutant['nn_hidden_layer_size'] = int(genome['nn_hidden_layer_size'] * random.uniform(0.8, 1.2))
    return mutant

def build_ensemble_from_genome(genome):
    xgb_clf = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, n_estimators=int(genome['xgb_n_estimators']), max_depth=int(genome['xgb_max_depth']), learning_rate=genome['xgb_learning_rate'], random_state=42, n_jobs=-1, tree_method='hist')
    rf_clf = RandomForestClassifier(n_estimators=int(genome['rf_n_estimators']), max_depth=int(genome['rf_max_depth']), random_state=42, n_jobs=-1)
    nn_clf = MLPClassifier(hidden_layer_sizes=(int(genome['nn_hidden_layer_size']), int(genome['nn_hidden_layer_size'] // 2)), alpha=genome['nn_alpha'], activation='relu', solver='adam', max_iter=500, random_state=42, early_stopping=True)
    lr_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    return VotingClassifier(estimators=[('xgb', xgb_clf), ('rf', rf_clf), ('nn', nn_clf), ('lr', lr_clf)], voting='soft', n_jobs=-1)

def train_meta_model(X, y, primary_model):
    try:
        preds = cross_val_predict(primary_model, X, y, cv=3)
        meta_y = (preds == y).astype(int)
        meta_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        meta_clf.fit(X, meta_y)
        return meta_clf
    except Exception as e:
        logger.error(f"Error training meta-model: {e}")
        return None

def evolve_and_train(X, y):
    print("   🧬 Initiating Evolutionary Cycle...")
    current_genome = load_or_initialize_genome(); mutant_genome = mutate_genome(current_genome); tscv = TimeSeriesSplit(n_splits=3)
    champ_model = build_ensemble_from_genome(current_genome); champ_scores = cross_val_score(champ_model, X, y, cv=tscv, scoring='neg_log_loss'); champ_fitness = -champ_scores.mean()
    mutant_model = build_ensemble_from_genome(mutant_genome); mutant_scores = cross_val_score(mutant_model, X, y, cv=tscv, scoring='neg_log_loss'); mutant_fitness = -mutant_scores.mean()
    
    if mutant_fitness < champ_fitness:
        print("      > 🚀 EVOLUTION! Mutant wins.")
        mutant_genome['best_score'] = mutant_fitness; mutant_genome['generation'] = current_genome['generation'] + 1; winner_genome = mutant_genome
    else:
        print("      > 💀 Champion remains.")
        winner_genome = current_genome
        
    final_model = build_ensemble_from_genome(winner_genome)
    meta_model = train_meta_model(X, y, final_model)
    calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=3)
    calibrated_model.fit(X, y)
    # Save genome
    with open(GENOME_FILE, 'w') as f: json.dump(winner_genome, f, indent=2)
    return calibrated_model, meta_model

# ==============================================================================
# 3. GOOGLE DATA FETCHING & PROCESSING (NEW CORE)
# ==============================================================================
def scrape_google_odds(match_name, sport_type="Soccer"):
    """
    Scrape Google for live odds for a specific match.
    Args:
        match_name (str): e.g., "Arsenal vs Chelsea"
        sport_type (str): Used for refining search query if needed
    Returns:
        dict: {'Home Win': odds, 'Draw': odds, 'Away Win': odds} or empty dict if not found
    """
    search_query = f"{match_name} {sport_type} odds"
    encoded_query = quote_plus(search_query)
    url = f"https://www.google.com/search?q={encoded_query}"
    
    headers = API_SESSION.headers.copy() # Use our configured session headers
    
    try:
        logger.info(f"Scraping Google for odds: {match_name}")
        response = API_SESSION.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for Google's sports odds structure (classes may change frequently)
        # This is the most fragile part - requires frequent updates to selectors
        # Common Google sports elements for odds
        odds_elements = soup.find_all(class_=lambda x: x and ('odds' in x.lower() or 'oddsboard' in x.lower() or 'imso_mh' in x.lower()))
        
        # Heuristic: Look for numbers in the right format near team names or generic odds labels
        text_content = soup.get_text()
        # Pattern: number followed by separator (- or :) followed by another number, e.g., "2 - 1", "3:0"
        score_pattern = r'(\d+)\s*[-:]\s*(\d+)'
        score_match = re.search(score_pattern, text_content)
        if score_match:
             # This might be the actual score, not odds. Be careful.
             # Odds are usually decimals like 2.10, 3.50, etc.
             pass
        
        # Look for decimal numbers that look like odds (usually > 1.0 and < 20.0)
        odds_pattern = r'\b\d+\.\d{2}\b'
        potential_odds = re.findall(odds_pattern, text_content)
        potential_odds = [float(o) for o in potential_odds if 1.0 < float(o) < 20.0] # Filter realistic odds
        
        # This is highly heuristic. We assume Google might list odds like [Home, Draw, Away] or [Home, Away] for 2-way sports
        # We need to associate these numbers with outcomes.
        # For now, let's just return the first 3 unique odds found, assuming they are H, D, A (for 3-way sports)
        # This is a MAJOR limitation of pure scraping without structured data.
        if len(potential_odds) >= 3:
            unique_odds = list(dict.fromkeys(potential_odds))[:3] # Remove duplicates, keep first 3
            if sport_type.lower() in ['soccer', 'football']:
                return {'Home Win': unique_odds[0], 'Draw': unique_odds[1], 'Away Win': unique_odds[2]}
            elif sport_type.lower() in ['nfl', 'nba', 'mlb']: # 2-way sport
                 return {'Home Win': unique_odds[0], 'Away Win': unique_odds[1]} # Ignore 3rd if present or treat as spread
        
        # Alternative: Look for specific text indicating odds
        # Example: "Odds: Arsenal 2.10 Draw 3.50 Chelsea 3.20"
        # This requires very specific parsing based on how the text is laid out.
        # Google's layout is complex and constantly changing.
        # This is why APIs are preferred.
        
        # Placeholder for more complex parsing logic if needed in the future
        # Look for common bookmaker names near potential odds
        # Look for specific Google sports data attributes (e.g., data-attr-odds)
        # Inspect the actual HTML structure returned by Google for your specific query.
        
        logger.info(f"No clear odds structure found for {match_name} on Google.")
        return {}
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error scraping odds for {match_name}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error parsing odds for {match_name}: {e}")
        return {}

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
    encoded_query = quote_plus(search_query)
    url = f"https://www.google.com/search?q={encoded_query}"
    
    headers = API_SESSION.headers.copy()
    
    try:
        logger.info(f"Scraping Google for scores: {date_str}, League: {league}")
        response = API_SESSION.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find score containers (again, fragile, depends on Google's layout)
        score_containers = soup.find_all('div', class_=lambda x: x and ('imso_mh' in x.lower() or 'score' in x.lower()))
        
        scores = []
        for container in score_containers:
            # Extract match info and score
            # Example structure (may vary):
            # <div class="imso_mh__mv"> <div class="imso_mh__tm-txt">HomeTeam</div> <div class="imso_mh__scr-sep">1 - 0</div> <div class="imso_mh__tm-txt">AwayTeam</div> </div>
            home_team_elem = container.find(class_=lambda x: x and 'imso_mh__tm-txt' in x) # First team found
            score_elem = container.find(class_=lambda x: x and 'imso_mh__scr-sep' in x) or container.find(class_=lambda x: x and 'imso_mh__s-t' in x)
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

# ==============================================================================
# 4. DYNAMIC SOCCER MODULE (Powered by Google)
# ==============================================================================
def run_global_soccer_module():
    """
    Generates bets using Google for live odds and historical data.
    This is a simplified version focusing on fetching data and basic processing.
    """
    print("--- Running Global Soccer Module (Google-Powered) ---")
    bets = []
    
    # Example: Fetch live odds for a specific upcoming match (you'd need a way to get upcoming matches first)
    # For demonstration, let's assume we have a list of matches to check.
    # In reality, you might scrape a site for schedules or use a simple predefined list.
    # Let's say we get today's date and search for matches happening today.
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    # Search for today's matches (this is also scraping-dependent)
    # We'll simulate getting a match name from somewhere else for now.
    # A more robust system would have a separate function to scrape schedules.
    # Placeholder: Assume we have a match name
    simulated_match_name = "Generic Team A vs Generic Team B" 
    
    # Fetch live odds from Google
    odds_dict = scrape_google_odds(simulated_match_name, "Soccer")
    if odds_dict:
        print(f"   > Odds found for {simulated_match_name}: {odds_dict}")
        # Example: Perform basic arbitrage check (requires 3 odds for H, D, A)
        if 'Home Win' in odds_dict and 'Draw' in odds_dict and 'Away Win' in odds_dict:
            h_odds, d_odds, a_odds = odds_dict['Home Win'], odds_dict['Draw'], odds_dict['Away Win']
            implied_prob = (1/h_odds) + (1/d_odds) + (1/a_odds)
            if implied_prob < 0.99: # 1% margin for error
                profit = (1 - implied_prob) / implied_prob
                bets.append({
                    'Date': datetime.now(timezone.utc).isoformat(), # Use current time or scheduled time if available
                    'Sport': 'Soccer',
                    'League': 'Simulated/Google',
                    'Match': simulated_match_name,
                    'Bet_Type': 'ARBITRAGE',
                    'Bet': 'ALL',
                    'Odds': 1 / (1 - profit), # Recalculate fair odds
                    'Edge': profit,
                    'Confidence': 1.0, # Arbitrage is risk-free theoretically
                    'Stake': 0.0, # Needs calculation
                    'Info': f"Google Arb: {h_odds:.2f}, {d_odds:.2f}, {a_odds:.2f}"
                })
                logger.info(f"   > Arbitrage opportunity found via Google: {simulated_match_name}")
        
        # Example: Use odds for model-based betting (requires historical data for training)
        # Since we removed football-data.co.uk, we need an alternative for historical data.
        # Option 1: Scrape historical results from Google for each team over the last N days/weeks
        # This is complex and slow but possible.
        # Option 2: Use a different free/paid API if available.
        # Option 3: Use the live odds and basic team info scraped from Google to create features.
        # For this example, let's assume we have a way to get minimal historical context via Google.
        # This is where the model training part becomes tricky without a dedicated historical data source.
        # Let's focus on the settlement and live odds aspects for now.
    
    # For now, return what we have (likely empty or with arb if found)
    # The core functionality of generating *value* bets based on historical model training is significantly impacted
    # by the removal of the historical data API. The settlement part (below) is where Google shines.
    return pd.DataFrame(bets)


# ==============================================================================
# 5. DYNAMIC SETTLEMENT ENGINE (Powered by Google)
# ==============================================================================
def settle_bets():
    """
    Settle bets by fetching scores from Google.
    """
    print("--- ⚖️ Running Google-Based Settlement Engine ---")
    history_file = 'betting_history.csv'
    if not os.path.exists(history_file): 
        logger.info("No history file found for settlement.")
        return
    
    df = pd.read_csv(history_file)
    if 'Result' not in df.columns: df['Result'] = 'Pending'
    if 'Profit' not in df.columns: df['Profit'] = 0.0
    if 'Score' not in df.columns: df['Score'] = ''
    
    # Filter for pending bets that are old enough to be settled (e.g., > 2 hours after match time)
    current_time = datetime.now(timezone.utc)
    pending_mask = (df['Result'] == 'Pending') & (df['Sport'] == 'Soccer') # Add other sports if needed
    
    for idx, row in df[pending_mask].iterrows():
        match_name = row['Match']
        # Assume 'Date' in the history file is the match start time in ISO format
        try:
            match_time = pd.to_datetime(row['Date'], utc=True)
        except:
            logger.warning(f"Could not parse date for row {idx}, skipping settlement.")
            continue
            
        # Only settle if match is old enough
        if match_time > current_time - timedelta(hours=2):
            continue # Skip recent matches
        
        # Determine the date to search for scores (match date)
        match_date_str = match_time.strftime('%Y-%m-%d')
        
        # Scrape Google for the final score of this specific match and date
        # This is the core of the Google settlement.
        # We need to search for the exact match on the specific date.
        scores_found = scrape_google_scores(match_date_str, league=row.get('League', ''), limit=50) # Search broadly for the date
        
        # Find the specific match in the results
        match_result = None
        for score_entry in scores_found:
            if match_name.lower() in score_entry['Match'].lower(): # Fuzzy match
                match_result = score_entry
                break
        
        if match_result:
            score_str = match_result['Score']
            # Parse the score string (e.g., "2 - 1", "3:0")
            score_parts = re.split(r'[-:]', score_str)
            if len(score_parts) == 2:
                try:
                    home_score = int(score_parts[0].strip())
                    away_score = int(score_parts[1].strip())
                    
                    # Determine the actual result
                    if home_score > away_score:
                        actual_result = 'Home Win'
                    elif away_score > home_score:
                        actual_result = 'Away Win'
                    else: # home_score == away_score
                        actual_result = 'Draw'
                    
                    # Determine if the bet was won
                    bet_outcome = row['Bet']
                    if (bet_outcome == 'Home Win' and actual_result == 'Home Win') or \
                       (bet_outcome == 'Away Win' and actual_result == 'Away Win') or \
                       (bet_outcome == 'Draw' and actual_result == 'Draw'):
                        df.loc[idx, 'Result'] = 'Win'
                        df.loc[idx, 'Profit'] = row['Stake'] * (row['Odds'] - 1)
                    else:
                        df.loc[idx, 'Result'] = 'Loss'
                        df.loc[idx, 'Profit'] = -row['Stake']
                    
                    df.loc[idx, 'Score'] = score_str
                    logger.info(f"   > Settled bet for {match_name} ({match_time.date()}) as {actual_result} with score {score_str}")
                    
                except (ValueError, IndexError):
                    logger.warning(f"   > Could not parse score '{score_str}' for {match_name}")
                    # Maybe mark as 'Auto-Settled' or leave as 'Pending'?
                    df.loc[idx, 'Result'] = 'Auto-Settled'
                    df.loc[idx, 'Profit'] = -row['Stake']
                    df.loc[idx, 'Score'] = 'N/A (Parse Error)'
            else:
                logger.warning(f"   > Unrecognized score format '{score_str}' for {match_name}")
                df.loc[idx, 'Result'] = 'Auto-Settled'
                df.loc[idx, 'Profit'] = -row['Stake']
                df.loc[idx, 'Score'] = 'N/A (Format Error)'
        else:
            # If no score found after sufficient time (e.g., 3 days), mark as Auto-Settled
            if match_time < current_time - timedelta(days=3):
                df.loc[idx, 'Result'] = 'Auto-Settled'
                df.loc[idx, 'Profit'] = -row['Stake']
                df.loc[idx, 'Score'] = 'N/A (Score Not Found)'
                logger.info(f"   > Auto-settled (score not found) bet for {match_name} ({match_time.date()})")
    
    # Save the updated history
    df.to_csv(history_file, index=False)
    settled_count = len(df[df['Result'].isin(['Win', 'Loss', 'Auto-Settled'])])
    logger.info(f"--- Settlement Complete: {settled_count} bets settled using Google ---")


# Placeholder implementations for other sports modules (NFL, NBA, MLB)
# They would follow the same pattern as soccer but adapted for 2-way outcomes (no draw)
def run_nfl_module():
    logger.warning("NFL module not implemented in Google-only version yet.")
    return pd.DataFrame()

def run_nba_module():
    logger.warning("NBA module not implemented in Google-only version yet.")
    return pd.DataFrame()

def run_mlb_module():
    logger.warning("MLB module not implemented in Google-only version yet.")
    return pd.DataFrame()

# ==============================================================================
# MODULE EXPORTS
# ==============================================================================
__all__ = [
    'run_global_soccer_module',
    'run_nfl_module',
    'run_nba_module',
    'run_mlb_module',
    'settle_bets'
]
