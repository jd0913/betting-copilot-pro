# betting_engine.py
# The Core Logic: AI Models, Google Data Fetching, Settlement, Feature Engineering
# v72.0 (Google-Only Edition - No Config Dependency)
# FIX: Removed import config and replaced with hardcoded values/env vars
# FIX: Integrated Google score lookup for settlement

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
import re # Added for score parsing

# ==============================================================================
# LOGGING SETUP (Professional standard)
# ==============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BettingEngine")

# ==============================================================================
# SECURE API SESSION (With retries and timeouts)
# ==============================================================================
def create_api_session():
    """Create resilient session with retry logic"""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({
        "User-Agent": "BettingCoPilot-Pro/1.0 (Contact: your@email.com)",
        "Accept": "application/json"
    })
    return session

API_SESSION = create_api_session()

# ==============================================================================
# CORE MATH MODULES (Preserved)
# ==============================================================================
def calculate_pythagorean_expectation(points_for, points_against, exponent=1.83):
    if points_for == 0 and points_against == 0: return 0.5
    return (points_for ** exponent) / ((points_for ** exponent) + (points_against ** exponent))

def zero_inflated_poisson(k, lam, pi=0.05):
    if k == 0:
        return pi + (1 - pi) * poisson.pmf(0, lam)
    else:
        return (1 - pi) * poisson.pmf(k, lam)

# ==============================================================================
# GENETIC AI & MODELS (Preserved)
# ==============================================================================
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
    xgb_clf = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, n_estimators=int(genome['xgb_n_estimators']), max_depth=int(genome['xgb_max_depth']), learning_rate=genome['xgb_learning_rate'], random_state=42, n_jobs=-1)
    rf_clf = RandomForestClassifier(n_estimators=int(genome['rf_n_estimators']), max_depth=int(genome['rf_max_depth']), random_state=42, n_jobs=-1)
    nn_clf = MLPClassifier(hidden_layer_sizes=(int(genome['nn_hidden_layer_size']), int(genome['nn_hidden_layer_size'] // 2)), alpha=genome['nn_alpha'], activation='relu', solver='adam', max_iter=500, random_state=42)
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

# ==============================================================================
# DATA FETCHING & ARBITRAGE (Updated for Google-Only)
# ==============================================================================
# HARDCODED GITHUB CREDENTIALS (Replaces config.GITHUB_CONFIG)
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "jd0913") # Use env var or default
GITHUB_REPO = os.getenv("GITHUB_REPO", "betting-copilot-pro") # Use env var or default

# Fixed URL formatting
LATEST_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/latest_bets.csv"
HISTORY_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/betting_history.csv"

def get_live_odds(sport_key):
    # REMOVED: The Odds API integration - no longer needed
    # This function is now a placeholder as live odds are fetched via Google scraping
    logger.info(f"Live odds for {sport_key} would be fetched via Google scraping in the future.")
    return []

def find_arbitrage(game, sport_type):
    # REMOVED: The Odds API arb logic - no longer needed
    # This function is now a placeholder as arb is found via Google scraping
    return 0, None, {'price': 0, 'book': ''}, {'price': 0, 'book': ''}, {'price': 0, 'book': ''}

def fuzzy_match_team(team_name, team_list):
    if not team_list: return None
    match, score = process.extractOne(team_name, team_list)
    if score >= 80: return match
    return None

def get_news_alert(team1, team2):
    # Kept the news alert functionality - uses Google search
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

# ==============================================================================
# GOOGLE SCORE LOOKUP FOR SETTLEMENT (NEW CORE FUNCTION)
# ==============================================================================

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
                    home_score = score_match.group(1)
                    away_score = score_match.group(2)
                    logger.info(f"   > Google score found: {home_score} - {away_score}")
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
                    logger.info(f"   > Google score found (fallback): {home_score} - {away_score}")
                    return f"{home_score} - {away_score}"
        
        logger.info(f"   > No score found on Google for {match_name} ({match_date_str})")
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

def settle_soccer_bets(df):
    """
    Settle soccer bets using Google score lookup.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    current_time = datetime.now(timezone.utc)
    settled_count = 0
    
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    for idx, row in df_copy.iterrows():
        # Skip if already properly settled
        if row.get('Result') in ['Win', 'Loss', 'Push'] and row.get('Score') and row.get('Score') != 'N/A' and row.get('Score') != 'nan':
            continue
            
        # Get match details
        try:
            match_time = pd.to_datetime(row['Date_Obj'], utc=True)
        except (ValueError, TypeError):
            continue # Skip if date parsing fails
            
        match_name = row['Match']
        predicted_bet = row['Bet']
        current_result = row.get('Result', 'Pending')
        current_score = row.get('Score', 'N/A')
        
        # Only process matches that occurred before Dec 8, 2025 @ 10pm UTC
        settlement_deadline = datetime(2025, 12, 8, 22, 0, 0, tzinfo=timezone.utc)
        if match_time > settlement_deadline:
            continue # Skip future matches or recent ones needing time to settle
        
        # If match is old enough, try to get score from Google
        if match_time < current_time - timedelta(hours=2): # At least 2 hours old
            score_str = get_google_score(match_name, match_time.strftime('%Y-%m-%d'))
            
            if score_str and ' - ' in score_str:
                try:
                    home_score_str, away_score_str = score_str.split(' - ')
                    home_score = int(home_score_str.strip())
                    away_score = int(away_score_str.strip())
                    
                    # Determine actual outcome
                    actual_result = determine_match_result(home_score, away_score)
                    
                    # Determine if bet won
                    if (predicted_bet == 'Home Win' and actual_result == 'Home Win') or \
                       (predicted_bet == 'Away Win' and actual_result == 'Away Win') or \
                       (predicted_bet == 'Draw' and actual_result == 'Draw'):
                        df_copy.at[idx, 'Result'] = 'Win'
                        df_copy.at[idx, 'Profit'] = row['Stake'] * (row['Odds'] - 1)
                    else:
                        df_copy.at[idx, 'Result'] = 'Loss'
                        df_copy.at[idx, 'Profit'] = -row['Stake']
                    
                    df_copy.at[idx, 'Score'] = score_str
                    settled_count += 1
                    logger.info(f"   > Settled bet for {match_name} as {actual_result} with score {score_str}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"   > Error processing Google score '{score_str}' for {match_name}: {str(e)}")
                    # Mark as Auto-Settled if score parsing fails
                    df_copy.at[idx, 'Result'] = 'Auto-Settled'
                    df_copy.at[idx, 'Profit'] = -row.get('Stake', 0)
                    df_copy.at[idx, 'Score'] = 'N/A (Parse Error)'
                    settled_count += 1
            else:
                # If no score found after 3 days, mark as Auto-Settled (conservative approach)
                if match_time < current_time - timedelta(days=3):
                    df_copy.at[idx, 'Result'] = 'Auto-Settled'
                    df_copy.at[idx, 'Profit'] = -row.get('Stake', 0)
                    df_copy.at[idx, 'Score'] = 'N/A (Score Not Found)'
                    settled_count += 1
                    logger.info(f"   > Auto-settled (score not found) bet for {match_name} ({match_time.date()})")
    
    if settled_count > 0:
        logger.info(f"   > Google-based settlement complete. {settled_count} bets settled.")
    
    return df_copy

# ==============================================================================
# GLOBAL SOCCER MODULE (Updated for Google-Only)
# ==============================================================================
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

def run_global_soccer_module():
    print("--- Running Global Soccer Module (Big 5 + UCL) ---"); bets = []
    LEAGUE_MAP = {'soccer_epl': 'E0', 'soccer_spain_la_liga': 'SP1', 'soccer_germany_bundesliga': 'D1', 'soccer_italy_serie_a': 'I1', 'soccer_france_ligue_one': 'F1', 'soccer_uefa_champs_league': 'UCL'}
    for sport_key, div_code in LEAGUE_MAP.items():
        print(f"   > Scanning {sport_key}...")
        odds_data = get_live_odds(sport_key) # This now returns empty list - live odds will come from Google
        brain = None; historical_df = None
        if div_code != 'UCL': brain, historical_df = train_league_brain(div_code)
        for game in odds_data: # This loop will now be skipped as odds_data is empty
            profit, arb_info, bh, bd, ba = find_arbitrage(game, 'Soccer')
            match_time = game.get('commence_time', 'Unknown')
            if profit > 0: bets.append({'Date': match_time, 'Sport': 'Soccer', 'League': sport_key, 'Match': f"{game['home_team']} vs {game['away_team']}", 'Bet Type': 'ARBITRAGE', 'Bet': 'ALL', 'Odds': 0.0, 'Edge': profit, 'Confidence': 1.0, 'Stake': 0.0, 'Info': arb_info}); continue
            if brain and historical_df is not None:
                model_home = fuzzy_match_team(game['home_team'], list(brain['elo_ratings'].keys())); model_away = fuzzy_match_team(game['away_team'], list(brain['elo_ratings'].keys()))
                if model_home and model_away:
                    h_elo, a_elo = brain['elo_ratings'].get(model_home, 1500), brain['elo_ratings'].get(model_away, 1500)
                    h_py = calculate_pythagorean_expectation(brain['gf'].get(model_home, 0), brain['ga'].get(model_home, 0))
                    a_py = calculate_pythagorean_expectation(brain['gf'].get(model_away, 0), brain['ga'].get(model_away, 0))
                    try: h_form = historical_df[historical_df['HomeTeam'] == model_home].sort_values('Date').iloc[-1]['HomeForm']; a_form = historical_df[historical_df['AwayTeam'] == model_away].sort_values('Date').iloc[-1]['AwayForm']
                    except: h_form, a_form = 1.5, 1.5
                    feat_scaled = brain['scaler'].transform(pd.DataFrame([{'elo_diff': h_elo - a_elo, 'form_diff': h_form - a_form, 'pyth_diff': h_py - a_py}]))
                    probs_alpha = brain['model'].predict_proba(feat_scaled)[0]
                    if brain['meta_model']:
                        trust_score = brain['meta_model'].predict_proba(feat_scaled)[0][1]
                        if trust_score < 0.55: probs_alpha = (probs_alpha + np.array([0.33, 0.33, 0.33])) / 2
                    try:
                        avg_goals_home, avg_goals_away = brain['avgs']; team_strengths = brain['team_strengths']
                        h_att, a_def = team_strengths.loc[model_home]['attack'], team_strengths.loc[model_away]['defence']; a_att, h_def = team_strengths.loc[model_away]['attack'], team_strengths.loc[model_home]['defence']
                        exp_h, exp_a = h_att * a_def * avg_goals_home, a_att * h_def * avg_goals_away
                        pm = np.array([[zero_inflated_poisson(i, exp_h) * zero_inflated_poisson(j, exp_a) for j in range(6)] for i in range(6)])
                        p_h, p_d, p_a = np.sum(np.tril(pm, -1)), np.sum(np.diag(pm)), np.sum(np.triu(pm, 1))
                    except: p_h, p_d, p_a = 0.33, 0.33, 0.33
                    final_probs = {'Home Win': probs_alpha[brain['le'].transform(['H'])[0]] * 0.7 + p_h * 0.3, 'Draw': probs_alpha[brain['le'].transform(['D'])[0]] * 0.7 + p_d * 0.3, 'Away Win': probs_alpha[brain['le'].transform(['A'])[0]] * 0.7 + p_a * 0.3}
                    h_vol = brain['volatility'].get(model_home, 0.25); a_vol = brain['volatility'].get(model_away, 0.25); vol_factor = 1.0 - ((h_vol + a_vol)/2 - 0.25)
                    for outcome, odds_data in [('Home Win', bh), ('Draw', bd), ('Away Win', ba)]:
                        if odds_data['price'] > 0:
                            edge = (final_probs[outcome] * odds_data['price']) - 1
                            if edge > 0.05: bets.append({'Date': match_time, 'Sport': 'Soccer', 'League': sport_key, 'Match': f"{game['home_team']} vs {game['away_team']}", 'Bet Type': 'Moneyline', 'Bet': outcome, 'Odds': odds_data['price'], 'Edge': edge, 'Confidence': final_probs[outcome], 'Stake': (edge/(odds_data['price']-1))*0.25*vol_factor, 'Info': f"Best: {odds_data['book']}"})
    return pd.DataFrame(bets)

# ==============================================================================
# NFL/NBA/MLB MODULES (Placeholder - implement similarly to soccer with Google scraping)
# ==============================================================================
def run_nfl_module():
    print("--- Running NFL Module ---")
    # Implement similar logic using Google scraping for live NFL odds
    return pd.DataFrame()

def run_nba_module():
    print("--- Running NBA Module ---")
    # Implement similar logic using Google scraping for live NBA odds
    return pd.DataFrame()

def run_mlb_module():
    print("--- Running MLB Module ---")
    # Implement similar logic using Google scraping for live MLB odds
    return pd.DataFrame()

# ==============================================================================
# SETTLEMENT ENGINE (Updated for Google-Only)
# ==============================================================================
def settle_bets():
    """Main settlement function using Google score lookup for all sports"""
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
    
    # Route by sport (currently only Soccer has specific settlement logic implemented)
    # For other sports, you would add similar functions
    if 'Sport' in df.columns:
        soccer_mask = df['Sport'] == 'Soccer'
        df.loc[soccer_mask] = settle_soccer_bets(df[soccer_mask].copy())
        # TODO: Add settle_nfl_bets(df[nfl_mask]), etc.
    else:
        df = settle_soccer_bets(df.copy()) # Default to soccer if no Sport column

    # Save the updated history
    try:
        df.to_csv(history_file, index=False)
        settled_count = len(df[df['Result'].isin(['Win', 'Loss', 'Push', 'Auto-Settled'])])
        print(f"Settlement complete. {settled_count} bets settled using Google score lookup.")
    except Exception as e:
        print(f"Failed to save settled history: {str(e)}")

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
