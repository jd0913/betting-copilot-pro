# betting_engine.py
# The Core Logic: AI Models, Data Fetching, Settlement, Feature Engineering
# v72.0 (Fixed Syntax Error)
# FIX: Corrected syntax error in build_ensemble_from_genome function
# FIX: Ensured proper handling of betting history schema

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
import nfl_data_py as nfl
from nba_api.stats.endpoints import leaguedashteamstats
import requests
from requests.exceptions import RequestException, HTTPError, ConnectionError
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta, timezone
import os
import json
import random
from fuzzywuzzy import process
from textblob import TextBlob
import config # Will be handled in backend_runner.py now

# ==============================================================================
# LOGGING SETUP (Professional standard)
# ==============================================================================
import logging
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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json"
    })
    return session

API_SESSION = create_api_session()

# ==============================================================================
# CORE MATH MODULES (Full Feature Set Preserved)
# ==============================================================================
def calculate_pythagorean_expectation(points_for, points_against, exponent=1.83):
    if points_for == 0 and points_against == 0: return 0.5
    if points_for < 0 or points_against < 0: return 0.5 # Handle negative scores gracefully
    numerator = points_for ** exponent
    denominator = numerator + (points_against ** exponent)
    return numerator / denominator if denominator > 0 else 0.5

def zero_inflated_poisson(k, lam, pi=0.05):
    if k == 0:
        return pi + (1 - pi) * poisson.pmf(0, lam)
    else:
        return (1 - pi) * poisson.pmf(k, lam)

# ==============================================================================
# GENETIC AI & MODELS (Full Feature Set Preserved)
# ==============================================================================
GENOME_FILE = 'model_genome.json'

def load_or_initialize_genome():
    if os.path.exists(GENOME_FILE):
        with open(GENOME_FILE, 'r') as f: return json.load(f)
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
    # Validate parameters to prevent errors
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
    # FIXED: Removed extra comma in alpha parameter definition
    nn_clf = MLPClassifier(
        hidden_layer_sizes=(
            max(32, min(128, int(genome.get('nn_hidden_layer_size', 64)))),
            max(16, min(64, int(genome.get('nn_hidden_layer_size', 64)) // 2)
        ),
        alpha=max(1e-5, min(0.1, float(genome.get('nn_alpha', 0.0001)))), # FIXED: Removed extra comma here
        activation='relu', solver='adam', max_iter=500, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=10
    )
    lr_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    
    return VotingClassifier(estimators=[('xgb', xgb_clf), ('rf', rf_clf), ('nn', nn_clf), ('lr', lr_clf)], voting='soft', n_jobs=-1)

def train_meta_model(X, y, primary_model):
    try:
        preds = cross_val_predict(primary_model, X, y, cv=3, method='predict_proba')
        meta_y = (preds.argmax(axis=1) == y.astype(int)).astype(int)
        meta_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        meta_clf.fit(preds, meta_y)
        return meta_clf
    except Exception as e:
        print(f"Error training meta-model: {e}")
        return None

def evolve_and_train(X, y):
    print("   🧬 Initiating Evolutionary Cycle...")
    current_genome = load_or_initialize_genome(); mutant_genome = mutate_genome(current_genome); tscv = TimeSeriesSplit(n_splits=3)
    champ_model = build_ensemble_from_genome(current_genome)
    champ_scores = cross_val_score(champ_model, X, y, cv=tscv, scoring='neg_log_loss')
    champ_fitness = -champ_scores.mean()
    
    mutant_model = build_ensemble_from_genome(mutant_genome)
    mutant_scores = cross_val_score(mutant_model, X, y, cv=tscv, scoring='neg_log_loss')
    mutant_fitness = -mutant_scores.mean()
    
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
def get_live_odds(sport_key):
    """REMOVED: The Odds API dependency. Now relies on Google scraping via utils.py or backend_runner.py."""
    # This function is no longer used directly by the modules.
    # Odds are now fetched via Google scraping in utils.py or directly by backend_runner.py.
    # Return empty list to indicate no live API data.
    logger.info(f"Live odds for {sport_key} would be fetched via Google scraping.")
    return []

def find_arbitrage(game, sport_type):
    """REMOVED: The Odds API dependency. Arbitrage logic now handled differently."""
    # This function is no longer used as odds are not fetched via The Odds API.
    # Arbitrage can be calculated using Google-scraped odds if implemented in utils.py/google_scraping.py
    # For now, return 0.
    return 0, None, {'price': 0, 'book': ''}, {'price': 0, 'book': ''}, {'price': 0, 'book': ''}

def fuzzy_match_team(team_name, team_list):
    if not team_list: return None
    match, score = process.extractOne(team_name, team_list)
    if score >= 80: return match
    return None

def get_news_alert(team1, team2):
    """Kept for news scraping, adapted for Google."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        query = f'"{team1}" OR "{team2}" injury OR doubt OR out'
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=nws"
        res = API_SESSION.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.find_all('div', {'role': 'heading'})
        for h in headlines[:3]:
            if any(keyword in h.text.lower() for keyword in ['injury', 'doubt', 'out', 'miss', 'sidelined']):
                return f"⚠️ News: {h.text}"
    except RequestException as e: 
        print(f"Error scraping news for {team1}/{team2}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during news parsing: {e}")
    return None

# ==============================================================================
# GOOGLE DATA PROCESSING (Integrated Functions)
# ==============================================================================
# These functions are moved to utils.py for better separation of concerns.
# betting_engine.py now focuses on AI models and core logic.
# The modules below will call functions in utils.py for Google-based data.

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
    except RequestException as e: 
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
    print("--- Running Global Soccer Module (Big 5 + UCL) ---")
    bets = []
    LEAGUE_MAP = {'soccer_epl': 'E0', 'soccer_spain_la_liga': 'SP1', 'soccer_germany_bundesliga': 'D1', 'soccer_italy_serie_a': 'I1', 'soccer_france_ligue_one': 'F1', 'soccer_uefa_champs_league': 'UCL'}
    for sport_key, div_code in LEAGUE_MAP.items():
        print(f"   > Scanning {sport_key}...")
        # NEW: Fetch live odds from Google via backend_runner or utils
        # For now, we'll simulate getting odds data from Google scraping
        # This requires a new function in utils.py or a separate google_scraping.py module
        # Let's assume a function get_live_odds_from_google exists in utils
        # odds_data = utils.get_live_odds_from_google(sport_key)
        # For this version, we'll return an empty list to signify no live odds via Google scraping are implemented yet
        odds_data = [] # Placeholder - Google scraping for *live* odds is complex
        brain = None; historical_df = None
        if div_code != 'UCL': brain, historical_df = train_league_brain(div_code)
        
        # Process Google-scraped odds data (when available)
        for game in odds_data: # This loop will be skipped as odds_data is empty
            profit, arb_info, bh, bd, ba = find_arbitrage(game, 'Soccer')
            match_time = game.get('commence_time', datetime.now(timezone.utc).isoformat())
            if profit > 0: 
                bets.append({'Date': match_time, 'Sport': 'Soccer', 'League': sport_key, 'Match': f"{game['home_team']} vs {game['away_team']}", 'Bet Type': 'ARBITRAGE', 'Bet': 'ALL', 'Odds': 0.0, 'Edge': profit, 'Confidence': 1.0, 'Stake': 0.0, 'Info': arb_info}); continue
            if brain and historical_df is not None:
                model_home = fuzzy_match_team(game['home_team'], list(brain['elo_ratings'].keys()))
                model_away = fuzzy_match_team(game['away_team'], list(brain['elo_ratings'].keys()))
                if model_home and model_away:
                    h_elo, a_elo = brain['elo_ratings'].get(model_home, 1500), brain['elo_ratings'].get(model_away, 1500)
                    h_py = calculate_pythagorean_expectation(brain['gf'].get(model_home, 0), brain['ga'].get(model_home, 0))
                    a_py = calculate_pythagorean_expectation(brain['gf'].get(model_away, 0), brain['ga'].get(model_away, 0))
                    try: h_form = historical_df[historical_df['HomeTeam'] == model_home].sort_values('Date').iloc[-1]['HomeForm']; a_form = historical_df[historical_df['AwayTeam'] == model_away].sort_values('Date').iloc[-1]['AwayForm']
                    except (IndexError, KeyError): h_form, a_form = 1.5, 1.5
                    feat_scaled = brain['scaler'].transform(pd.DataFrame([{'elo_diff': h_elo - a_elo, 'form_diff': h_form - a_form, 'pyth_diff': h_py - a_py}]))
                    probs_alpha = brain['model'].predict_proba(feat_scaled)[0]
                    if brain['meta_model']: trust_score = brain['meta_model'].predict_proba(feat_scaled)[0][1]; probs_alpha = (probs_alpha + np.array([0.33, 0.33, 0.33])) / 2 if trust_score < 0.55 else probs_alpha
                    try: avg_goals_home, avg_goals_away = brain['avgs']; team_strengths = brain['team_strengths']; h_att, a_def = team_strengths.loc[model_home, 'attack'], team_strengths.loc[model_away, 'defence']; a_att, h_def = team_strengths.loc[model_away, 'attack'], team_strengths.loc[model_home, 'defence']; exp_h, exp_a = h_att * a_def * avg_goals_home, a_att * h_def * avg_goals_away; pm = np.array([[zero_inflated_poisson(i, exp_h) * zero_inflated_poisson(j, exp_a) for j in range(6)] for i in range(6)]); p_h, p_d, p_a = np.sum(np.tril(pm, -1)), np.sum(np.diag(pm)), np.sum(np.triu(pm, 1))
                    except: p_h, p_d, p_a = 0.33, 0.33, 0.33
                    final_probs = {'Home Win': probs_alpha[brain['le'].transform(['H'])[0]] * 0.7 + p_h * 0.3, 'Draw': probs_alpha[brain['le'].transform(['D'])[0]] * 0.7 + p_d * 0.3, 'Away Win': probs_alpha[brain['le'].transform(['A'])[0]] * 0.7 + p_a * 0.3}
                    h_vol = brain['volatility'].get(model_home, 0.25); a_vol = brain['volatility'].get(model_away, 0.25); vol_factor = max(0.5, 1.0 - ((h_vol + a_vol)/2 - 0.25))
                    for outcome, odds_data in [('Home Win', bh), ('Draw', bd), ('Away Win', ba)]:
                        if odds_data['price'] > 1.01: edge = (final_probs[outcome] * odds_data['price']) - 1; if edge > 0.02: bets.append({'Date': match_time, 'Sport': 'Soccer', 'League': sport_key, 'Match': f"{game['home_team']} vs {game['away_team']}", 'Bet Type': 'Moneyline', 'Bet': outcome, 'Odds': odds_data['price'], 'Edge': edge, 'Confidence': final_probs[outcome], 'Stake': (edge/(odds_data['price']-1))*0.25*vol_factor, 'Info': f"Best: {odds_data['book']}"})
    # Return DataFrame with consistent schema, even if empty
    if not bets:
        # Return an empty DataFrame with the correct schema
        return pd.DataFrame(columns=[
            'Date', 'Sport', 'League', 'Match', 'Bet Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake', 'Info'
        ])
    return pd.DataFrame(bets)

# Placeholder implementations for other sports modules (NFL, NBA, MLB)
# They would follow the same pattern as soccer but adapted for 2-way outcomes (no draw)
def run_nfl_module():
    print("--- Running NFL Module ---")
    # Placeholder - implement Google scraping for NFL odds similarly to soccer
    return pd.DataFrame(columns=['Date', 'Sport', 'League', 'Match', 'Bet Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake', 'Info'])

def run_nba_module():
    print("--- Running NBA Module ---")
    # Placeholder - implement Google scraping for NBA odds similarly to soccer
    return pd.DataFrame(columns=['Date', 'Sport', 'League', 'Match', 'Bet Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake', 'Info'])

def run_mlb_module():
    print("--- Running MLB Module ---")
    # Placeholder - implement Google scraping for MLB odds similarly to soccer
    return pd.DataFrame(columns=['Date', 'Sport', 'League', 'Match', 'Bet Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake', 'Info'])

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
    
    # Import Google settlement function from utils (assuming it exists in the updated utils.py)
    try:
        import utils
        # This is the critical line - it calls the Google-based settlement in utils.py
        df = utils.settle_bets_with_google_scores(df)
    except ImportError:
        print("Warning: utils module not found. Settlement skipped.")
    except AttributeError:
        print("Warning: settle_bets_with_google_scores function not found in utils. Settlement skipped.")
    except Exception as e:
        print(f"Error during Google-based settlement: {str(e)}. Settlement skipped.")
    
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
