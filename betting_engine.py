# betting_engine.py
# The Core Logic: AI Models, Data Fetching, Settlement, Feature Engineering, and Advanced AI
# v75.0 - IMPROVEMENT: Enhanced settlement, refined model evolution, better feature engineering,
# focus on learning from losses and improving win rate.
# ANALYSIS INSIGHTS: Addresses low confidence bets, auto-settlements, and potential model drift.

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
import config
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# LOGGING SETUP (Professional standard)
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
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
# 1. SMART MATH MODULES (Full Feature Set Preserved)
# ==============================================================================
def calculate_pythagorean_expectation(points_for, points_against, exponent=1.7):
    if points_for == 0 and points_against == 0: return 0.5
    return (points_for ** exponent) / ((points_for ** exponent) + (points_against ** exponent))

def zero_inflated_poisson(k, lam, pi=0.05):
    if k == 0:
        return pi + (1 - pi) * poisson.pmf(0, lam)
    else:
        return (1 - pi) * poisson.pmf(k, lam)

# ==============================================================================
# 2. ENHANCED GENETIC AI & MODELS (Focus on Learning from Losses)
# ==============================================================================
GENOME_FILE = 'model_genome.json'

def load_or_initialize_genome():
    if os.path.exists(GENOME_FILE):
        with open(GENOME_FILE, 'r') as f: return json.load(f)
    return {
        'generation': 0, 'best_score': 10.0, 'xgb_n_estimators': 200, 'xgb_max_depth': 3, 'xgb_learning_rate': 0.1,
        'rf_n_estimators': 200, 'rf_max_depth': 10, 'nn_hidden_layer_size': 64, 'nn_alpha': 0.0001,
        # New fields for advanced evolution
        'mutation_rate': 0.3, 'selection_pressure': 0.7, 'meta_model_score': 10.0,
        'improvement_count': 0, 'last_update': datetime.now().isoformat()
    }

def mutate_genome(genome):
    """Mutate genome with adaptive rates based on performance and loss analysis."""
    mutant = genome.copy()
    mutation_rate = genome.get('mutation_rate', 0.3)
    selection_pressure = genome.get('selection_pressure', 0.7)
    
    # Adaptive mutation based on improvement stagnation (addresses model drift/loss of skill)
    if genome['generation'] > 5 and genome.get('improvement_count', 0) == 0:
        mutation_rate = min(0.5, mutation_rate * 1.1) # Explore more if stuck
        logger.info(f"Increasing mutation rate to {mutation_rate} due to stagnation.")
    
    # Apply mutations with selection pressure
    if random.random() < mutation_rate:
        # XGBoost mutations
        if random.random() < selection_pressure:
            mutant['xgb_n_estimators'] = int(genome['xgb_n_estimators'] * random.uniform(0.8, 1.2))
        if random.random() < selection_pressure:
            mutant['xgb_learning_rate'] = max(0.001, min(0.3, genome['xgb_learning_rate'] * random.uniform(0.8, 1.2)))
        if random.random() < selection_pressure:
            mutant['xgb_max_depth'] = max(2, min(10, int(genome['xgb_max_depth'] * random.uniform(0.9, 1.1))))
        
        # RF mutations
        if random.random() < selection_pressure:
            mutant['rf_n_estimators'] = int(genome['rf_n_estimators'] * random.uniform(0.8, 1.2))
        if random.random() < selection_pressure:
            mutant['rf_max_depth'] = max(5, min(15, int(genome['rf_max_depth'] * random.uniform(0.9, 1.1))))
        
        # NN mutations
        if random.random() < selection_pressure:
            mutant['nn_hidden_layer_size'] = int(genome['nn_hidden_layer_size'] * random.uniform(0.8, 1.2))
        if random.random() < selection_pressure:
            mutant['nn_alpha'] = max(1e-5, min(0.1, genome['nn_alpha'] * random.uniform(0.8, 1.2)))
    
    # Mutate hyperparameters of the evolutionary process itself
    if random.random() < 0.1: # 10% chance to mutate meta-parameters
        mutant['mutation_rate'] = max(0.1, min(0.5, genome['mutation_rate'] * random.uniform(0.9, 1.1)))
        mutant['selection_pressure'] = max(0.5, min(0.9, genome['selection_pressure'] * random.uniform(0.95, 1.05)))
    
    mutant['last_update'] = datetime.now().isoformat()
    return mutant

def build_ensemble_from_genome(genome):
    """Build model ensemble from genome parameters."""
    # Ensure parameters are within valid ranges
    n_est_xgb = max(50, min(500, int(genome['xgb_n_estimators'])))
    n_est_rf = max(50, min(300, int(genome['rf_n_estimators'])))
    max_depth_xgb = max(2, min(10, int(genome['xgb_max_depth'])))
    max_depth_rf = max(3, min(15, int(genome['rf_max_depth'])))
    nn_size = max(32, min(128, int(genome['nn_hidden_layer_size'])))
    nn_alpha = max(1e-5, min(0.1, float(genome['nn_alpha'])))
    lr_xgb = max(0.01, min(0.3, float(genome['xgb_learning_rate'])))

    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False,
        n_estimators=n_est_xgb, max_depth=max_depth_xgb, learning_rate=lr_xgb,
        random_state=42, n_jobs=-1, tree_method='hist', verbosity=0 # Suppress XGBoost output
    )
    rf_clf = RandomForestClassifier(
        n_estimators=n_est_rf, max_depth=max_depth_rf, random_state=42, n_jobs=-1
    )
    nn_clf = MLPClassifier(
        hidden_layer_sizes=(nn_size, nn_size // 2), alpha=nn_alpha,
        activation='relu', solver='adam', max_iter=500, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=10
    )
    lr_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)

    return VotingClassifier(estimators=[('xgb', xgb_clf), ('rf', rf_clf), ('nn', nn_clf), ('lr', lr_clf)], voting='soft', n_jobs=-1)

def train_meta_model(X, y, primary_model):
    """Train a model to predict the success of the primary model on a sample (Meta-Learning)."""
    try:
        # Get predictions from the primary model
        y_pred_proba = cross_val_predict(primary_model, X, y, cv=3, method='predict_proba')
        y_pred_labels = cross_val_predict(primary_model, X, y, cv=3, method='predict')
        
        # Define meta-target: Did the primary model predict correctly? (1) or incorrectly? (0)
        meta_y = (y_pred_labels == y.astype(int)).astype(int)
        
        # Train meta-learner
        meta_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        meta_clf.fit(y_pred_proba, meta_y)
        
        # Evaluate meta-model accuracy
        meta_score = meta_clf.score(y_pred_proba, meta_y)
        logger.info(f"Meta-model trained. Accuracy: {meta_score:.3f}")
        return meta_clf, meta_score
    except Exception as e:
        logger.error(f"Error training meta-model: {e}")
        return None, 10.0 # Return high error score if it fails

def evolve_and_train(X, y):
    """Evolve and train the model with enhanced evaluation including meta-learning."""
    print("   🧬 Initiating Evolutionary Cycle...")
    current_genome = load_or_initialize_genome()
    mutant_genome = mutate_genome(current_genome)
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Evaluate Primary Model Fitness
    champ_model = build_ensemble_from_genome(current_genome)
    champ_scores = cross_val_score(champ_model, X, y, cv=tscv, scoring='neg_log_loss')
    champ_primary_fitness = -champ_scores.mean() # Lower log loss is better

    mutant_model = build_ensemble_from_genome(mutant_genome)
    mutant_scores = cross_val_score(mutant_model, X, y, cv=tscv, scoring='neg_log_loss')
    mutant_primary_fitness = -mutant_scores.mean()

    # Evaluate Meta-Model Fitness (Learning from losses/successes)
    meta_model_champ, champ_meta_score = train_meta_model(X, y, champ_model)
    meta_model_mutant, mutant_meta_score = train_meta_model(X, y, mutant_model)
    
    # Combine primary and meta fitness for selection
    # A good primary model with a good meta-model (good at predicting its own success) is preferred.
    # We want to minimize loss, so lower combined score is better.
    # For meta-score, lower is better (higher accuracy means lower error rate for the meta model)
    combined_champ_fitness = (champ_primary_fitness + champ_meta_score) / 2
    combined_mutant_fitness = (mutant_primary_fitness + mutant_meta_score) / 2

    if combined_mutant_fitness < combined_champ_fitness: # Mutant wins
        print("      > 🚀 EVOLUTION! Mutant wins based on combined fitness (primary + meta).")
        winner_genome = mutant_genome
        winner_model = mutant_model
        improvement = True
    else: # Champion wins
        print("      > 💀 Champion remains.")
        winner_genome = current_genome
        winner_model = champ_model
        improvement = False

    # Update genome stats
    winner_genome['best_score'] = min(winner_genome['best_score'], 
                                     combined_mutant_fitness if combined_mutant_fitness < combined_champ_fitness else combined_champ_fitness)
    winner_genome['meta_model_score'] = min(winner_genome.get('meta_model_score', 10.0), 
                                           mutant_meta_score if combined_mutant_fitness < combined_champ_fitness else champ_meta_score)
    winner_genome['generation'] += 1
    if improvement:
        winner_genome['improvement_count'] = winner_genome.get('improvement_count', 0) + 1
    else:
        winner_genome['improvement_count'] = 0 # Reset counter on no improvement
    
    with open(GENOME_FILE, 'w') as f: json.dump(winner_genome, f, indent=2)
    
    # Calibrate and return the winning model
    calibrated_model = CalibratedClassifierCV(winner_model, method='isotonic', cv=3)
    calibrated_model.fit(X, y)
    print(f"   > 🏆 Generation {winner_genome['generation']} complete.")
    return calibrated_model, meta_model_champ if not improvement else meta_model_mutant

# ==============================================================================
# 3. DATA FETCHING & ARBITRAGE (Full Feature Set Preserved)
# ==============================================================================
def get_live_odds(sport_key):
    if "PASTE_YOUR" in config.API_CONFIG.get("THE_ODDS_API_KEY", "PASTE_YOUR"):
        print("Odds API key not configured. Skipping live odds fetch.")
        return []
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
    params = {'api_key': config.API_CONFIG["THE_ODDS_API_KEY"], 'regions': 'us', 'markets': 'h2h,spreads', 'oddsFormat': 'decimal'}
    try:
        response = API_SESSION.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching live odds for {sport_key}: {e}")
        return []

def find_arbitrage(game, sport_type):
    best_home = {'price': 0, 'book': ''}; best_away = {'price': 0, 'book': ''}; best_draw = {'price': 0, 'book': ''}
    for bookmaker in game['bookmakers']:
        h2h = next((m for m in bookmaker['markets'] if m['key'] == 'h2h'), None)
        if h2h:
            for outcome in h2h['outcomes']:
                price = outcome['price']; name = outcome['name']
                if name == game['home_team'] and price > best_home['price']: best_home = {'price': price, 'book': bookmaker['title']}
                elif name == game['away_team'] and price > best_away['price']: best_away = {'price': price, 'book': bookmaker['title']}
                elif name == 'Draw' and price > best_draw['price']: best_draw = {'price': price, 'book': bookmaker['title']}
    
    implied_prob = 0; arb_info = None
    if sport_type == 'Soccer':
        if best_home['price'] > 0 and best_away['price'] > 0 and best_draw['price'] > 0:
            implied_prob = (1/best_home['price']) + (1/best_away['price']) + (1/best_draw['price'])
            if implied_prob < 1.0: arb_info = f"Home: {best_home['book']} ({best_home['price']}) | Draw: {best_draw['book']} ({best_draw['price']}) | Away: {best_away['book']} ({best_away['price']})"
    else: # Other sports (2-way)
        if best_home['price'] > 0 and best_away['price'] > 0:
            implied_prob = (1/best_home['price']) + (1/best_away['price'])
            if implied_prob < 1.0: arb_info = f"Home: {best_home['book']} ({best_home['price']}) | Away: {best_away['book']} ({best_away['price']})"
    
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
        res = API_SESSION.get(url, headers=headers, timeout=5)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.find_all('div', {'role': 'heading'})
        for h in headlines[:3]:
            if any(keyword in h.text.lower() for keyword in ['injury', 'doubt', 'out', 'miss', 'sidelined']):
                return f"⚠️ News: {h.text}"
    except requests.exceptions.RequestException as e:
        print(f"Error scraping news for {team1}/{team2}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during news parsing: {e}")
    return None

# ==============================================================================
# 4. ENHANCED GLOBAL SOCCER MODULE (Learning from Losses, Feature Refinement)
# ==============================================================================
def calculate_soccer_features(df):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    elo_ratings = {team: 1500 for team in teams}
    k_factor = 20
    home_elos, away_elos = []
    team_variance = {team: [] for team in teams}
    team_goals_for = {team: 0 for team in teams}
    team_goals_against = {team: 0 for team in teams}
    home_pyth, away_pyth = [], []

    for i, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        
        # Elo
        r_h, r_a = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500)
        home_elos.append(r_h); away_elos.append(r_a)
        e_h = 1 / (1 + 10**((r_a - r_h) / 400))
        s_h = 1 if row['FTR'] == 'H' else 0 if row['FTR'] == 'A' else 0.5
        error = (s_h - e_h)**2
        team_variance[h].append(error); team_variance[a].append(error)
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
    seasons = ['2324', '2223', '2122']
    try:
        df = pd.concat([pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{s}/{div_code}.csv', parse_dates=['Date'], dayfirst=True, on_bad_lines='skip', encoding='latin1') for s in seasons]).sort_values('Date').reset_index(drop=True)
    except Exception as e:
        print(f"Error fetching historical data for league {div_code}: {e}")
        return None, None
        
    if df.empty: return None, None
    
    df, elo_ratings, volatility_map, gf, ga = calculate_soccer_features(df)
    
    # Enhanced feature engineering - Include recent form and performance against specific opponents
    h_stats = df[['Date', 'HomeTeam', 'FTR']].rename(columns={'HomeTeam': 'Team'})
    h_stats['Points'] = h_stats['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    a_stats = df[['Date', 'AwayTeam', 'FTR']].rename(columns={'AwayTeam': 'Team'})
    a_stats['Points'] = a_stats['FTR'].map({'A': 3, 'D': 1, 'H': 0})
    all_stats = pd.concat([h_stats, a_stats]).sort_values('Date')
    all_stats['Form_EWMA'] = all_stats.groupby('Team')['Points'].transform(lambda x: x.ewm(span=5, adjust=False).mean().shift(1))
    # Add opponent-specific stats (e.g., average goals conceded by opponent in last N games)
    # This is complex and requires a separate function or loop. Simplified here.
    df = pd.merge(df, all_stats[['Date', 'Team', 'Form_EWMA']], left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Form_EWMA': 'HomeForm'})
    df = pd.merge(df, all_stats[['Date', 'Team', 'Form_EWMA']], left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Form_EWMA': 'AwayForm'})

    df.dropna(subset=['B365H', 'B365D', 'B365A', 'HomeForm', 'AwayForm'], inplace=True)
    df['elo_diff'] = df['HomeElo'] - df['AwayElo']
    df['form_diff'] = df['HomeForm'] - df['AwayForm']
    df['pyth_diff'] = df['HomePyth'] - df['AwayPyth']
    
    # Add more features based on analysis insight (e.g., volatility as a feature)
    df['HomeVolatility'] = df['HomeTeam'].map(volatility_map)
    df['AwayVolatility'] = df['AwayTeam'].map(volatility_map)
    df['volatility_diff'] = df['HomeVolatility'] - df['AwayVolatility']
    
    features = ['elo_diff', 'form_diff', 'pyth_diff', 'volatility_diff'] # Include new features
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
        odds_data = get_live_odds(sport_key)
        brain = None; historical_df = None
        if div_code != 'UCL': brain, historical_df = train_league_brain(div_code)
        for game in odds_data:
            profit, arb_info, bh, bd, ba = find_arbitrage(game, 'Soccer')
            match_time = game.get('commence_time', 'Unknown')
            if profit > 0: 
                bets.append({'Date': match_time, 'Sport': 'Soccer', 'League': sport_key, 'Match': f"{game['home_team']} vs {game['away_team']}", 'Bet Type': 'ARBITRAGE', 'Bet': 'ALL', 'Odds': 0.0, 'Edge': profit, 'Confidence': 1.0, 'Stake': 0.0, 'Info': arb_info})
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
                    except IndexError: # If team not found in historical data recently, use default
                        h_form, a_form = 1.5, 1.5
                    
                    # Include volatility feature in prediction
                    h_vol = brain['volatility'].get(model_home, 0.25)
                    a_vol = brain['volatility'].get(model_away, 0.25)
                    
                    feat_scaled = brain['scaler'].transform(pd.DataFrame([{
                        'elo_diff': h_elo - a_elo, 'form_diff': h_form - a_form, 'pyth_diff': h_py - a_py, 'volatility_diff': h_vol - a_vol # Include new feature
                    }]))
                    probs_alpha = brain['model'].predict_proba(feat_scaled)[0]
                    if brain['meta_model']:
                        trust_score = brain['meta_model'].predict_proba(feat_scaled)[0][1] # Confidence in primary model
                        if trust_score < 0.55: # If meta-model thinks primary is unreliable, adjust confidence
                            print(f"   > ⚠️ Meta-model confidence low ({trust_score:.2f}) for {game['home_team']} vs {game['away_team']}. Adjusting prediction.")
                            # Apply smoothing or discount edge based on meta-confidence
                            # Example: Reduce edge if low confidence
                            # This is a simple adjustment; a more complex one could involve re-weighting features or using the meta-probability directly.
                            # For now, just log it.
                        else:
                             print(f"   > ✅ Meta-model confidence high ({trust_score:.2f}) for {game['home_team']} vs {game['away_team']}.")
                    
                    try:
                        avg_goals_home, avg_goals_away = brain['avgs']
                        team_strengths = brain['team_strengths']
                        h_att, a_def = team_strengths.loc[model_home]['attack'], team_strengths.loc[model_away]['defence']
                        a_att, h_def = team_strengths.loc[model_away]['attack'], team_strengths.loc[model_home]['defence']
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
                    # Volatility factor remains but now considers team-specific volatility
                    vol_factor = 1.0 - ((h_vol + a_vol)/2 - 0.25) # Adjusted based on actual volatility
                    for outcome, odds_data in [('Home Win', bh), ('Draw', bd), ('Away Win', ba)]:
                        if odds_data['price'] > 0:
                            edge = (final_probs[outcome] * odds_data['price']) - 1
                            # Adjust edge based on meta-model confidence (simplified)
                            adjusted_edge = edge * trust_score if trust_score < 1.0 else edge
                            
                            if adjusted_edge > 0.05: # Use adjusted edge for filtering
                                bets.append({
                                    'Date': match_time,
                                    'Sport': 'Soccer',
                                    'League': sport_key,
                                    'Match': f"{game['home_team']} vs {game['away_team']}",
                                    'Bet Type': 'Moneyline',
                                    'Bet': outcome,
                                    'Odds': odds_data['price'],
                                    'Edge': adjusted_edge, # Store adjusted edge
                                    'Confidence': final_probs[outcome] * trust_score, # Store confidence adjusted by meta-model
                                    'Stake': (adjusted_edge/(odds_data['price']-1))*0.25*vol_factor,
                                    'Info': f"Best: {odds_data['book']}, MetaTrust: {trust_score:.2f}" # Include meta-trust in info
                                })
    return pd.DataFrame(bets)

# Placeholder implementations for other sports (NFL, NBA, MLB) - follow similar pattern to soccer
def run_nfl_module():
    print("--- Running NFL Module ---")
    # Similar logic as run_global_soccer_module, adapted for NFL
    return pd.DataFrame()

def run_nba_module():
    print("--- Running NBA Module ---")
    # Similar logic as run_global_soccer_module, adapted for NBA
    return pd.DataFrame()

def run_mlb_module():
    print("--- Running MLB Module ---")
    # Similar logic as run_global_soccer_module, adapted for MLB
    return pd.DataFrame()

# ==============================================================================
# 5. ENHANCED SETTLEMENT ENGINE (Analysis-Based Improvements)
# ==============================================================================
def settle_bets():
    print("--- ⚖️ Running Settlement Engine ---")
    history_file = 'betting_history.csv'
    if not os.path.exists(history_file): return
    df = pd.read_csv(history_file)
    if 'Result' not in df.columns: df['Result'] = 'Pending'
    if 'Profit' not in df.columns: df['Profit'] = 0.0
    
    # Focus on settling Soccer matches that are clearly finished (analysis showed many 'Auto-Settled' for old matches)
    pending = df[(df['Result'] == 'Pending') & (df['Sport'] == 'Soccer')]
    if not pending.empty:
        try:
            # Fetch results for all relevant leagues (ensure data freshness)
            results = pd.read_csv('https://www.football-data.co.uk/mmz4281/2324/E0.csv', encoding='latin1') # Use appropriate league code based on match
            # Consider fetching from multiple leagues if necessary
            
            # Ensure Date column exists and is in correct format
            if 'Date' not in results.columns:
                 logger.warning("Date column not found in external results file.")
                 return df # Cannot settle without date match

            # Convert dates to UTC and filter out future results (analysis insight: ensure date filtering is robust)
            try:
                results['Date'] = pd.to_datetime(results['Date'], format='%d/%m/%y', errors='coerce', utc=True)
            except ValueError:
                results['Date'] = pd.to_datetime(results['Date'], errors='coerce', utc=True)
            
            # Only consider results for matches that have already occurred
            results = results[results['Date'] <= datetime.now(timezone.utc).normalize()]
            logger.info(f"Fetched {len(results)} historical results for settlement.")

            for idx, row in pending.iterrows():
                try:
                    teams = row['Match'].split(' vs '); home = teams[0].strip(); away = teams[1].strip()
                    
                    # Find the corresponding result by team name and date (analysis insight: match by both name and date)
                    match_date = pd.to_datetime(row['Date'], utc=True).normalize() # Normalize to date only
                    match = results[
                        (results['HomeTeam'] == home) & 
                        (results['AwayTeam'] == away) & 
                        (results['Date'] == match_date) # Match by date
                    ].tail(1) 
                    
                    if not match.empty and pd.notna(match.iloc[0]['FTR']):
                        res = match.iloc[0]['FTR']
                        # Log the outcome for analysis
                        logger.info(f"Settling {row['Match']} ({row['Date']}) as {res} (Score: {match.iloc[0]['FTHG']}-{match.iloc[0]['FTAG']})")
                        
                        # --- Settlement Logic ---
                        won = (row['Bet'] == 'Home Win' and res == 'H') or \
                              (row['Bet'] == 'Draw' and res == 'D') or \
                              (row['Bet'] == 'Away Win' and res == 'A')

                        if won:
                            df.loc[idx, 'Result'] = 'Win'
                            df.loc[idx, 'Profit'] = row['Stake'] * (row['Odds'] - 1)
                        else:
                            df.loc[idx, 'Result'] = 'Loss'
                            df.loc[idx, 'Profit'] = -row['Stake']
                            
                        df.loc[idx, 'Score'] = f"{match.iloc[0]['FTHG']}-{match.iloc[0]['FTAG']}"
                        
                except Exception as e:
                    # Errors during single row processing: keep as pending
                    logger.warning(f"Error processing row {idx} for {row['Match']}: {e}")
                    # Optionally, mark as 'Auto-Settled' with loss after a certain period if needed
                    # Or log for manual review
                    continue 
        except Exception as e:
            # Errors during file fetch/filter: keep all as pending
            logger.error(f"Error fetching/processing external soccer results: {e}")
            # Could implement a fallback to Google Score Lookup here if needed
            pass 
            
    # Save the updated history
    df.to_csv(history_file, index=False)
    logger.info(f"Settlement engine completed. {len(df[df['Result'] != 'Pending'])} bets settled.")

# ==============================================================================
# MODULE EXPORTS (For backend_runner.py)
# ==============================================================================
__all__ = [
    'run_global_soccer_module',
    'run_nfl_module',
    'run_nba_module',
    'run_mlb_module',
    'settle_bets'
]
