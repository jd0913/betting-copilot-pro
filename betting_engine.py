# betting_engine.py
# The Core Logic: AI Models, Data Fetching, Settlement, and Feature Engineering

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
import nfl_data_py as nfl
from nba_api.stats.endpoints import leaguedashteamstats
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import os
import json
import random
from fuzzywuzzy import process
from textblob import TextBlob
import config

# ==============================================================================
# 1. GENETIC AI & MODELS
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
    calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=3)
    calibrated_model.fit(X, y)
    return calibrated_model

# ==============================================================================
# 2. DATA FETCHING & ARBITRAGE
# ==============================================================================
def get_live_odds(sport_key):
    if "PASTE_YOUR" in config.API_CONFIG["THE_ODDS_API_KEY"]: return []
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
    params = {'api_key': config.API_CONFIG["THE_ODDS_API_KEY"], 'regions': 'us', 'markets': 'h2h,spreads', 'oddsFormat': 'decimal'}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200: return response.json()
        return []
    except: return []

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
    else: 
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

# ==============================================================================
# 3. SETTLEMENT ENGINE (Universal)
# ==============================================================================
def settle_bets():
    print("--- ⚖️ Running Universal Settlement Engine ---")
    history_file = 'betting_history.csv'
    if not os.path.exists(history_file): return
    df = pd.read_csv(history_file)
    if 'Result' not in df.columns: df['Result'] = 'Pending'
    if 'Profit' not in df.columns: df['Profit'] = 0.0
    
    pending = df[df['Result'] == 'Pending']
    if pending.empty: return

    # Soccer Settlement
    soccer_pending = pending[pending['Sport'] == 'Soccer']
    if not soccer_pending.empty:
        try:
            results = pd.read_csv('https://www.football-data.co.uk/mmz4281/2324/E0.csv', encoding='latin1')
            for idx, row in soccer_pending.iterrows():
                try:
                    teams = row['Match'].split(' vs '); home = teams[0]; away = teams[1]
                    match = results[(results['HomeTeam'] == home) & (results['AwayTeam'] == away)].tail(1)
                    if not match.empty and pd.notna(match.iloc[0]['FTR']):
                        res = match.iloc[0]['FTR']
                        won = (row['Bet'] == 'Home Win' and res == 'H') or (row['Bet'] == 'Draw' and res == 'D') or (row['Bet'] == 'Away Win' and res == 'A')
                        df.at[idx, 'Result'] = 'Win' if won else 'Loss'
                        df.at[idx, 'Profit'] = row['Stake'] * (row['Odds'] - 1) if won else -row['Stake']
                except: continue
        except: pass

    # NFL Settlement
    nfl_pending = pending[pending['Sport'] == 'NFL']
    if not nfl_pending.empty:
        try:
            games = nfl.import_schedules(years=[2023, 2024])
            finished = games[games['result'].notna()]
            for idx, row in nfl_pending.iterrows():
                try:
                    teams = row['Match'].split(' @ '); away = teams[0]; home = teams[1]
                    game = finished[(finished['home_team'] == home) & (finished['away_team'] == away)].tail(1)
                    if not game.empty:
                        g = game.iloc[0]
                        winner = g['home_team'] if g['home_score'] > g['away_score'] else g['away_team']
                        if 'Moneyline' in row['Bet Type']:
                            bet_team = row['Bet'].replace(' Win', '').replace('Home', home).replace('Away', away).strip()
                            won = (bet_team in winner)
                            df.at[idx, 'Result'] = 'Win' if won else 'Loss'
                            df.at[idx, 'Profit'] = row['Stake'] * (row['Odds'] - 1) if won else -row['Stake']
                except: continue
        except: pass

    df.to_csv(history_file, index=False)
