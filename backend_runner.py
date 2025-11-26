# ==============================================================================
# backend_runner.py - "The Unbreakable Engine" (v33.1)
# Features: Genetic AI, Neural Nets, Live US Odds, Arbitrage, Multi-Sport, Archival
# Fixes: NBA Variable Scope Error & NFL Team Mapping
# ==============================================================================
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
import warnings
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

warnings.filterwarnings('ignore')

# ==============================================================================
# 🔐 API CONFIGURATION
# ==============================================================================
API_CONFIG = {
    # PASTE YOUR ODDS API KEY HERE
    "THE_ODDS_API_KEY": "0c5a163c2e9a8c4b6a5d33c56747ecf1", 
    
    # PASTE YOUR DISCORD WEBHOOK HERE
    "DISCORD_WEBHOOK": "PASTE_YOUR_WEBHOOK_HERE" 
}

# ==============================================================================
# 1. THE DARWIN MODULE: GENETIC EVOLUTION ENGINE
# ==============================================================================
GENOME_FILE = 'model_genome.json'

def load_or_initialize_genome():
    if os.path.exists(GENOME_FILE):
        with open(GENOME_FILE, 'r') as f: return json.load(f)
    return {
        'generation': 0, 'best_score': 10.0,
        'xgb_n_estimators': 200, 'xgb_max_depth': 3, 'xgb_learning_rate': 0.1,
        'rf_n_estimators': 200, 'rf_max_depth': 10,
        'nn_hidden_layer_size': 64, 'nn_alpha': 0.0001
    }

def mutate_genome(genome):
    mutant = genome.copy()
    mutation_rate = 0.3
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
    current_genome = load_or_initialize_genome()
    mutant_genome = mutate_genome(current_genome)
    tscv = TimeSeriesSplit(n_splits=3)
    
    champ_model = build_ensemble_from_genome(current_genome)
    champ_scores = cross_val_score(champ_model, X, y, cv=tscv, scoring='neg_log_loss')
    champ_fitness = -champ_scores.mean()
    
    mutant_model = build_ensemble_from_genome(mutant_genome)
    mutant_scores = cross_val_score(mutant_model, X, y, cv=tscv, scoring='neg_log_loss')
    mutant_fitness = -mutant_scores.mean()
    
    print(f"      > Champion Fitness: {champ_fitness:.5f}")
    print(f"      > Mutant Fitness:   {mutant_fitness:.5f}")
    
    if mutant_fitness < champ_fitness:
        print("      > 🚀 EVOLUTION! The Mutant has defeated the Champion.")
        mutant_genome['best_score'] = mutant_fitness
        mutant_genome['generation'] = current_genome['generation'] + 1
        winner_genome = mutant_genome
        with open(GENOME_FILE, 'w') as f: json.dump(winner_genome, f)
    else:
        print("      > 💀 The Mutant failed. The Champion remains.")
        winner_genome = current_genome
        
    final_model = build_ensemble_from_genome(winner_genome)
    calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=3)
    calibrated_model.fit(X, y)
    return calibrated_model

# ==============================================================================
# 2. LIVE ODDS & ARBITRAGE MODULE
# ==============================================================================
def get_live_odds(sport_key):
    if "PASTE_YOUR" in API_CONFIG["THE_ODDS_API_KEY"]: return []
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
    params = {'api_key': API_CONFIG["THE_ODDS_API_KEY"], 'regions': 'us', 'markets': 'h2h,spreads', 'oddsFormat': 'decimal'}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200: return response.json()
        print(f"⚠️ Odds API Error: {response.status_code} - {response.text}")
        return []
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return []

def find_arbitrage(game, sport_type):
    best_home = {'price': 0, 'book': ''}
    best_away = {'price': 0, 'book': ''}
    best_draw = {'price': 0, 'book': ''}
    
    for bookmaker in game['bookmakers']:
        h2h = next((m for m in bookmaker['markets'] if m['key'] == 'h2h'), None)
        if h2h:
            for outcome in h2h['outcomes']:
                price = outcome['price']
                name = outcome['name']
                if name == game['home_team'] and price > best_home['price']:
                    best_home = {'price': price, 'book': bookmaker['title']}
                elif name == game['away_team'] and price > best_away['price']:
                    best_away = {'price': price, 'book': bookmaker['title']}
                elif name == 'Draw' and price > best_draw['price']:
                    best_draw = {'price': price, 'book': bookmaker['title']}

    implied_prob = 0
    arb_info = None
    
    if sport_type == 'Soccer':
        if best_home['price'] > 0 and best_away['price'] > 0 and best_draw['price'] > 0:
            implied_prob = (1/best_home['price']) + (1/best_away['price']) + (1/best_draw['price'])
            if implied_prob < 1.0:
                arb_info = f"Home: {best_home['book']} ({best_home['price']}) | Draw: {best_draw['book']} ({best_draw['price']}) | Away: {best_away['book']} ({best_away['price']})"
    else: 
        if best_home['price'] > 0 and best_away['price'] > 0:
            implied_prob = (1/best_home['price']) + (1/best_away['price'])
            if implied_prob < 1.0:
                arb_info = f"Home: {best_home['book']} ({best_home['price']}) | Away: {best_away['book']} ({best_away['price']})"
    
    if arb_info:
        profit = (1 - implied_prob) / implied_prob
        return profit, arb_info, best_home, best_draw, best_away
    return 0, None, best_home, best_draw, best_away

def fuzzy_match_team(team_name, team_list):
    if not team_list: return None
    match, score = process.extractOne(team_name, team_list)
    if score >= 80: return match
    return None

# ==============================================================================
# 3. SOCCER MODULE
# ==============================================================================
def calculate_soccer_features(df):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    elo_ratings = {team: 1500 for team in teams}
    k_factor = 20
    home_elos, away_elos = [], []
    team_variance = {team: [] for team in teams}
    
    for i, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        r_h, r_a = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500)
        home_elos.append(r_h); away_elos.append(r_a)
        e_h = 1 / (1 + 10**((r_a - r_h) / 400))
        s_h = 1 if row['FTR'] == 'H' else 0 if row['FTR'] == 'A' else 0.5
        error = (s_h - e_h)**2
        team_variance[h].append(error); team_variance[a].append(error)
        elo_ratings[h] += k_factor * (s_h - e_h)
        elo_ratings[a] += k_factor * ((1-s_h) - (1-e_h))

    df['HomeElo'], df['AwayElo'] = home_elos, away_elos
    volatility_map = {t: np.std(v[-10:]) if len(v) > 10 else 0.25 for t, v in team_variance.items()}
    return df, elo_ratings, volatility_map

def train_soccer_brain():
    print("   > Training Soccer Brain (Darwinian)...")
    seasons = ['2324', '2223', '2122']
    df = pd.concat([pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{s}/E0.csv', parse_dates=['Date'], dayfirst=True, on_bad_lines='skip', encoding='latin1') for s in seasons]).sort_values('Date').reset_index(drop=True)
    
    df, elo_ratings, volatility_map = calculate_soccer_features(df)
    
    h_stats = df[['Date', 'HomeTeam', 'FTR']].rename(columns={'HomeTeam': 'Team'})
    h_stats['Points'] = h_stats['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    a_stats = df[['Date', 'AwayTeam', 'FTR']].rename(columns={'AwayTeam': 'Team'})
    a_stats['Points'] = a_stats['FTR'].map({'A': 3, 'D': 1, 'H': 0})
    all_stats = pd.concat([h_stats, a_stats]).sort_values('Date')
    all_stats['Form_EWMA'] = all_stats.groupby('Team')['Points'].transform(lambda x: x.ewm(span=5, adjust=False).mean().shift(1))
    df = pd.merge(df, all_stats[['Date', 'Team', 'Form_EWMA']], left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Form_EWMA': 'HomeForm'})
    df = pd.merge(df, all_stats[['Date', 'Team', 'Form_EWMA']], left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Form_EWMA': 'AwayForm'})
    
    df.dropna(subset=['B365H', 'B365D', 'B365A', 'HomeForm', 'AwayForm'], inplace=True)
    df['elo_diff'] = df['HomeElo'] - df['AwayElo']
    df['form_diff'] = df['HomeForm'] - df['AwayForm']
    
    features = ['elo_diff', 'form_diff']
    X, y = df[features], df['FTR']
    le = LabelEncoder(); y_encoded = le.fit_transform(y)
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    
    living_model = evolve_and_train(X_scaled, y_encoded)
    
    avg_goals_home = df['FTHG'].mean()
    avg_goals_away = df['FTAG'].mean()
    home_strength = df.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'}).rename(columns={'FTHG': 'h_gf_avg', 'FTAG': 'h_ga_avg'})
    away_strength = df.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'}).rename(columns={'FTAG': 'a_gf_avg', 'FTHG': 'a_ga_avg'})
    team_strengths = pd.concat([home_strength, away_strength], axis=1).fillna(1)
    team_strengths['attack'] = (team_strengths['h_gf_avg'] / avg_goals_home + team_strengths['a_gf_avg'] / avg_goals_away) / 2
    team_strengths['defence'] = (team_strengths['h_ga_avg'] / avg_goals_away + team_strengths['a_ga_avg'] / avg_goals_home) / 2
    
    return {'model': living_model, 'le': le, 'scaler': scaler, 'elo_ratings': elo_ratings, 'volatility': volatility_map, 'team_strengths': team_strengths, 'avgs': (avg_goals_home, avg_goals_away)}, df

def run_soccer_module(brain, historical_df):
    print("--- Running Soccer Module (US Pro) ---")
    bets = []
    odds_data = get_live_odds('soccer_epl')
    
    elo_ratings = brain['elo_ratings']
    team_strengths = brain['team_strengths']
    avg_goals_home, avg_goals_away = brain['avgs']
    
    for game in odds_data:
        profit, arb_info, bh, bd, ba = find_arbitrage(game, 'Soccer')
        if profit > 0:
            bets.append({'Sport': 'Soccer', 'League': 'EPL', 'Match': f"{game['home_team']} vs {game['away_team']}", 'Bet Type': 'ARBITRAGE', 'Bet': 'ALL', 'Odds': 0.0, 'Edge': profit, 'Confidence': 1.0, 'Stake': 0.0, 'Info': arb_info})
            continue
            
        model_home = fuzzy_match_team(game['home_team'], list(elo_ratings.keys()))
        model_away = fuzzy_match_team(game['away_team'], list(elo_ratings.keys()))
        
        if model_home and model_away:
            h_elo, a_elo = elo_ratings.get(model_home, 1500), elo_ratings.get(model_away, 1500)
            try:
                h_form = historical_df[historical_df['HomeTeam'] == model_home].sort_values('Date').iloc[-1]['HomeForm']
                a_form = historical_df[historical_df['AwayTeam'] == model_away].sort_values('Date').iloc[-1]['AwayForm']
            except: h_form, a_form = 1.5, 1.5
            
            feat_scaled = brain['scaler'].transform(pd.DataFrame([{'elo_diff': h_elo - a_elo, 'form_diff': h_form - a_form}]))
            probs_alpha = brain['model'].predict_proba(feat_scaled)[0]
            
            try:
                h_att, a_def = team_strengths.loc[model_home]['attack'], team_strengths.loc[model_away]['defence']
                a_att, h_def = team_strengths.loc[model_away]['attack'], team_strengths.loc[model_home]['defence']
                exp_h, exp_a = h_att * a_def * avg_goals_home, a_att * h_def * avg_goals_away
                pm = np.array([[poisson.pmf(i, exp_h) * poisson.pmf(j, exp_a) for j in range(6)] for i in range(6)])
                p_h, p_d, p_a = np.sum(np.tril(pm, -1)), np.sum(np.diag(pm)), np.sum(np.triu(pm, 1))
            except: p_h, p_d, p_a = 0.33, 0.33, 0.33
            
            final_probs = {
                'Home Win': probs_alpha[brain['le'].transform(['H'])[0]] * 0.7 + p_h * 0.3,
                'Draw': probs_alpha[brain['le'].transform(['D'])[0]] * 0.7 + p_d * 0.3,
                'Away Win': probs_alpha[brain['le'].transform(['A'])[0]] * 0.7 + p_a * 0.3
            }
            
            h_vol = brain['volatility'].get(model_home, 0.25)
            a_vol = brain['volatility'].get(model_away, 0.25)
            vol_factor = 1.0 - ((h_vol + a_vol)/2 - 0.25)

            if bh['price'] > 0:
                edge = (final_probs['Home Win'] * bh['price']) - 1
                if edge > 0.05:
                    bets.append({'Sport': 'Soccer', 'League': 'EPL', 'Match': f"{game['home_team']} vs {game['away_team']}", 'Bet Type': 'Moneyline', 'Bet': 'Home Win', 'Odds': bh['price'], 'Edge': edge, 'Confidence': final_probs['Home Win'], 'Stake': (edge/(bh['price']-1))*0.25*vol_factor, 'Info': f"Best: {bh['book']}"})
            if bd['price'] > 0:
                edge = (final_probs['Draw'] * bd['price']) - 1
                if edge > 0.05:
                    bets.append({'Sport': 'Soccer', 'League': 'EPL', 'Match': f"{game['home_team']} vs {game['away_team']}", 'Bet Type': 'Moneyline', 'Bet': 'Draw', 'Odds': bd['price'], 'Edge': edge, 'Confidence': final_probs['Draw'], 'Stake': (edge/(bd['price']-1))*0.25*vol_factor, 'Info': f"Best: {bd['book']}"})
            if ba['price'] > 0:
                edge = (final_probs['Away Win'] * ba['price']) - 1
                if edge > 0.05:
                    bets.append({'Sport': 'Soccer', 'League': 'EPL', 'Match': f"{game['home_team']} vs {game['away_team']}", 'Bet Type': 'Moneyline', 'Bet': 'Away Win', 'Odds': ba['price'], 'Edge': edge, 'Confidence': final_probs['Away Win'], 'Stake': (edge/(ba['price']-1))*0.25*vol_factor, 'Info': f"Best: {ba['book']}"})

    return pd.DataFrame(bets)

# ==============================================================================
# 4. NFL MODULE (US Pro - Fixed)
# ==============================================================================
def run_nfl_module():
    print("--- Running NFL Module (US Pro) ---")
    bets = []
    odds_data = get_live_odds('americanfootball_nfl')
    
    # *** FIX: Define team_map at the top level of the function ***
    team_map = {
        "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
        "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
        "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
        "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
    }
    
    try:
        seasons = [2023, 2022, 2021]
        df = nfl.import_pbp_data(years=seasons, downcast=True, cache=False)
        team_stats = df.groupby(['season', 'home_team']).agg({'yards_gained': 'mean', 'turnover_lost': 'mean'}).reset_index()
        ypp_map = team_stats.groupby('home_team')['yards_gained'].mean().to_dict()
    except: ypp_map = {}
    
    for game in odds_data:
        profit, arb_info, bh, _, ba = find_arbitrage(game, 'NFL')
        if profit > 0:
            bets.append({'Sport': 'NFL', 'League': 'NFL', 'Match': f"{game['away_team']} @ {game['home_team']}", 'Bet Type': 'ARBITRAGE', 'Bet': 'ALL', 'Odds': 0.0, 'Edge': profit, 'Confidence': 1.0, 'Stake': 0.0, 'Info': arb_info})
            continue
            
        # Value Check
        h_abbr = team_map.get(game['home_team'])
        a_abbr = team_map.get(game['away_team'])
        
        if h_abbr and a_abbr:
            h_ypp = ypp_map.get(h_abbr, 5.0)
            a_ypp = ypp_map.get(a_abbr, 5.0)
            pred_margin = (h_ypp - a_ypp) * 7 + 2.5
            
            model_prob_home = 0.50 + (pred_margin / 20)
            
            if bh['price'] > 0:
                edge = (model_prob_home * bh['price']) - 1
                if edge > 0.05:
                    bets.append({'Sport': 'NFL', 'League': 'NFL', 'Match': f"{game['away_team']} @ {game['home_team']}", 'Bet Type': 'Moneyline', 'Bet': 'Home Win', 'Odds': bh['price'], 'Edge': edge, 'Confidence': model_prob_home, 'Stake': (edge/(bh['price']-1))*0.25, 'Info': f"Best: {bh['book']}"})

    return pd.DataFrame(bets)

# ==============================================================================
# 5. NBA MODULE (US Pro - Fixed)
# ==============================================================================
def run_nba_module():
    print("--- Running NBA Module (US Pro) ---")
    bets = []
    odds_data = get_live_odds('basketball_nba')
    
    # *** FIX: Initialize variables BEFORE try block ***
    team_power = {}
    team_names = []
    
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(season="2023-24", measure_type_detailed_defense="Four Factors").get_data_frames()[0]
        stats['EFF'] = (stats['EFG_PCT']*0.4) - (stats['TM_TOV_PCT']*0.25) + (stats['OREB_PCT']*0.2) + (stats['FTA_RATE']*0.15)
        team_power = stats.set_index('TEAM_NAME')['EFF'].to_dict()
        team_names = list(team_power.keys())
    except Exception as e:
        print(f"NBA Stats Error: {e}")
        # team_power and team_names remain empty/default
    
    for game in odds_data:
        profit, arb_info, bh, _, ba = find_arbitrage(game, 'NBA')
        if profit > 0:
            bets.append({'Sport': 'NBA', 'League': 'NBA', 'Match': f"{game['away_team']} @ {game['home_team']}", 'Bet Type': 'ARBITRAGE', 'Bet': 'ALL', 'Odds': 0.0, 'Edge': profit, 'Confidence': 1.0, 'Stake': 0.0, 'Info': arb_info})
            continue
            
        # Value Check with Fuzzy Matching
        # *** FIX: Check if team_names is populated before matching ***
        if not team_names: continue

        model_home = fuzzy_match_team(game['home_team'], team_names)
        model_away = fuzzy_match_team(game['away_team'], team_names)
        
        if model_home and model_away:
            h_eff = team_power.get(model_home, 0.5)
            a_eff = team_power.get(model_away, 0.5)
            pred_margin = (h_eff - a_eff) * 200 + 3
            model_prob_home = 0.50 + (pred_margin / 30)
            
            if bh['price'] > 0:
                edge = (model_prob_home * bh['price']) - 1
                if edge > 0.05:
                    bets.append({'Sport': 'NBA', 'League': 'NBA', 'Match': f"{game['away_team']} @ {game['home_team']}", 'Bet Type': 'Moneyline', 'Bet': 'Home Win', 'Odds': bh['price'], 'Edge': edge, 'Confidence': model_prob_home, 'Stake': (edge/(bh['price']-1))*0.25, 'Info': f"Best: {bh['book']}"})

    return pd.DataFrame(bets)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def run_backend_analysis():
    print("--- Starting Daily Global Backend Analysis (US Pro) ---")
    
    # 1. Train Brains
    soccer_brain, soccer_hist = train_soccer_brain()
    
    # 2. Run Modules
    soccer_bets = run_soccer_module(soccer_brain, soccer_hist)
    nfl_bets = run_nfl_module()
    nba_bets = run_nba_module()
    
    all_bets = pd.concat([soccer_bets, nfl_bets, nba_bets], ignore_index=True)
    
    # 3. Save & Archive
    if not all_bets.empty:
        all_bets['Date_Generated'] = datetime.now().strftime('%Y-%m-%d')
        all_bets.to_csv('latest_bets.csv', index=False)
        
        history_file = 'betting_history.csv'
        if os.path.exists(history_file):
            pd.concat([pd.read_csv(history_file), all_bets]).drop_duplicates(subset=['Date_Generated', 'Match', 'Bet']).to_csv(history_file, index=False)
        else:
            all_bets.to_csv(history_file, index=False)
            
        print(f"\nSuccessfully saved {len(all_bets)} recommendations.")
        
        # Discord Alert
        WEBHOOK_URL = API_CONFIG["DISCORD_WEBHOOK"]
        if "PASTE_YOUR" not in WEBHOOK_URL:
            try:
                msg = f"🚀 **Betting Co-Pilot:** {len(all_bets)} Bets Found!\n"
                arbs = all_bets[all_bets['Bet Type'] == 'ARBITRAGE']
                if not arbs.empty: msg += f"🚨 **{len(arbs)} ARBITRAGE OPPORTUNITIES!**\n"
                requests.post(WEBHOOK_URL, json={"content": msg})
            except: pass
    else:
        print("\nNo value bets found.")
        pd.DataFrame(columns=['Date_Generated', 'Sport', 'League', 'Match', 'Bet Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake', 'Info']).to_csv('latest_bets.csv', index=False)

if __name__ == "__main__":
    run_backend_analysis()
