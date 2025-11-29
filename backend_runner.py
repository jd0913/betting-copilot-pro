# ==============================================================================
# backend_runner.py - "The Fixed Monolith" Engine (v38.1)
# Features: Date Filtering, Prob Capping, Genetic AI, Multi-Sport, Archival
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
from datetime import datetime, timedelta
import os
import json
import random
from fuzzywuzzy import process
from textblob import TextBlob

warnings.filterwarnings('ignore')

# ==============================================================================
# 🔐 API CONFIGURATION
# ==============================================================================
API_CONFIG = {
    "THE_ODDS_API_KEY": "0c5a163c2e9a8c4b6a5d33c56747ecf1", 
    "DISCORD_WEBHOOK": "PASTE_YOUR_WEBHOOK_HERE" 
}

# ==============================================================================
# 1. THE DARWIN MODULE: GENETIC EVOLUTION
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
    current_genome = load_or_initialize_genome(); mutant_genome = mutate_genome(current_genome); tscv = TimeSeriesSplit(n_splits=3)
    champ_model = build_ensemble_from_genome(current_genome); champ_scores = cross_val_score(champ_model, X, y, cv=tscv, scoring='neg_log_loss'); champ_fitness = -champ_scores.mean()
    mutant_model = build_ensemble_from_genome(mutant_genome); mutant_scores = cross_val_score(mutant_model, X, y, cv=tscv, scoring='neg_log_loss'); mutant_fitness = -mutant_scores.mean()
    if mutant_fitness < champ_fitness:
        mutant_genome['best_score'] = mutant_fitness; mutant_genome['generation'] = current_genome['generation'] + 1; winner_genome = mutant_genome
        with open(GENOME_FILE, 'w') as f: json.dump(winner_genome, f)
    else: winner_genome = current_genome
    final_model = build_ensemble_from_genome(winner_genome); calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=3); calibrated_model.fit(X, y)
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

def get_news_alert(team1, team2):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        query = f'"{team1}" OR "{team2}" injury OR doubt OR out'
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=nws"
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.find_all('div', {'role': 'heading'})
        for h in headlines[:3]:
            if any(keyword in h.text.lower() for keyword in ['injury', 'doubt', 'out', 'miss', 'sidelined']):
                return f"⚠️ News: {h.text}"
    except: return None
    return None

# ==============================================================================
# 3. GLOBAL SOCCER MODULE
# ==============================================================================
def calculate_soccer_features(df):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique(); elo_ratings = {team: 1500 for team in teams}; k_factor = 20; home_elos, away_elos = [], []; team_variance = {team: [] for team in teams}
    for i, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']; r_h, r_a = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500); home_elos.append(r_h); away_elos.append(r_a)
        e_h = 1 / (1 + 10**((r_a - r_h) / 400)); s_h = 1 if row['FTR'] == 'H' else 0 if row['FTR'] == 'A' else 0.5
        error = (s_h - e_h)**2; team_variance[h].append(error); team_variance[a].append(error)
        elo_ratings[h] += k_factor * (s_h - e_h); elo_ratings[a] += k_factor * ((1-s_h) - (1-e_h))
    df['HomeElo'], df['AwayElo'] = home_elos, away_elos
    volatility_map = {t: np.std(v[-10:]) if len(v) > 10 else 0.25 for t, v in team_variance.items()}
    return df, elo_ratings, volatility_map

def train_league_brain(div_code):
    seasons = ['2324', '2223', '2122']; 
    try: df = pd.concat([pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{s}/{div_code}.csv', parse_dates=['Date'], dayfirst=True, on_bad_lines='skip', encoding='latin1') for s in seasons]).sort_values('Date').reset_index(drop=True)
    except: return None, None
    if df.empty: return None, None
    df, elo_ratings, volatility_map = calculate_soccer_features(df)
    h_stats = df[['Date', 'HomeTeam', 'FTR']].rename(columns={'HomeTeam': 'Team'}); h_stats['Points'] = h_stats['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    a_stats = df[['Date', 'AwayTeam', 'FTR']].rename(columns={'AwayTeam': 'Team'}); a_stats['Points'] = a_stats['FTR'].map({'A': 3, 'D': 1, 'H': 0})
    all_stats = pd.concat([h_stats, a_stats]).sort_values('Date'); all_stats['Form_EWMA'] = all_stats.groupby('Team')['Points'].transform(lambda x: x.ewm(span=5, adjust=False).mean().shift(1))
    df = pd.merge(df, all_stats[['Date', 'Team', 'Form_EWMA']], left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Form_EWMA': 'HomeForm'})
    df = pd.merge(df, all_stats[['Date', 'Team', 'Form_EWMA']], left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Form_EWMA': 'AwayForm'})
    df.dropna(subset=['B365H', 'B365D', 'B365A', 'HomeForm', 'AwayForm'], inplace=True)
    df['elo_diff'] = df['HomeElo'] - df['AwayElo']; df['form_diff'] = df['HomeForm'] - df['AwayForm']
    features = ['elo_diff', 'form_diff']; X, y = df[features], df['FTR']
    le = LabelEncoder(); y_encoded = le.fit_transform(y); scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    living_model = evolve_and_train(X_scaled, y_encoded)
    avg_goals_home = df['FTHG'].mean(); avg_goals_away = df['FTAG'].mean()
    home_strength = df.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'}).rename(columns={'FTHG': 'h_gf_avg', 'FTAG': 'h_ga_avg'})
    away_strength = df.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'}).rename(columns={'FTAG': 'a_gf_avg', 'FTHG': 'a_ga_avg'})
    team_strengths = pd.concat([home_strength, away_strength], axis=1).fillna(1)
    team_strengths['attack'] = (team_strengths['h_gf_avg'] / avg_goals_home + team_strengths['a_gf_avg'] / avg_goals_away) / 2
    team_strengths['defence'] = (team_strengths['h_ga_avg'] / avg_goals_away + team_strengths['a_ga_avg'] / avg_goals_home) / 2
    return {'model': living_model, 'le': le, 'scaler': scaler, 'elo_ratings': elo_ratings, 'volatility': volatility_map, 'team_strengths': team_strengths, 'avgs': (avg_goals_home, avg_goals_away)}, df

def run_global_soccer_module():
    print("--- Running Global Soccer Module (Big 5 + UCL) ---"); bets = []
    LEAGUE_MAP = {'soccer_epl': 'E0', 'soccer_spain_la_liga': 'SP1', 'soccer_germany_bundesliga': 'D1', 'soccer_italy_serie_a': 'I1', 'soccer_france_ligue_one': 'F1', 'soccer_uefa_champs_league': 'UCL'}
    for sport_key, div_code in LEAGUE_MAP.items():
        print(f"   > Scanning {sport_key}...")
        odds_data = get_live_odds(sport_key)
        brain = None; historical_df = None
        if div_code != 'UCL': brain, historical_df = train_league_brain(div_code)
        for game in odds_data:
            profit, arb_info, bh, bd, ba = find_arbitrage(game, 'Soccer')
            match_time = game.get('commence_time', 'Unknown')
            if profit > 0: bets.append({'Date': match_time, 'Sport': 'Soccer', 'League': sport_key, 'Match': f"{game['home_team']} vs {game['away_team']}", 'Bet Type': 'ARBITRAGE', 'Bet': 'ALL', 'Odds': 0.0, 'Edge': profit, 'Confidence': 1.0, 'Stake': 0.0, 'Info': arb_info}); continue
            if brain and historical_df is not None:
                model_home = fuzzy_match_team(game['home_team'], list(brain['elo_ratings'].keys())); model_away = fuzzy_match_team(game['away_team'], list(brain['elo_ratings'].keys()))
                if model_home and model_away:
                    h_elo, a_elo = brain['elo_ratings'].get(model_home, 1500), brain['elo_ratings'].get(model_away, 1500)
                    try: h_form = historical_df[historical_df['HomeTeam'] == model_home].sort_values('Date').iloc[-1]['HomeForm']; a_form = historical_df[historical_df['AwayTeam'] == model_away].sort_values('Date').iloc[-1]['AwayForm']
                    except: h_form, a_form = 1.5, 1.5
                    feat_scaled = brain['scaler'].transform(pd.DataFrame([{'elo_diff': h_elo - a_elo, 'form_diff': h_form - a_form}]))
                    probs_alpha = brain['model'].predict_proba(feat_scaled)[0]
                    try:
                        avg_goals_home, avg_goals_away = brain['avgs']; team_strengths = brain['team_strengths']
                        h_att, a_def = team_strengths.loc[model_home]['attack'], team_strengths.loc[model_away]['defence']; a_att, h_def = team_strengths.loc[model_away]['attack'], team_strengths.loc[model_home]['defence']
                        exp_h, exp_a = h_att * a_def * avg_goals_home, a_att * h_def * avg_goals_away
                        pm = np.array([[poisson.pmf(i, exp_h) * poisson.pmf(j, exp_a) for j in range(6)] for i in range(6)])
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
# 4. NFL MODULE (Fixed: Date Filter & Prob Capping)
# ==============================================================================
def run_nfl_module():
    print("--- Running NFL Module (US Pro) ---")
    bets = []
    odds_data = get_live_odds('americanfootball_nfl')
    
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
        
        # *** FIX: Filter Schedule for Next 7 Days Only ***
        schedule = nfl.import_schedules(years=[2024, 2025])
        today = pd.Timestamp.now().normalize()
        next_week = today + pd.Timedelta(days=7)
        schedule['gameday'] = pd.to_datetime(schedule['gameday'])
        upcoming_games = schedule[(schedule['gameday'] >= today) & (schedule['gameday'] <= next_week)]
        
    except: ypp_map = {}; upcoming_games = pd.DataFrame()
    
    for game in odds_data:
        profit, arb_info, bh, _, ba = find_arbitrage(game, 'NFL')
        match_time = game.get('commence_time', 'Unknown')
        
        # Skip if match is too far in future
        try:
            game_date = datetime.strptime(match_time, "%Y-%m-%dT%H:%M:%SZ")
            if (game_date - datetime.now()).days > 7: continue
        except: pass

        if profit > 0:
            bets.append({'Date': match_time, 'Sport': 'NFL', 'League': 'NFL', 'Match': f"{game['away_team']} @ {game['home_team']}", 'Bet Type': 'ARBITRAGE', 'Bet': 'ALL', 'Odds': 0.0, 'Edge': profit, 'Confidence': 1.0, 'Stake': 0.0, 'Info': arb_info})
            continue
            
        h_abbr = team_map.get(game['home_team'])
        a_abbr = team_map.get(game['away_team'])
        
        if h_abbr and a_abbr:
            h_ypp = ypp_map.get(h_abbr, 5.0)
            a_ypp = ypp_map.get(a_abbr, 5.0)
            pred_margin = (h_ypp - a_ypp) * 7 + 2.5
            model_prob_home = 0.50 + (pred_margin / 20)
            
            # Cap probability
            model_prob_home = max(0.10, min(0.90, model_prob_home))
            
            best_home = {'price': 0, 'book': ''}
            for bookmaker in game['bookmakers']:
                h2h = next((m for m in bookmaker['markets'] if m['key'] == 'h2h'), None)
                if h2h:
                    for outcome in h2h['outcomes']:
                        if outcome['name'] == game['home_team'] and outcome['price'] > best_home['price']: best_home = {'price': outcome['price'], 'book': bookmaker['title']}
            
            if best_home['price'] > 0:
                edge = (model_prob_home * best_home['price']) - 1
                if edge > 0.05:
                    bets.append({'Date': match_time, 'Sport': 'NFL', 'League': 'NFL', 'Match': f"{game['away_team']} @ {game['home_team']}", 'Bet Type': 'Moneyline', 'Bet': 'Home Win', 'Odds': best_home['price'], 'Edge': edge, 'Confidence': model_prob_home, 'Stake': (edge/(best_home['price']-1))*0.25, 'Info': f"Best: {best_home['book']}"})

    return pd.DataFrame(bets)

# ==============================================================================
# 5. NBA MODULE (Fixed: Date Filtering)
# ==============================================================================
def run_nba_module():
    print("--- Running NBA Module (US Pro) ---")
    bets = []
    odds_data = get_live_odds('basketball_nba')
    
    team_power = {}
    team_names = []
    
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(season="2024-25", measure_type_detailed_defense="Four Factors").get_data_frames()[0]
        stats['EFF'] = (stats['EFG_PCT']*0.4) - (stats['TM_TOV_PCT']*0.25) + (stats['OREB_PCT']*0.2) + (stats['FTA_RATE']*0.15)
        team_power = stats.set_index('TEAM_NAME')['EFF'].to_dict()
        team_names = list(team_power.keys())
    except: pass
    
    for game in odds_data:
        # *** FIX: Date Filter ***
        match_time = game.get('commence_time', 'Unknown')
        try:
            game_date = datetime.strptime(match_time, "%Y-%m-%dT%H:%M:%SZ")
            if (game_date - datetime.now()).days > 3: continue 
        except: pass

        profit, arb_info, bh, _, ba = find_arbitrage(game, 'NBA')
        if profit > 0:
            bets.append({'Date': match_time, 'Sport': 'NBA', 'League': 'NBA', 'Match': f"{game['away_team']} @ {game['home_team']}", 'Bet Type': 'ARBITRAGE', 'Bet': 'ALL', 'Odds': 0.0, 'Edge': profit, 'Confidence': 1.0, 'Stake': 0.0, 'Info': arb_info})
            continue
            
        if not team_names: continue

        model_home = fuzzy_match_team(game['home_team'], team_names)
        model_away = fuzzy_match_team(game['away_team'], team_names)
        
        if model_home and model_away:
            h_eff = team_power.get(model_home, 0.5)
            a_eff = team_power.get(model_away, 0.5)
            pred_margin = (h_eff - a_eff) * 200 + 3
            model_prob_home = 0.50 + (pred_margin / 30)
            
            # Cap probability
            model_prob_home = max(0.15, min(0.85, model_prob_home))
            
            best_home = {'price': 0, 'book': ''}
            for bookmaker in game['bookmakers']:
                h2h = next((m for m in bookmaker['markets'] if m['key'] == 'h2h'), None)
                if h2h:
                    for outcome in h2h['outcomes']:
                        if outcome['name'] == game['home_team'] and outcome['price'] > best_home['price']: best_home = {'price': outcome['price'], 'book': bookmaker['title']}
            
            if best_home['price'] > 0:
                edge = (model_prob_home * best_home['price']) - 1
                if edge > 0.05:
                    bets.append({'Date': match_time, 'Sport': 'NBA', 'League': 'NBA', 'Match': f"{game['away_team']} @ {game['home_team']}", 'Bet Type': 'Moneyline', 'Bet': 'Home Win', 'Odds': best_home['price'], 'Edge': edge, 'Confidence': model_prob_home, 'Stake': (edge/(best_home['price']-1))*0.25, 'Info': f"Best: {best_home['book']}"})

    return pd.DataFrame(bets)

# ==============================================================================
# 6. MLB MODULE
# ==============================================================================
def run_mlb_module():
    print("--- Running MLB Module (Moneyball) ---"); bets = []; odds_data = get_live_odds('baseball_mlb')
    for game in odds_data:
        profit, arb_info, bh, _, ba = find_arbitrage(game, 'MLB'); match_time = game.get('commence_time', 'Unknown')
        if profit > 0: bets.append({'Date': match_time, 'Sport': 'MLB', 'League': 'MLB', 'Match': f"{game['away_team']} @ {game['home_team']}", 'Bet Type': 'ARBITRAGE', 'Bet': 'ALL', 'Odds': 0.0, 'Edge': profit, 'Confidence': 1.0, 'Stake': 0.0, 'Info': arb_info}); continue
        best_home = {'price': 0, 'book': ''}
        for bookmaker in game['bookmakers']:
            h2h = next((m for m in bookmaker['markets'] if m['key'] == 'h2h'), None)
            if h2h:
                for outcome in h2h['outcomes']:
                    if outcome['name'] == game['home_team'] and outcome['price'] > best_home['price']: best_home = {'price': outcome['price'], 'book': bookmaker['title']}
        if best_home['price'] > 2.10: bets.append({'Date': match_time, 'Sport': 'MLB', 'League': 'MLB', 'Match': f"{game['away_team']} @ {game['home_team']}", 'Bet Type': 'Moneyline', 'Bet': 'Home Win', 'Odds': best_home['price'], 'Edge': 0.05, 'Confidence': 0.50, 'Stake': 0.01, 'Info': f"Best: {best_home['book']}"})
    return pd.DataFrame(bets)

# ==============================================================================
# 7. SETTLEMENT ENGINE
# ==============================================================================
def settle_bets():
    print("--- ⚖️ Running Settlement Engine ---")
    history_file = 'betting_history.csv'
    if not os.path.exists(history_file): return
    df = pd.read_csv(history_file)
    if 'Result' not in df.columns: df['Result'] = 'Pending'
    if 'Profit' not in df.columns: df['Profit'] = 0.0
    if 'Date' not in df.columns: df['Date'] = datetime.now().strftime('%Y-%m-%d')
    
    pending = df[df['Result'] == 'Pending']
    if pending.empty: return

    now = datetime.utcnow()

    # Soccer Settlement
    soccer_pending = pending[pending['Sport'] == 'Soccer']
    if not soccer_pending.empty:
        try:
            results = pd.read_csv('https://www.football-data.co.uk/mmz4281/2324/E0.csv', encoding='latin1')
            results['Date'] = pd.to_datetime(results['Date'], dayfirst=True)
            for idx, row in soccer_pending.iterrows():
                try:
                    match_date = pd.to_datetime(row['Date']).replace(tzinfo=None)
                    if match_date > now: continue 
                    teams = row['Match'].split(' vs '); home = teams[0]; away = teams[1]
                    match = results[(results['HomeTeam'] == home) & (results['AwayTeam'] == away) & (results['Date'] >= match_date - timedelta(days=1))].tail(1)
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
                    match_date = pd.to_datetime(row['Date']).replace(tzinfo=None)
                    if match_date > now: continue
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

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def run_backend_analysis():
    print("--- Starting Daily Global Backend Analysis (US Pro) ---")
    settle_bets()
    
    soccer_bets = run_global_soccer_module()
    nfl_bets = run_nfl_module()
    nba_bets = run_nba_module()
    mlb_bets = run_mlb_module()
    
    all_bets = pd.concat([soccer_bets, nfl_bets, nba_bets, mlb_bets], ignore_index=True)
    
    if not all_bets.empty:
        all_bets['Date_Generated'] = datetime.now().strftime('%Y-%m-%d')
        all_bets.to_csv('latest_bets.csv', index=False)
        
        history_file = 'betting_history.csv'
        if os.path.exists(history_file):
            pd.concat([pd.read_csv(history_file), all_bets]).drop_duplicates(subset=['Date_Generated', 'Match', 'Bet']).to_csv(history_file, index=False)
        else:
            all_bets.to_csv(history_file, index=False)
            
        print(f"\nSuccessfully saved {len(all_bets)} recommendations.")
        
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
        pd.DataFrame(columns=['Date', 'Date_Generated', 'Sport', 'League', 'Match', 'Bet Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake', 'Info']).to_csv('latest_bets.csv', index=False)

if __name__ == "__main__":
    run_backend_analysis()
