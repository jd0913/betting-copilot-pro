# ==============================================================================
# backend_runner.py - The "Darwinian Singularity" Engine (v25.0)
# Features: Recursive Self-Improvement, Genetic Algorithms, Evolutionary Memory
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from scipy.stats import poisson
import joblib
import warnings
import json
import random
import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
import nfl_data_py as nfl
from nba_api.stats.endpoints import leaguedashteamstats

warnings.filterwarnings('ignore')

# ==============================================================================
# THE DARWIN MODULE: RECURSIVE SELF-IMPROVEMENT
# ==============================================================================
GENOME_FILE = 'model_genome.json'

def load_or_initialize_genome():
    """Loads the current best 'DNA' (hyperparameters) or creates a seed."""
    if os.path.exists(GENOME_FILE):
        with open(GENOME_FILE, 'r') as f:
            return json.load(f)
    else:
        # The "Seed AI" configuration
        return {
            'n_estimators': 200,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'generation': 0,
            'best_score': 10.0 # Lower log_loss is better
        }

def mutate_genome(genome):
    """Creates a 'Mutant' by randomly altering the current best DNA."""
    mutant = genome.copy()
    mutation_rate = 0.2 # 20% chance to mutate a gene
    
    if random.random() < mutation_rate:
        mutant['n_estimators'] = int(genome['n_estimators'] * random.uniform(0.8, 1.2))
    if random.random() < mutation_rate:
        mutant['max_depth'] = max(1, int(genome['max_depth'] + random.choice([-1, 1])))
    if random.random() < mutation_rate:
        mutant['learning_rate'] = genome['learning_rate'] * random.uniform(0.8, 1.2)
    if random.random() < mutation_rate:
        mutant['subsample'] = min(1.0, max(0.5, genome['subsample'] * random.uniform(0.9, 1.1)))
    
    return mutant

def evaluate_fitness(model_params, X, y):
    """Tests the DNA. Returns the fitness score (Log Loss)."""
    clf = xgb.XGBClassifier(
        objective='multi:softprob', 
        eval_metric='mlogloss', 
        use_label_encoder=False,
        n_estimators=model_params['n_estimators'],
        max_depth=model_params['max_depth'],
        learning_rate=model_params['learning_rate'],
        subsample=model_params['subsample'],
        colsample_bytree=model_params['colsample_bytree'],
        random_state=42,
        n_jobs=-1
    )
    # TimeSeriesSplit ensures we don't cheat by predicting the past with the future
    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(clf, X, y, cv=tscv, scoring='neg_log_loss')
    return -scores.mean() # Convert neg_log_loss to positive (lower is better)

def evolve_brain(X, y):
    """The Core Loop: Mutate -> Test -> Evolve."""
    print("   🧬 Initiating Evolutionary Cycle...")
    current_genome = load_or_initialize_genome()
    
    # 1. Create a Mutant
    mutant_genome = mutate_genome(current_genome)
    
    # 2. Battle: Champion vs. Mutant
    print(f"      > Champion (Gen {current_genome['generation']}) Fitness: {current_genome['best_score']:.5f}")
    
    # Re-evaluate champion (in case data changed)
    champ_score = evaluate_fitness(current_genome, X, y)
    mutant_score = evaluate_fitness(mutant_genome, X, y)
    
    print(f"      > Mutant Fitness: {mutant_score:.5f}")
    
    # 3. Natural Selection
    if mutant_score < champ_score:
        print("      > 🚀 EVOLUTION! The Mutant has defeated the Champion.")
        mutant_genome['best_score'] = mutant_score
        mutant_genome['generation'] = current_genome['generation'] + 1
        winner = mutant_genome
        
        # Save the new, smarter brain DNA
        with open(GENOME_FILE, 'w') as f:
            json.dump(winner, f)
    else:
        print("      > 💀 The Mutant failed. The Champion remains.")
        winner = current_genome
        # Update champion score if data changed
        winner['best_score'] = champ_score
        with open(GENOME_FILE, 'w') as f:
            json.dump(winner, f)
            
    # Return the best model configuration found so far
    return xgb.XGBClassifier(
        objective='multi:softprob', 
        use_label_encoder=False,
        n_estimators=winner['n_estimators'],
        max_depth=winner['max_depth'],
        learning_rate=winner['learning_rate'],
        subsample=winner['subsample'],
        colsample_bytree=winner['colsample_bytree'],
        random_state=42
    )

# ==============================================================================
# SOCCER MODULE (With Evolution)
# ==============================================================================
def calculate_elo(matches):
    teams = pd.concat([matches['HomeTeam'], matches['AwayTeam']]).unique()
    elo_ratings = {team: 1500 for team in teams}
    home_elos, away_elos = [], []
    for i, row in matches.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        r_h, r_a = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500)
        home_elos.append(r_h); away_elos.append(r_a)
        goal_diff = abs(row['FTHG'] - row['FTAG'])
        k_factor = 20 * (1 + goal_diff / 10)
        e_h = 1 / (1 + 10**((r_a - r_h) / 400))
        s_h = 1 if row['FTR'] == 'H' else 0 if row['FTR'] == 'A' else 0.5
        elo_ratings[h] += k_factor * (s_h - e_h)
        elo_ratings[a] += k_factor * ((1-s_h) - (1-e_h))
    matches['HomeElo'], matches['AwayElo'] = home_elos, away_elos
    return matches, elo_ratings

def run_soccer_module():
    print("--- Running Soccer Module (Darwinian) ---")
    SOCCER_LEAGUE_CONFIG = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
    all_soccer_bets = []
    
    try:
        fixtures_df = pd.read_csv('https://www.football-data.co.uk/fixtures.csv', encoding='latin1')
        
        for league_div, league_name in SOCCER_LEAGUE_CONFIG.items():
            print(f"   Processing {league_name}...")
            seasons = ['2324', '2223', '2122']
            df = pd.concat([pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{s}/{league_div}.csv', parse_dates=['Date'], dayfirst=True, on_bad_lines='skip', encoding='latin1') for s in seasons]).sort_values('Date').reset_index(drop=True)
            if df.empty: continue
            
            df, elo_ratings = calculate_elo(df)
            df['elo_diff'] = df['HomeElo'] - df['AwayElo']
            
            avg_goals_home = df['FTHG'].mean()
            avg_goals_away = df['FTAG'].mean()
            home_strength = df.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'}).rename(columns={'FTHG': 'h_gf_avg', 'FTAG': 'h_ga_avg'})
            away_strength = df.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'}).rename(columns={'FTAG': 'a_gf_avg', 'FTHG': 'a_ga_avg'})
            team_strengths = pd.concat([home_strength, away_strength], axis=1).fillna(1)
            team_strengths['attack'] = (team_strengths['h_gf_avg'] / avg_goals_home + team_strengths['a_gf_avg'] / avg_goals_away) / 2
            team_strengths['defence'] = (team_strengths['h_ga_avg'] / avg_goals_away + team_strengths['a_ga_avg'] / avg_goals_home) / 2
            
            df.dropna(subset=['B365H', 'B365D', 'B365A'], inplace=True)
            features = ['elo_diff']
            X, y = df[features], df['FTR']
            le = LabelEncoder(); y_encoded = le.fit_transform(y)
            scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
            
            # *** EVOLUTIONARY STEP ***
            # The model evolves its hyperparameters here
            best_model = evolve_brain(X_scaled, y_encoded)
            
            calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
            calibrated_model.fit(X_scaled, y_encoded)
            
            league_col = 'Div' if 'Div' in fixtures_df.columns else 'League'
            league_fixtures = fixtures_df[fixtures_df[league_col] == league_div]

            for i, fixture in league_fixtures.iterrows():
                h, a = fixture['HomeTeam'], fixture['AwayTeam']
                h_elo, a_elo = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500)
                feat_scaled = scaler.transform(pd.DataFrame([{'elo_diff': h_elo - a_elo}]))
                probs_alpha = calibrated_model.predict_proba(feat_scaled)[0]
                
                try:
                    h_att, a_def = team_strengths.loc[h]['attack'], team_strengths.loc[a]['defence']
                    a_att, h_def = team_strengths.loc[a]['attack'], team_strengths.loc[h]['defence']
                    exp_h, exp_a = h_att * a_def * avg_goals_home, a_att * h_def * avg_goals_away
                    pm = np.array([[poisson.pmf(i, exp_h) * poisson.pmf(j, exp_a) for j in range(6)] for i in range(6)])
                    p_h, p_d, p_a = np.sum(np.tril(pm, -1)), np.sum(np.diag(pm)), np.sum(np.triu(pm, 1))
                except: p_h, p_d, p_a = 0.33, 0.33, 0.33
                
                final_probs = {
                    'Home Win': probs_alpha[le.transform(['H'])[0]] * 0.6 + p_h * 0.4,
                    'Draw': probs_alpha[le.transform(['D'])[0]] * 0.6 + p_d * 0.4,
                    'Away Win': probs_alpha[le.transform(['A'])[0]] * 0.6 + p_a * 0.4
                }
                total = sum(final_probs.values())
                final_probs = {k: v/total for k,v in final_probs.items()}

                for outcome, odds_col in [('Home Win', 'B365H'), ('Draw', 'B365D'), ('Away Win', 'B365A')]:
                    if pd.notna(fixture[odds_col]) and fixture[odds_col] > 0:
                        edge = (final_probs[outcome] * fixture[odds_col]) - 1
                        if edge > 0.05:
                            all_soccer_bets.append({'Sport': 'Soccer', 'League': league_name, 'Match': f"{h} vs {a}", 'Bet Type': 'Moneyline', 'Bet': outcome, 'Odds': fixture[odds_col], 'Edge': edge, 'Confidence': final_probs[outcome]})
    except Exception as e:
        print(f"!! SOCCER MODULE ERROR: {e}")
    return pd.DataFrame(all_soccer_bets)

# ==============================================================================
# NFL MODULE
# ==============================================================================
def run_nfl_module():
    print("--- Running NFL Module ---")
    nfl_bets = []
    try:
        seasons = [2023, 2022, 2021]
        df = nfl.import_pbp_data(years=seasons, downcast=True, cache=False)
        df['spread_line'] = df['home_score'] - df['away_score']
        home_power = df.groupby('home_team')['spread_line'].mean()
        away_power = df.groupby('away_team')['spread_line'].mean() * -1
        team_power = (home_power + away_power) / 2
        
        schedule = nfl.import_schedules(years=[2024])
        weekly_odds = nfl.import_weekly_data(years=[2024])
        if weekly_odds.empty: return pd.DataFrame()
        next_week = weekly_odds[weekly_odds['spread_line'].notna()]['week'].min()
        if pd.isna(next_week): return pd.DataFrame()
            
        upcoming_games = schedule[schedule['week'] == next_week]
        upcoming_odds = weekly_odds[weekly_odds['week'] == next_week]
        
        for i, game in upcoming_games.iterrows():
            h, a = game['home_team'], game['away_team']
            odds_row = upcoming_odds[(upcoming_odds['home_team'] == h) & (upcoming_odds['away_team'] == a)]
            if odds_row.empty: continue
            game_odds = odds_row.iloc[0]
            h_p, a_p = team_power.get(h, 0), team_power.get(a, 0)
            pred_margin = (h_p - a_p) + 2.5
            spread = game_odds['spread_line']
            if pred_margin > (spread * -1) + 1.5:
                edge = pred_margin - (spread * -1)
                nfl_bets.append({'Sport': 'NFL', 'League': 'NFL', 'Match': f"{a} @ {h}", 'Bet Type': 'Point Spread', 'Bet': f"{h} {spread}", 'Odds': 1.91, 'Edge': edge/10, 'Confidence': 0.60})
    except Exception as e:
        print(f"!! NFL MODULE ERROR: {e}")
    return pd.DataFrame(nfl_bets)

# ==============================================================================
# NBA MODULE
# ==============================================================================
def run_nba_module():
    print("--- Running NBA Module ---")
    nba_bets = []
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(season="2023-24", measure_type_detailed_defense="Four Factors").get_data_frames()[0]
        stats['EFF'] = (stats['EFG_PCT']*0.4) - (stats['TM_TOV_PCT']*0.25) + (stats['OREB_PCT']*0.2) + (stats['FTA_RATE']*0.15)
        team_power = stats.set_index('TEAM_ABBREVIATION')['EFF'].to_dict()
        upcoming_games = [] 
        for game in upcoming_games:
            h, a = game['home'], game['away']
            pred_margin = (team_power.get(h, 0.5) - team_power.get(a, 0.5)) * 200 + 3
            if pred_margin > (game['spread'] * -1) + 2:
                edge = pred_margin - (game['spread'] * -1)
                nba_bets.append({'Sport': 'NBA', 'League': 'NBA', 'Match': f"{a} @ {h}", 'Bet Type': 'Point Spread', 'Bet': f"{h} {game['spread']}", 'Odds': 1.91, 'Edge': edge/20, 'Confidence': 0.60})
    except Exception as e:
        print(f"!! NBA MODULE ERROR: {e}")
    return pd.DataFrame(nba_bets)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def run_backend_analysis():
    print("--- Starting Daily Global Backend Analysis ---")
    pd.DataFrame(columns=['Sport', 'League', 'Match', 'Bet Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake']).to_csv('latest_bets.csv', index=False)
    
    soccer_bets = run_soccer_module()
    nfl_bets = run_nfl_module()
    nba_bets = run_nba_module()
    
    all_bets = pd.concat([soccer_bets, nfl_bets, nba_bets], ignore_index=True)
    
    if not all_bets.empty:
        all_bets['Stake'] = (all_bets['Edge'] / (all_bets['Odds'] - 1)) * 0.25
        all_bets['Stake'] = all_bets['Stake'].clip(lower=0.0, upper=0.05)
        all_bets.to_csv('latest_bets.csv', index=False)
        print(f"\nSuccessfully saved {len(all_bets)} recommendations.")
    else:
        print("\nNo value bets found.")

if __name__ == "__main__":
    run_backend_analysis()
