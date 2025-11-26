# ==============================================================================
# backend_runner.py - "The Living Machine" (v31.0)
# Features: Neural Nets + Genetic Evolution + Recursive Improvement + Multi-Sport
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

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. THE DARWIN MODULE: GENETIC EVOLUTION ENGINE
# ==============================================================================
GENOME_FILE = 'model_genome.json'

def load_or_initialize_genome():
    """Loads the current best 'DNA' or creates a seed."""
    if os.path.exists(GENOME_FILE):
        with open(GENOME_FILE, 'r') as f:
            return json.load(f)
    else:
        # The "Seed AI" configuration
        return {
            'generation': 0,
            'best_score': 10.0,
            # XGBoost Genes
            'xgb_n_estimators': 200,
            'xgb_max_depth': 3,
            'xgb_learning_rate': 0.1,
            # Random Forest Genes
            'rf_n_estimators': 200,
            'rf_max_depth': 10,
            # Neural Network Genes
            'nn_hidden_layer_size': 64,
            'nn_alpha': 0.0001
        }

def mutate_genome(genome):
    """Creates a 'Mutant' by randomly altering the DNA."""
    mutant = genome.copy()
    mutation_rate = 0.3 # 30% chance to mutate
    
    # Mutate XGBoost
    if random.random() < mutation_rate:
        mutant['xgb_n_estimators'] = int(genome['xgb_n_estimators'] * random.uniform(0.8, 1.2))
    if random.random() < mutation_rate:
        mutant['xgb_learning_rate'] = genome['xgb_learning_rate'] * random.uniform(0.8, 1.2)
    
    # Mutate Random Forest
    if random.random() < mutation_rate:
        mutant['rf_n_estimators'] = int(genome['rf_n_estimators'] * random.uniform(0.8, 1.2))
    
    # Mutate Neural Net
    if random.random() < mutation_rate:
        mutant['nn_hidden_layer_size'] = int(genome['nn_hidden_layer_size'] * random.uniform(0.8, 1.2))
        
    return mutant

def build_ensemble_from_genome(genome):
    """Constructs the Voting Classifier using the specific DNA provided."""
    
    # 1. XGBoost
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False,
        n_estimators=int(genome['xgb_n_estimators']),
        max_depth=int(genome['xgb_max_depth']),
        learning_rate=genome['xgb_learning_rate'],
        random_state=42, n_jobs=-1
    )
    
    # 2. Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=int(genome['rf_n_estimators']),
        max_depth=int(genome['rf_max_depth']),
        random_state=42, n_jobs=-1
    )
    
    # 3. Neural Network
    nn_clf = MLPClassifier(
        hidden_layer_sizes=(int(genome['nn_hidden_layer_size']), int(genome['nn_hidden_layer_size'] // 2)),
        alpha=genome['nn_alpha'],
        activation='relu', solver='adam', max_iter=500, random_state=42
    )
    
    # 4. Logistic Regression (Baseline)
    lr_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    
    return VotingClassifier(
        estimators=[('xgb', xgb_clf), ('rf', rf_clf), ('nn', nn_clf), ('lr', lr_clf)],
        voting='soft', n_jobs=-1
    )

def evolve_and_train(X, y):
    """The Core Loop: Mutate -> Test -> Evolve -> Train Final."""
    print("   🧬 Initiating Evolutionary Cycle...")
    current_genome = load_or_initialize_genome()
    mutant_genome = mutate_genome(current_genome)
    
    # Evaluate Fitness (Cross-Validation Log Loss)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Test Champion
    champ_model = build_ensemble_from_genome(current_genome)
    champ_scores = cross_val_score(champ_model, X, y, cv=tscv, scoring='neg_log_loss')
    champ_fitness = -champ_scores.mean()
    
    # Test Mutant
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
        
    # Train final model with winning DNA
    final_model = build_ensemble_from_genome(winner_genome)
    calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=3)
    calibrated_model.fit(X, y)
    
    return calibrated_model

# ==============================================================================
# 2. SOCCER MODULE (Time-Decay + BTTS + Evolution)
# ==============================================================================
def calculate_advanced_features(df):
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

def run_soccer_module():
    print("--- Running Soccer Module (V31.0 Living Machine) ---")
    SOCCER_LEAGUE_CONFIG = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
    all_soccer_bets = []
    
    try:
        fixtures_df = pd.read_csv('https://www.football-data.co.uk/fixtures.csv', encoding='latin1')
        
        for league_div, league_name in SOCCER_LEAGUE_CONFIG.items():
            print(f"   Processing {league_name}...")
            seasons = ['2324', '2223', '2122']
            df = pd.concat([pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{s}/{league_div}.csv', parse_dates=['Date'], dayfirst=True, on_bad_lines='skip', encoding='latin1') for s in seasons]).sort_values('Date').reset_index(drop=True)
            if df.empty: continue
            
            df, elo_ratings, volatility_map = calculate_advanced_features(df)
            df['elo_diff'] = df['HomeElo'] - df['AwayElo']
            
            # Time-Decay Form (EWMA)
            h_stats = df[['Date', 'HomeTeam', 'FTR']].rename(columns={'HomeTeam': 'Team'})
            h_stats['Points'] = h_stats['FTR'].map({'H': 3, 'D': 1, 'A': 0})
            a_stats = df[['Date', 'AwayTeam', 'FTR']].rename(columns={'AwayTeam': 'Team'})
            a_stats['Points'] = a_stats['FTR'].map({'A': 3, 'D': 1, 'H': 0})
            all_stats = pd.concat([h_stats, a_stats]).sort_values('Date')
            all_stats['Form_EWMA'] = all_stats.groupby('Team')['Points'].transform(lambda x: x.ewm(span=5, adjust=False).mean().shift(1))
            df = pd.merge(df, all_stats[['Date', 'Team', 'Form_EWMA']], left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Form_EWMA': 'HomeForm'})
            df = pd.merge(df, all_stats[['Date', 'Team', 'Form_EWMA']], left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Form_EWMA': 'AwayForm'})
            
            df.dropna(subset=['B365H', 'B365D', 'B365A', 'HomeForm', 'AwayForm'], inplace=True)
            df['form_diff'] = df['HomeForm'] - df['AwayForm']
            
            # *** EVOLVE AND TRAIN ***
            features = ['elo_diff', 'form_diff']
            X, y = df[features], df['FTR']
            le = LabelEncoder(); y_encoded = le.fit_transform(y)
            scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
            
            living_model = evolve_and_train(X_scaled, y_encoded)
            
            # Poisson Stats
            avg_goals_home = df['FTHG'].mean()
            avg_goals_away = df['FTAG'].mean()
            home_strength = df.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'}).rename(columns={'FTHG': 'h_gf_avg', 'FTAG': 'h_ga_avg'})
            away_strength = df.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'}).rename(columns={'FTAG': 'a_gf_avg', 'FTHG': 'a_ga_avg'})
            team_strengths = pd.concat([home_strength, away_strength], axis=1).fillna(1)
            team_strengths['attack'] = (team_strengths['h_gf_avg'] / avg_goals_home + team_strengths['a_gf_avg'] / avg_goals_away) / 2
            team_strengths['defence'] = (team_strengths['h_ga_avg'] / avg_goals_away + team_strengths['a_ga_avg'] / avg_goals_home) / 2
            
            league_col = 'Div' if 'Div' in fixtures_df.columns else 'League'
            league_fixtures = fixtures_df[fixtures_df[league_col] == league_div]

            for i, fixture in league_fixtures.iterrows():
                h, a = fixture['HomeTeam'], fixture['AwayTeam']
                
                # 1. Living Model Prediction
                h_elo, a_elo = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500)
                feat_scaled = scaler.transform(pd.DataFrame([{'elo_diff': h_elo - a_elo, 'form_diff': 0}]))
                probs_alpha = living_model.predict_proba(feat_scaled)[0]
                
                # 2. Poisson Prediction
                try:
                    h_att, a_def = team_strengths.loc[h]['attack'], team_strengths.loc[a]['defence']
                    a_att, h_def = team_strengths.loc[a]['attack'], team_strengths.loc[h]['defence']
                    exp_h, exp_a = h_att * a_def * avg_goals_home, a_att * h_def * avg_goals_away
                    
                    p_h_score = 1 - poisson.pmf(0, exp_h)
                    p_a_score = 1 - poisson.pmf(0, exp_a)
                    prob_btts_yes = p_h_score * p_a_score
                    
                    pm = np.array([[poisson.pmf(i, exp_h) * poisson.pmf(j, exp_a) for j in range(6)] for i in range(6)])
                    p_h, p_d, p_a = np.sum(np.tril(pm, -1)), np.sum(np.diag(pm)), np.sum(np.triu(pm, 1))
                except: 
                    p_h, p_d, p_a = 0.33, 0.33, 0.33
                    prob_btts_yes = 0.5
                
                final_probs = {
                    'Home Win': probs_alpha[le.transform(['H'])[0]] * 0.7 + p_h * 0.3,
                    'Draw': probs_alpha[le.transform(['D'])[0]] * 0.7 + p_d * 0.3,
                    'Away Win': probs_alpha[le.transform(['A'])[0]] * 0.7 + p_a * 0.3
                }
                total = sum(final_probs.values())
                final_probs = {k: v/total for k,v in final_probs.items()}

                h_vol = volatility_map.get(h, 0.25)
                a_vol = volatility_map.get(a, 0.25)
                match_vol = (h_vol + a_vol) / 2
                vol_factor = 1.0 - (match_vol - 0.25)

                for outcome, odds_col in [('Home Win', 'B365H'), ('Draw', 'B365D'), ('Away Win', 'B365A')]:
                    if pd.notna(fixture[odds_col]) and fixture[odds_col] > 0:
                        edge = (final_probs[outcome] * fixture[odds_col]) - 1
                        if edge > 0.05:
                            all_soccer_bets.append({'Sport': 'Soccer', 'League': league_name, 'Match': f"{h} vs {a}", 'Bet Type': 'Moneyline', 'Bet': outcome, 'Odds': fixture[odds_col], 'Edge': edge, 'Confidence': final_probs[outcome], 'Vol_Factor': vol_factor})
                
                if prob_btts_yes > 0.60:
                     all_soccer_bets.append({'Sport': 'Soccer', 'League': league_name, 'Match': f"{h} vs {a}", 'Bet Type': 'BTTS', 'Bet': 'Yes', 'Odds': 0.0, 'Edge': 0.0, 'Confidence': prob_btts_yes, 'Vol_Factor': vol_factor})

    except Exception as e:
        print(f"!! SOCCER MODULE ERROR: {e}")
    return pd.DataFrame(all_soccer_bets)

# ==============================================================================
# NFL MODULE (Unchanged)
# ==============================================================================
def run_nfl_module():
    print("--- Running NFL Module ---")
    nfl_bets = []
    try:
        seasons = [2023, 2022, 2021]
        df = nfl.import_pbp_data(years=seasons, downcast=True, cache=False)
        team_stats = df.groupby(['season', 'home_team']).agg({'yards_gained': 'mean', 'turnover_lost': 'mean'}).reset_index()
        team_stats.rename(columns={'home_team': 'team', 'yards_gained': 'ypp_off', 'turnover_lost': 'tov_off'}, inplace=True)
        ypp_map = team_stats.groupby('team')['ypp_off'].mean().to_dict()
        tov_map = team_stats.groupby('team')['tov_off'].mean().to_dict()
        
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
            h_score = ypp_map.get(h, 5.0) - (tov_map.get(h, 1.0) * 4)
            a_score = ypp_map.get(a, 5.0) - (tov_map.get(a, 1.0) * 4)
            pred_margin = (h_score - a_score) * 2 + 2.5
            spread = game_odds['spread_line']
            if pred_margin > (spread * -1) + 1.5:
                edge = pred_margin - (spread * -1)
                nfl_bets.append({'Sport': 'NFL', 'League': 'NFL', 'Match': f"{a} @ {h}", 'Bet Type': 'Point Spread', 'Bet': f"{h} {spread}", 'Odds': 1.91, 'Edge': edge/10, 'Confidence': 0.60, 'Vol_Factor': 1.0})
    except Exception as e:
        print(f"!! NFL MODULE ERROR: {e}")
    return pd.DataFrame(nfl_bets)

# ==============================================================================
# NBA MODULE (Unchanged)
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
                nba_bets.append({'Sport': 'NBA', 'League': 'NBA', 'Match': f"{a} @ {h}", 'Bet Type': 'Point Spread', 'Bet': f"{h} {game['spread']}", 'Odds': 1.91, 'Edge': edge/20, 'Confidence': 0.60, 'Vol_Factor': 1.0})
    except Exception as e:
        print(f"!! NBA MODULE ERROR: {e}")
    return pd.DataFrame(nba_bets)

# ==============================================================================
# NOTIFICATION SYSTEM
# ==============================================================================
def send_discord_alert(df):
    WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL_HERE" 
    if "YOUR_DISCORD_WEBHOOK_URL_HERE" in WEBHOOK_URL: return

    if df.empty:
        msg = "🤖 **Betting Co-Pilot:** No value bets found today."
    else:
        top_bets = df[df['Edge'] > 0.05].sort_values('Edge', ascending=False).head(5)
        msg = f"🚀 **Betting Co-Pilot: {len(df)} Opportunities!**\n\n"
        for i, row in top_bets.iterrows():
            sport_icon = "⚽" if row['Sport'] == "Soccer" else "🏈" if row['Sport'] == "NFL" else "🏀"
            msg += f"{sport_icon} **{row['Match']}**\n   👉 {row['Bet']} @ {row['Odds']:.2f}\n   📈 Edge: {row['Edge']:.2%} | 💰 Stake: {row['Stake']:.2%}\n\n"
    
    try: requests.post(WEBHOOK_URL, json={"content": msg})
    except: pass

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def run_backend_analysis():
    print("--- Starting Daily Global Backend Analysis ---")
    
    soccer_bets = run_soccer_module()
    nfl_bets = run_nfl_module()
    nba_bets = run_nba_module()
    
    all_bets = pd.concat([soccer_bets, nfl_bets, nba_bets], ignore_index=True)
    
    # Add Timestamp
    all_bets['Date_Generated'] = datetime.now().strftime('%Y-%m-%d')
    
    if not all_bets.empty:
        # Kelly Staking with Volatility Targeting
        all_bets['Stake'] = (all_bets['Edge'] / (all_bets['Odds'] - 1)) * 0.25 * all_bets['Vol_Factor']
        all_bets['Stake'] = all_bets['Stake'].clip(lower=0.0, upper=0.05)
        
        # Save Latest
        all_bets.to_csv('latest_bets.csv', index=False)
        
        # Archive History
        history_file = 'betting_history.csv'
        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
            updated_history = pd.concat([history_df, all_bets]).drop_duplicates(subset=['Date_Generated', 'Match', 'Bet'])
            updated_history.to_csv(history_file, index=False)
        else:
            all_bets.to_csv(history_file, index=False)
            
        print(f"\nSuccessfully saved {len(all_bets)} recommendations.")
        send_discord_alert(all_bets)
    else:
        print("\nNo value bets found.")
        pd.DataFrame(columns=['Date_Generated', 'Sport', 'League', 'Match', 'Bet Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake', 'Vol_Factor']).to_csv('latest_bets.csv', index=False)
        send_discord_alert(pd.DataFrame())

if __name__ == "__main__":
    run_backend_analysis()
