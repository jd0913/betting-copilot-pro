# ==============================================================================
# backend_runner.py - The "God Tier" Automated Backend (v22.0)
# Features: Auto-Tuning, Four Factors (NBA), YPP (NFL), Rest Days (Soccer)
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import poisson
import joblib
import warnings
import nfl_data_py as nfl
from nba_api.stats.endpoints import leaguedashteamstats
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ==============================================================================
# HELPER: Automated Hyperparameter Tuning
# ==============================================================================
def tune_xgboost(X, y, objective='multi:softprob', metric='mlogloss'):
    """
    Automatically finds the best hyperparameters for XGBoost using Randomized Search.
    """
    print("   > Tuning model parameters...")
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    clf = xgb.XGBClassifier(objective=objective, eval_metric=metric, use_label_encoder=False)
    # Use TimeSeriesSplit to prevent data leakage (training on future to predict past)
    cv = TimeSeriesSplit(n_splits=3)
    
    search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, scoring='neg_log_loss', cv=cv, verbose=0, n_jobs=-1, random_state=42)
    search.fit(X, y)
    
    print(f"   > Best Params: {search.best_params_}")
    return search.best_estimator_

# ==============================================================================
# MODULE 1: SOCCER (Elo + Poisson + Rest Days + Shot Dominance)
# ==============================================================================
def calculate_soccer_features(df):
    # 1. Elo Ratings
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    elo_ratings = {team: 1500 for team in teams}
    k_factor = 20
    home_elos, away_elos = [], []
    
    # 2. Rest Days Calculation
    last_match_date = {team: pd.to_datetime('2000-01-01') for team in teams}
    home_rest, away_rest = [], []

    for i, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        match_date = row['Date']
        
        # Elo
        r_h, r_a = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500)
        home_elos.append(r_h); away_elos.append(r_a)
        e_h = 1 / (1 + 10**((r_a - r_h) / 400))
        s_h = 1 if row['FTR'] == 'H' else 0 if row['FTR'] == 'A' else 0.5
        elo_ratings[h] += k_factor * (s_h - e_h)
        elo_ratings[a] += k_factor * ((1-s_h) - (1-e_h))
        
        # Rest Days
        h_last, a_last = last_match_date.get(h), last_match_date.get(a)
        home_rest.append((match_date - h_last).days if h_last else 7)
        away_rest.append((match_date - a_last).days if a_last else 7)
        last_match_date[h] = match_date
        last_match_date[a] = match_date

    df['HomeElo'], df['AwayElo'] = home_elos, away_elos
    df['HomeRest'], df['AwayRest'] = home_rest, away_rest
    
    # 3. Shot Dominance (Proxy for xG)
    # HST = Home Shots on Target, AST = Away Shots on Target
    df['HomeShotDom'] = df.groupby('HomeTeam')['HST'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['AwayShotDom'] = df.groupby('AwayTeam')['AST'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    return df, elo_ratings

def run_soccer_module():
    print("--- Running Soccer Module (V22.0) ---")
    SOCCER_LEAGUE_CONFIG = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
    all_soccer_bets = []
    
    try:
        fixtures_df = pd.read_csv('https://www.football-data.co.uk/fixtures.csv', encoding='latin1')
        
        for league_div, league_name in SOCCER_LEAGUE_CONFIG.items():
            print(f"   Processing {league_name}...")
            seasons = ['2324', '2223', '2122']
            df = pd.concat([pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{s}/{league_div}.csv', parse_dates=['Date'], dayfirst=True, on_bad_lines='skip', encoding='latin1') for s in seasons]).sort_values('Date').reset_index(drop=True)
            if df.empty: continue
            
            # Feature Engineering
            df, elo_ratings = calculate_soccer_features(df)
            df.dropna(subset=['B365H', 'B365D', 'B365A', 'HomeShotDom', 'AwayShotDom'], inplace=True)
            
            df['elo_diff'] = df['HomeElo'] - df['AwayElo']
            df['rest_diff'] = df['HomeRest'] - df['AwayRest']
            df['shot_diff'] = df['HomeShotDom'] - df['AwayShotDom']
            
            # Train Tuned Model
            features = ['elo_diff', 'rest_diff', 'shot_diff']
            X, y = df[features], df['FTR']
            le = LabelEncoder(); y_encoded = le.fit_transform(y)
            scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
            
            # Auto-Tune XGBoost
            best_model = tune_xgboost(X_scaled, y_encoded)
            calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
            calibrated_model.fit(X_scaled, y_encoded)
            
            # Poisson Stats
            avg_goals_home = df['FTHG'].mean()
            avg_goals_away = df['FTAG'].mean()
            home_strength = df.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'}).rename(columns={'FTHG': 'h_gf_avg', 'FTAG': 'h_ga_avg'})
            away_strength = df.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'}).rename(columns={'FTAG': 'a_gf_avg', 'FTHG': 'a_ga_avg'})
            
            # Predict Live Fixtures
            league_col = 'Div' if 'Div' in fixtures_df.columns else 'League'
            league_fixtures = fixtures_df[fixtures_df[league_col] == league_div]

            for i, fixture in league_fixtures.iterrows():
                h, a = fixture['HomeTeam'], fixture['AwayTeam']
                
                # 1. Model Alpha Prediction
                h_elo, a_elo = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500)
                # Assume 7 days rest for future games as default
                feat_row = pd.DataFrame([{'elo_diff': h_elo - a_elo, 'rest_diff': 0, 'shot_diff': 0}]) 
                # (Note: In a full prod system, we'd look up recent shot stats for the live teams here)
                
                feat_scaled = scaler.transform(feat_row)
                probs_alpha = calibrated_model.predict_proba(feat_scaled)[0]
                
                # 2. Model Bravo (Poisson) Prediction
                try:
                    h_att = (home_strength.loc[h]['h_gf_avg'] / avg_goals_home + away_strength.loc[h]['a_gf_avg'] / avg_goals_away) / 2
                    a_def = (home_strength.loc[a]['h_ga_avg'] / avg_goals_away + away_strength.loc[a]['a_ga_avg'] / avg_goals_home) / 2
                    a_att = (home_strength.loc[a]['h_gf_avg'] / avg_goals_home + away_strength.loc[a]['a_gf_avg'] / avg_goals_away) / 2
                    h_def = (home_strength.loc[h]['h_ga_avg'] / avg_goals_away + away_strength.loc[h]['a_ga_avg'] / avg_goals_home) / 2
                    
                    exp_h = h_att * a_def * avg_goals_home
                    exp_a = a_att * h_def * avg_goals_away
                    
                    pm = np.array([[poisson.pmf(i, exp_h) * poisson.pmf(j, exp_a) for j in range(6)] for i in range(6)])
                    p_h, p_d, p_a = np.sum(np.tril(pm, -1)), np.sum(np.diag(pm)), np.sum(np.triu(pm, 1))
                except:
                    p_h, p_d, p_a = 0.33, 0.33, 0.33

                # Ensemble
                final_probs = {
                    'Home Win': probs_alpha[le.transform(['H'])[0]] * 0.6 + p_h * 0.4,
                    'Draw': probs_alpha[le.transform(['D'])[0]] * 0.6 + p_d * 0.4,
                    'Away Win': probs_alpha[le.transform(['A'])[0]] * 0.6 + p_a * 0.4
                }
                
                # Normalize
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
# MODULE 2: NFL (Yards Per Play + Turnovers + Tuning)
# ==============================================================================
def run_nfl_module():
    print("--- Running NFL Module (V22.0) ---")
    nfl_bets = []
    try:
        seasons = [2023, 2022, 2021]
        df = nfl.import_pbp_data(years=seasons, downcast=True, cache=False)
        
        # Advanced Feature Engineering
        # Calculate Yards Per Play (YPP) and Turnover Differential
        # This is a simplified aggregation for the example
        team_stats = df.groupby(['season', 'home_team']).agg({'yards_gained': 'mean', 'turnover_lost': 'mean'}).reset_index()
        team_stats.rename(columns={'home_team': 'team', 'yards_gained': 'ypp_off', 'turnover_lost': 'tov_off'}, inplace=True)
        
        # Create a simple power rating based on YPP
        # (In a full system, you'd merge this back to every game)
        power_map = team_stats.groupby('team')['ypp_off'].mean().to_dict()
        
        # Schedule & Odds
        schedule = nfl.import_schedules(years=[2024])
        weekly_odds = nfl.import_weekly_data(years=[2024])
        
        if weekly_odds.empty: return pd.DataFrame()
        
        next_week = weekly_odds[weekly_odds['spread_line'].notna()]['week'].min()
        if pd.isna(next_week): return pd.DataFrame()
            
        upcoming_games = schedule[schedule['week'] == next_week]
        upcoming_odds = weekly_odds[weekly_odds['week'] == next_week]
        
        for i, game in upcoming_games.iterrows():
            h, a = game['home_team'], game['away_team']
            # Map team names if necessary (NFL data usually consistent)
            
            odds_row = upcoming_odds[(upcoming_odds['home_team'] == h) & (upcoming_odds['away_team'] == a)]
            if odds_row.empty: continue
            game_odds = odds_row.iloc[0]
            
            # Prediction Logic: Compare YPP
            h_ypp = power_map.get(h, 5.0)
            a_ypp = power_map.get(a, 5.0)
            
            # Simple heuristic: 1 yard per play diff ~= 7 points
            pred_margin = (h_ypp - a_ypp) * 7 + 2.5 # +2.5 for home field
            
            spread_line = game_odds['spread_line']
            
            if pred_margin > (spread_line * -1) + 1.5: # 1.5 point buffer
                edge = pred_margin - (spread_line * -1)
                nfl_bets.append({'Sport': 'NFL', 'League': 'NFL', 'Match': f"{a} @ {h}", 'Bet Type': 'Point Spread', 'Bet': f"{h} {spread_line}", 'Odds': 1.91, 'Edge': edge/10, 'Confidence': 0.60}) # Normalized edge
                
    except Exception as e:
        print(f"!! NFL MODULE ERROR: {e}")
    return pd.DataFrame(nfl_bets)

# ==============================================================================
# MODULE 3: NBA (Four Factors Model)
# ==============================================================================
def run_nba_module():
    print("--- Running NBA Module (V22.0) ---")
    nba_bets = []
    try:
        # Fetch Four Factors
        # eFG%, TOV%, ORB%, FT Rate
        stats = leaguedashteamstats.LeagueDashTeamStats(season="2023-24", measure_type_detailed_defense="Four Factors").get_data_frames()[0]
        
        # Create an Efficiency Rating (Weighted Four Factors)
        # Weights: eFG (40%), TOV (25%), ORB (20%), FT (15%)
        stats['EFF_RATING'] = (stats['EFG_PCT']*0.4) - (stats['TM_TOV_PCT']*0.25) + (stats['OREB_PCT']*0.2) + (stats['FTA_RATE']*0.15)
        
        team_power = stats.set_index('TEAM_ABBREVIATION')['EFF_RATING'].to_dict()
        
        # Placeholder for live schedule scraping (NBA API schedule endpoint is complex)
        # We will simulate one game to show the logic works
        upcoming_games = [{'home': 'BOS', 'away': 'LAL', 'spread': -6.5}]
        
        for game in upcoming_games:
            h, a = game['home'], game['away']
            h_eff = team_power.get(h, 0.5)
            a_eff = team_power.get(a, 0.5)
            
            # Heuristic: 0.01 diff in Eff Rating ~= 2 points
            pred_margin = (h_eff - a_eff) * 200 + 3 # +3 Home Court
            
            if pred_margin > (game['spread'] * -1):
                edge = pred_margin - (game['spread'] * -1)
                if edge > 2.0:
                    nba_bets.append({'Sport': 'NBA', 'League': 'NBA', 'Match': f"{a} @ {h}", 'Bet Type': 'Point Spread', 'Bet': f"{h} {game['spread']}", 'Odds': 1.91, 'Edge': edge/20, 'Confidence': 0.60})

    except Exception as e:
        print(f"!! NBA MODULE ERROR: {e}")
    return pd.DataFrame(nba_bets)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def run_backend_analysis():
    print("--- Starting Daily Global Backend Analysis ---")
    
    # Initialize empty file
    pd.DataFrame(columns=['Sport', 'League', 'Match', 'Bet Type', 'Bet', 'Odds', 'Edge', 'Confidence']).to_csv('latest_bets.csv', index=False)
    
    soccer_bets = run_soccer_module()
    nfl_bets = run_nfl_module()
    nba_bets = run_nba_module()
    
    all_bets = pd.concat([soccer_bets, nfl_bets, nba_bets], ignore_index=True)
    
    if not all_bets.empty:
        # Calculate Kelly Stake
        # Stake = (Edge / (Odds - 1)) * 0.25 (Quarter Kelly)
        # Note: Edge here is decimal (0.10 = 10%). 
        all_bets['Stake'] = (all_bets['Edge'] / (all_bets['Odds'] - 1)) * 0.25
        all_bets['Stake'] = all_bets['Stake'].clip(lower=0.0, upper=0.05) # Cap at 5% bankroll
        
        all_bets.to_csv('latest_bets.csv', index=False)
        print(f"\nSuccessfully saved {len(all_bets)} recommendations.")
    else:
        print("\nNo value bets found.")

if __name__ == "__main__":
    run_backend_analysis()
