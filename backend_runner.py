# ==============================================================================
# backend_runner.py - The "Everything Machine" Master Controller
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
import nfl_data_py as nfl
from nba_api.stats.endpoints import leaguedashteamstats
from scipy.stats import poisson

warnings.filterwarnings('ignore')

# ==============================================================================
# SOCCER MODULE
# ==============================================================================
def run_soccer_module():
    print("--- Running Soccer Module ---")
    SOCCER_LEAGUE_CONFIG = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
    all_soccer_bets = []
    try:
        fixtures_df = pd.read_csv('https://www.football-data.co.uk/fixtures.csv', encoding='latin1')
        for league_div, league_name in SOCCER_LEAGUE_CONFIG.items():
            seasons = ['2324', '2223', '2122']
            df = pd.concat([pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{s}/{league_div}.csv', parse_dates=['Date'], dayfirst=True, on_bad_lines='skip', encoding='latin1') for s in seasons]).sort_values('Date').reset_index(drop=True)
            if df.empty: continue
            
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
                h_attack, a_defence = team_strengths.loc[h]['attack'], team_strengths.loc[a]['defence']
                a_attack, h_defence = team_strengths.loc[a]['attack'], team_strengths.loc[h]['defence']
                exp_h_g = h_attack * a_defence * avg_goals_home
                exp_a_g = a_attack * h_defence * avg_goals_away
                
                prob_matrix = np.array([[poisson.pmf(i, exp_h_g) * poisson.pmf(j, exp_a_g) for j in range(6)] for i in range(6)])
                prob_h, prob_d, prob_a = np.sum(np.tril(prob_matrix, -1)), np.sum(np.diag(prob_matrix)), np.sum(np.triu(prob_matrix, 1))
                prob_over = 1 - (prob_matrix[0,0] + prob_matrix[0,1] + prob_matrix[1,0] + prob_matrix[1,1] + prob_matrix[0,2] + prob_matrix[2,0])
                
                # Moneyline Bets
                for outcome, prob, odds_col in [('Home Win', prob_h, 'B365H'), ('Draw', prob_d, 'B365D'), ('Away Win', prob_a, 'B365A')]:
                    if pd.notna(fixture[odds_col]) and fixture[odds_col] > 0:
                        edge = (prob * fixture[odds_col]) - 1
                        if edge > 0.10: all_soccer_bets.append({'Sport': 'Soccer', 'League': league_name, 'Match': f"{h} vs {a}", 'Bet Type': 'Moneyline', 'Bet': outcome, 'Odds': fixture[odds_col], 'Edge': edge, 'Confidence': prob})
                
                # Over/Under Bets
                if pd.notna(fixture['B365>2.5']) and fixture['B365>2.5'] > 0:
                    edge_over = (prob_over * fixture['B365>2.5']) - 1
                    if edge_over > 0.10: all_soccer_bets.append({'Sport': 'Soccer', 'League': league_name, 'Match': f"{h} vs {a}", 'Bet Type': 'Total Goals', 'Bet': 'Over 2.5', 'Odds': fixture['B365>2.5'], 'Edge': edge_over, 'Confidence': prob_over})
    except Exception as e:
        print(f"!! SOCCER MODULE FAILED: {e}")
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
        df['total_line'] = df['home_score'] + df['away_score']
        
        home_power = df.groupby('home_team')['spread_line'].mean()
        away_power = df.groupby('away_team')['spread_line'].mean() * -1
        team_power = (home_power + away_power) / 2
        
        schedule = nfl.import_schedules(years=[2024])
        weekly_odds = nfl.import_weekly_data(years=[2024])
        next_week = weekly_odds[weekly_odds['spread_line'].notna()]['week'].min()
        if pd.isna(next_week): return pd.DataFrame()
            
        upcoming_games = schedule[schedule['week'] == next_week]
        upcoming_odds = weekly_odds[weekly_odds['week'] == next_week]
        
        for i, game in upcoming_games.iterrows():
            h, a = game['home_team'], game['away_team']
            game_odds = upcoming_odds[(upcoming_odds['home_team'] == h) & (upcoming_odds['away_team'] == a)].iloc[0]
            
            pred_margin = team_power.get(h, 0) - team_power.get(a, 0)
            spread_line = game_odds['spread_line']
            
            if pred_margin > (spread_line * -1):
                edge = pred_margin - (spread_line * -1)
                if edge > 1.0: # Edge of more than 1 point
                    nfl_bets.append({'Sport': 'NFL', 'League': 'NFL', 'Match': f"{a} @ {h}", 'Bet Type': 'Point Spread', 'Bet': f"{h} {spread_line}", 'Odds': -110, 'Edge': edge, 'Confidence': pred_margin})
    except Exception as e:
        print(f"!! NFL MODULE FAILED: {e}")
    return pd.DataFrame(nfl_bets)

# ==============================================================================
# NBA MODULE
# ==============================================================================
def run_nba_module():
    print("--- Running NBA Module ---")
    nba_bets = []
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(season="2023-24").get_data_frames()[0]
        stats['PLUS_MINUS_PER_100'] = stats['PLUS_MINUS'] / (stats['GP'] * (stats['MIN'] / 48)) * 100
        team_power = stats.set_index('TEAM_ABBREVIATION')['PLUS_MINUS_PER_100']
        
        # (In a real system, you'd scrape live weekly schedules and odds here)
        # For this script, we'll create a placeholder fixture to demonstrate the logic
        placeholder_fixture = {'home_team': 'BOS', 'away_team': 'LAL', 'spread': -5.5}
        
        h, a = placeholder_fixture['home_team'], placeholder_fixture['away_team']
        pred_margin = team_power.get(h, 0) - team_power.get(a, 0)
        spread_line = placeholder_fixture['spread']
        
        if pred_margin > (spread_line * -1):
            edge = pred_margin - (spread_line * -1)
            if edge > 1.0:
                nba_bets.append({'Sport': 'NBA', 'League': 'NBA', 'Match': f"{a} @ {h}", 'Bet Type': 'Point Spread', 'Bet': f"{h} {spread_line}", 'Odds': -110, 'Edge': edge, 'Confidence': pred_margin})
    except Exception as e:
        print(f"!! NBA MODULE FAILED: {e}")
    return pd.DataFrame(nba_bets)

# ==============================================================================
# Main Execution
# ==============================================================================
def run_backend_analysis():
    """The master backend function. Runs all sport modules."""
    print("--- Starting Daily Global Backend Analysis ---")
    
    soccer_bets_df = run_soccer_module()
    nfl_bets_df = run_nfl_module()
    nba_bets_df = run_nba_module()
    
    all_bets_df = pd.concat([soccer_bets_df, nfl_bets_df, nba_bets_df], ignore_index=True)
    
    if not all_bets_df.empty:
        all_bets_df.to_csv('latest_bets.csv', index=False)
        print(f"\nSuccessfully saved {len(all_bets_df)} total recommendations to latest_bets.csv")
    else:
        print("\nNo value bets found across all sports. Saving an empty file.")
        pd.DataFrame([]).to_csv('latest_bets.csv', index=False)

if __name__ == "__main__":
    run_backend_analysis()
