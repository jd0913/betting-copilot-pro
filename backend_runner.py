# ==============================================================================
# backend_runner.py - The Automated Backend Engine (Corrected)
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import poisson
import joblib
import warnings
from itertools import combinations
import os
# This import was missing from the backend script but is needed for Model Charlie
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# ==============================================================================
# Part 1: The Brain - Training the "Quant-in-a-Box" Portfolio
# ==============================================================================

def calculate_elo(matches):
    """Calculates Elo ratings for all teams over the historical data."""
    teams = pd.concat([matches['HomeTeam'], matches['AwayTeam']]).unique()
    elo_ratings = {team: 1500 for team in teams}
    k_factor = 20
    home_elos, away_elos = [], []
    for i, row in matches.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        r_h, r_a = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500)
        home_elos.append(r_h); away_elos.append(r_a)
        e_h = 1 / (1 + 10**((r_a - r_h) / 400))
        s_h = 1 if row['FTR'] == 'H' else 0 if row['FTR'] == 'A' else 0.5
        elo_ratings[h] += k_factor * (s_h - e_h)
        elo_ratings[a] += k_factor * ((1-s_h) - (1-e_h))
    matches['HomeElo'], matches['AwayElo'] = home_elos, away_elos
    return matches, elo_ratings

def train_the_brain_quant():
    """
    Trains the complete three-model portfolio.
    """
    print("--- Training The Brain (Quant-in-a-Box) ---")
    
    # --- Data Acquisition ---
    seasons = ['2324', '2223', '2122', '2021']
    df = pd.concat([pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{s}/E0.csv', parse_dates=['Date'], dayfirst=True, on_bad_lines='skip') for s in seasons]).sort_values('Date').reset_index(drop=True)
    if df.empty: raise ValueError("Data acquisition failed.")
    
    # --- Feature Engineering ---
    df, elo_ratings = calculate_elo(df)
    home_games = df[['Date', 'HomeTeam', 'FTHG', 'FTAG']].rename(columns={'HomeTeam': 'team', 'FTHG': 'gf', 'FTAG': 'ga'})
    away_games = df[['Date', 'AwayTeam', 'FTAG', 'FTHG']].rename(columns={'AwayTeam': 'team', 'FTAG': 'gf', 'FTHG': 'ga'})
    team_games = pd.concat([home_games, away_games]).sort_values(['team', 'Date'])
    team_games['gd'] = team_games['gf'] - team_games['ga']
    team_games['form_gd'] = team_games.groupby('team')['gd'].shift(1).rolling(5, min_periods=1).mean()
    df = pd.merge(df, team_games[['Date', 'team', 'form_gd']], left_on=['Date', 'HomeTeam'], right_on=['Date', 'team'], how='left').rename(columns={'form_gd': 'H_form_gd'})
    df = pd.merge(df, team_games[['Date', 'team', 'form_gd']], left_on=['Date', 'AwayTeam'], right_on=['Date', 'team'], how='left').rename(columns={'form_gd': 'A_form_gd'})
    df.dropna(subset=['B365H', 'B365D', 'B365A', 'H_form_gd', 'A_form_gd'], inplace=True)
    df['elo_diff'] = df['HomeElo'] - df['AwayElo']
    df['form_diff'] = df['H_form_gd'] - df['A_form_gd']
    
    # --- Model Alpha Training ---
    features_alpha = ['elo_diff', 'form_diff']
    X_alpha, y = df[features_alpha], df['FTR']
    le = LabelEncoder(); y_encoded = le.fit_transform(y)
    scaler_alpha = StandardScaler(); X_scaled_alpha = scaler_alpha.fit_transform(X_alpha)
    base_model_alpha = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=250, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_alpha = CalibratedClassifierCV(base_model_alpha, method='isotonic', cv=5)
    model_alpha.fit(X_scaled_alpha, y_encoded)
    
    # --- Model Bravo Training ---
    avg_goals_home = df['FTHG'].mean()
    avg_goals_away = df['FTAG'].mean()
    home_strength = df.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'}).rename(columns={'FTHG': 'h_gf_avg', 'FTAG': 'h_ga_avg'})
    away_strength = df.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'}).rename(columns={'FTAG': 'a_gf_avg', 'FTHG': 'a_ga_avg'})
    team_strengths = pd.concat([home_strength, away_strength], axis=1)
    team_strengths['attack'] = (team_strengths['h_gf_avg'] / avg_goals_home + team_strengths['a_gf_avg'] / avg_goals_away) / 2
    team_strengths['defence'] = (team_strengths['h_ga_avg'] / avg_goals_away + team_strengths['a_ga_avg'] / avg_goals_home) / 2
    model_bravo = {'team_strengths': team_strengths[['attack', 'defence']], 'avg_goals_home': avg_goals_home, 'avg_goals_away': avg_goals_away}

    # --- Model Charlie Training ---
    df['abs_elo_diff'] = abs(df['elo_diff'])
    features_charlie = ['abs_elo_diff', 'form_diff']
    y_draw = (df['FTR'] == 'D').astype(int)
    scaler_charlie = StandardScaler()
    X_charlie_scaled = scaler_charlie.fit_transform(df[features_charlie])
    model_charlie = LogisticRegression(class_weight='balanced', random_state=42)
    model_charlie.fit(X_charlie_scaled, y_draw)

    # --- Return the complete brain and the processed historical data ---
    return {
        'model_alpha': {'model': model_alpha, 'le': le, 'features': features_alpha, 'scaler': scaler_alpha, 'elo_ratings': elo_ratings},
        'model_bravo': model_bravo,
        'model_charlie': {'model': model_charlie, 'features': features_charlie, 'scaler': scaler_charlie}
    }, df

# ==============================================================================
# Part 2: The Co-Pilot - Prediction and Analysis Logic
# ==============================================================================

def predict_with_portfolio(fixture, brain, historical_df):
    """Generates predictions from all three models in the portfolio."""
    home_team, away_team = fixture['HomeTeam'], fixture['AwayTeam']
    probs_alpha, probs_bravo, prob_draw_charlie = {'H': 0.333, 'D': 0.333, 'A': 0.333}, {'H': 0.333, 'D': 0.333, 'A': 0.333}, 0.333
    try:
        elo_ratings = brain['model_alpha']['elo_ratings']
        home_elo, away_elo = elo_ratings.get(home_team, 1500), elo_ratings.get(away_team, 1500)
        home_form = historical_df[historical_df['HomeTeam'] == home_team].sort_values('Date').iloc[-1]
        away_form = historical_df[historical_df['AwayTeam'] == away_team].sort_values('Date').iloc[-1]
        features_raw_a = pd.DataFrame([{'elo_diff': home_elo - away_elo, 'form_diff': home_form['H_form_gd'] - away_form['A_form_gd']}])
        features_scaled_a = brain['model_alpha']['scaler'].transform(features_raw_a)
        probs_alpha_raw = brain['model_alpha']['model'].predict_proba(features_scaled_a)
        le = brain['model_alpha']['le']
        probs_alpha = {le.classes_[i]: prob for i, prob in enumerate(probs_alpha_raw)}
        strengths = brain['model_bravo']['team_strengths']
        h_attack, a_defence = strengths.loc[home_team]['attack'], strengths.loc[away_team]['defence']
        a_attack, h_defence = strengths.loc[away_team]['attack'], strengths.loc[home_team]['defence']
        expected_h_goals = h_attack * a_defence * brain['model_bravo']['avg_goals_home']
        expected_a_goals = a_attack * h_defence * brain['model_bravo']['avg_goals_away']
        prob_matrix = np.array([[poisson.pmf(i, expected_h_goals) * poisson.pmf(j, expected_a_goals) for j in range(6)] for i in range(6)])
        prob_h, prob_d, prob_a = np.sum(np.tril(prob_matrix, -1)), np.sum(np.diag(prob_matrix)), np.sum(np.triu(prob_matrix, 1))
        total_prob = prob_h + prob_d + prob_a
        probs_bravo = {'H': prob_h / total_prob, 'D': prob_d / total_prob, 'A': prob_a / total_prob}
        features_raw_c = pd.DataFrame([{'abs_elo_diff': abs(home_elo - away_elo), 'form_diff': home_form['H_form_gd'] - away_form['A_form_gd']}])
        features_scaled_c = brain['model_charlie']['scaler'].transform(features_raw_c)
        prob_draw_charlie = brain['model_charlie']['model'].predict_proba(features_scaled_c)
    except (IndexError, KeyError): pass
    return probs_alpha, probs_bravo, prob_draw_charlie

def run_backend_analysis():
    """
    The main backend function. Trains, predicts, and saves the results.
    """
    print("--- Starting Daily Backend Analysis ---")
    
    # 1. Train the Brain
    brain, historical_df = train_the_brain_quant()
    
    # 2. Get Live Fixtures
    print("--- Downloading Live Fixtures ---")
    try:
        fixtures_df = pd.read_csv('https://www.football-data.co.uk/fixtures.csv', encoding='latin1')
        epl_fixtures = fixtures_df[fixtures_df['Div'] == 'E0']
        if epl_fixtures.empty:
            print("No upcoming EPL fixtures found. Saving empty file.")
            pd.DataFrame([]).to_csv('latest_bets.csv', index=False)
            return
    except Exception as e:
        print(f"Could not fetch fixtures: {e}")
        return

    # 3. Analyze Fixtures
    print(f"--- Analyzing {len(epl_fixtures)} Fixtures ---")
    value_bets = []
    for index, fixture in epl_fixtures.iterrows():
        probs_alpha, probs_bravo, prob_draw_charlie = predict_with_portfolio(fixture, brain, historical_df)
        final_probs = {
            'H': probs_alpha['H'] * 0.6 + probs_bravo['H'] * 0.4,
            'A': probs_alpha['A'] * 0.6 + probs_bravo['A'] * 0.4,
            'D': probs_alpha['D'] * 0.5 + probs_bravo['D'] * 0.3 + prob_draw_charlie * 0.2
        }
        total_prob = sum(final_probs.values())
        final_probs = {k: v / total_prob for k, v in final_probs.items()}
        for outcome, odds_col in [('H', 'B365H'), ('D', 'B365D'), ('A', 'B365A')]:
            if pd.notna(fixture[odds_col]) and fixture[odds_col] > 0:
                edge = (final_probs[outcome] * fixture[odds_col]) - 1
                if edge > 0.05:
                    kelly_fraction = edge / (fixture[odds_col] - 1) if fixture[odds_col] > 1 else 0
                    value_bets.append({'Match': f"{fixture['HomeTeam']} vs {fixture['AwayTeam']}", 'Bet': {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome], 'Odds': fixture[odds_col], 'Edge': edge, 'Confidence': final_probs[outcome], 'Stake (Kelly/4)': kelly_fraction / 4})
    
    # 4. Save the Results
    if value_bets:
        results_df = pd.DataFrame(value_bets)
        results_df.to_csv('latest_bets.csv', index=False)
        print(f"Successfully saved {len(results_df)} recommendations to latest_bets.csv")
    else:
        print("No value bets found. Saving an empty file.")
        pd.DataFrame([]).to_csv('latest_bets.csv', index=False)

if __name__ == "__main__":
    run_backend_analysis()
