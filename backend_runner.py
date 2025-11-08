# ==============================================================================
# Part 0: Setup and Dependencies
# ==============================================================================
!pip install pandas scikit-learn xgboost joblib tqdm matplotlib

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import poisson
import joblib
import warnings
from itertools import combinations
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

warnings.filterwarnings('ignore')

# ==============================================================================
# Part 1: The Brain v13.0 - Training "The Final Machine" Portfolio
# ==============================================================================
# This version includes a dynamic K-factor and the 4-model portfolio.

def calculate_elo(matches):
    """Calculates Elo ratings with a dynamic K-factor based on goal difference."""
    teams = pd.concat([matches['HomeTeam'], matches['AwayTeam']]).unique()
    elo_ratings = {team: 1500 for team in teams}
    
    home_elos, away_elos = [], []
    for i, row in tqdm(matches.iterrows(), total=len(matches), desc="Calculating Elo Ratings"):
        h, a = row['HomeTeam'], row['AwayTeam']
        r_h, r_a = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500)
        home_elos.append(r_h); away_elos.append(r_a)
        
        goal_diff = abs(row['FTHG'] - row['FTAG'])
        k_factor = 20 * (1 + goal_diff / 10) # Dynamic K-Factor
        
        e_h = 1 / (1 + 10**((r_a - r_h) / 400))
        s_h = 1 if row['FTR'] == 'H' else 0 if row['FTR'] == 'A' else 0.5
        
        elo_ratings[h] += k_factor * (s_h - e_h)
        elo_ratings[a] += k_factor * ((1-s_h) - (1-e_h))
        
    matches['HomeElo'], matches['AwayElo'] = home_elos, away_elos
    return matches, elo_ratings

def train_the_brain_final():
    """
    Trains the complete four-model portfolio and saves the brain.
    """
    print("--- Training The Brain v13.0 (The Final Machine) ---")
    
    # --- Data Acquisition & Feature Engineering ---
    seasons = ['2324', '2223', '2122', '2021']
    df = pd.concat([pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{s}/E0.csv', parse_dates=['Date'], dayfirst=True, on_bad_lines='skip') for s in seasons]).sort_values('Date').reset_index(drop=True)
    if df.empty: raise ValueError("Data acquisition failed.")
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
    
    # --- Train Models ---
    print("\n--- Training Model Portfolio ---")
    # Model Alpha (Power-Form)
    features_alpha = ['elo_diff', 'form_diff']
    X_alpha, y = df[features_alpha], df['FTR']
    le = LabelEncoder(); y_encoded = le.fit_transform(y)
    scaler_alpha = StandardScaler(); X_scaled_alpha = scaler_alpha.fit_transform(X_alpha)
    base_model_alpha = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=250, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_alpha = CalibratedClassifierCV(base_model_alpha, method='isotonic', cv=5)
    model_alpha.fit(X_scaled_alpha, y_encoded)
    
    # Model Bravo (Goal-Expectancy)
    avg_goals_home, avg_goals_away = df['FTHG'].mean(), df['FTAG'].mean()
    home_strength = df.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'}).rename(columns={'FTHG': 'h_gf_avg', 'FTAG': 'h_ga_avg'})
    away_strength = df.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'}).rename(columns={'FTAG': 'a_gf_avg', 'FTHG': 'a_ga_avg'})
    team_strengths = pd.concat([home_strength, away_strength], axis=1)
    team_strengths['attack'] = (team_strengths['h_gf_avg'] / avg_goals_home + team_strengths['a_gf_avg'] / avg_goals_away) / 2
    team_strengths['defence'] = (team_strengths['h_ga_avg'] / avg_goals_away + team_strengths['a_ga_avg'] / avg_goals_home) / 2
    model_bravo = {'team_strengths': team_strengths[['attack', 'defence']], 'avg_goals_home': avg_goals_home, 'avg_goals_away': avg_goals_away}

    # Model Charlie (Draw Specialist)
    df['abs_elo_diff'] = abs(df['elo_diff'])
    features_charlie = ['abs_elo_diff', 'form_diff']
    y_draw = (df['FTR'] == 'D').astype(int)
    scaler_charlie = StandardScaler()
    X_charlie_scaled = scaler_charlie.fit_transform(df[features_charlie])
    model_charlie = LogisticRegression(class_weight='balanced', random_state=42)
    model_charlie.fit(X_charlie_scaled, y_draw)

    # Model Delta (Market Model)
    print("\n--- Training Model Delta (Market Model) ---")
    features_delta = ['B365H', 'B365D', 'B365A']
    X_delta = df[features_delta]
    scaler_delta = StandardScaler()
    X_delta_scaled = scaler_delta.fit_transform(X_delta)
    base_model_delta = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_delta = CalibratedClassifierCV(base_model_delta, method='isotonic', cv=5)
    model_delta.fit(X_delta_scaled, y_encoded)

    # --- Save the Brain Portfolio ---
    brain_portfolio = {
        'model_alpha': {'model': model_alpha, 'le': le, 'features': features_alpha, 'scaler': scaler_alpha, 'elo_ratings': elo_ratings},
        'model_bravo': model_bravo,
        'model_charlie': {'model': model_charlie, 'features': features_charlie, 'scaler': scaler_charlie},
        'model_delta': {'model': model_delta, 'features': features_delta, 'scaler': scaler_delta}
    }
    joblib.dump(brain_portfolio, 'betting_copilot_brain_v13.joblib')
    joblib.dump(df, 'historical_data_with_features_v13.joblib')
    print("\n✅ Brain v13.0 Portfolio training complete.")

# --- Execute the Training ---
train_the_brain_final()


# ==============================================================================
# Part 2: The Co-Pilot v13.0 - The Live Final Machine
# ==============================================================================

def predict_with_portfolio(fixture, brain, historical_df):
    """Generates predictions from all four models in the portfolio."""
    home_team, away_team = fixture['HomeTeam'], fixture['AwayTeam']
    probs_alpha, probs_bravo, prob_draw_charlie, probs_delta = {'H': 0.333, 'D': 0.333, 'A': 0.333}, {'H': 0.333, 'D': 0.333, 'A': 0.333}, 0.333, {'H': 0.333, 'D': 0.333, 'A': 0.333}
    try:
        elo_ratings = brain['model_alpha']['elo_ratings']
        home_elo, away_elo = elo_ratings.get(home_team, 1500), elo_ratings.get(away_team, 1500)
        home_form = historical_df[historical_df['HomeTeam'] == home_team].sort_values('Date').iloc[-1]
        away_form = historical_df[historical_df['AwayTeam'] == away_team].sort_values('Date').iloc[-1]
        features_raw_a = pd.DataFrame([{'elo_diff': home_elo - away_elo, 'form_diff': home_form['H_form_gd'] - away_form['A_form_gd']}])
        features_scaled_a = brain['model_alpha']['scaler'].transform(features_raw_a)
        probs_alpha_raw = brain['model_alpha']['model'].predict_proba(features_scaled_a)[0]
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
        prob_draw_charlie = brain['model_charlie']['model'].predict_proba(features_scaled_c)[0][1]
        features_raw_d = fixture[['B365H', 'B365D', 'B365A']].to_frame().T
        features_scaled_d = brain['model_delta']['scaler'].transform(features_raw_d)
        probs_delta_raw = brain['model_delta']['model'].predict_proba(features_scaled_d)[0]
        probs_delta = {le.classes_[i]: prob for i, prob in enumerate(probs_delta_raw)}
    except (IndexError, KeyError): pass
    return probs_alpha, probs_bravo, prob_draw_charlie, probs_delta

def get_risk_profile(bet, brain):
    """Categorizes a bet based on its characteristics."""
    elo_ratings = brain['model_alpha']['elo_ratings']
    match_teams = bet['Match'].split(' vs ')
    home_elo = elo_ratings.get(match_teams[0], 1500)
    away_elo = elo_ratings.get(match_teams[1], 1500)
    if bet['Bet'] == 'Home Win' and home_elo > 1600 and bet['Edge'] > 0.15: return "⭐ Fallen Angel"
    if bet['Bet'] == 'Away Win' and away_elo > 1600 and bet['Edge'] > 0.15: return "⭐ Fallen Angel"
    if bet['Bet'] in ['Home Win', 'Away Win'] and bet['Odds'] > 3.0 and bet['Edge'] > 0.1: return "⚡ Rising Star"
    if abs(home_elo - away_elo) < 50 and bet['Bet'] == 'Draw': return "⚖️ Derby Stalemate"
    return "Value Bet"

def run_the_copilot():
    """
    Loads the v13.0 brain and runs all final analysis modules.
    """
    print("\n\n=============================================")
    print("🚀 LAUNCHING THE BETTING CO-PILOT (v13.0 The Final Machine) 🚀")
    print("=============================================")
    
    try:
        brain = joblib.load('betting_copilot_brain_v13.joblib')
        historical_df = joblib.load('historical_data_with_features_v13.joblib')
    except FileNotFoundError:
        print("❌ Brain file not found. Please run the training function first."); return
        
    print("\n--- Downloading Real Upcoming Fixtures ---")
    try:
        fixtures_df = pd.read_csv('https://www.football-data.co.uk/fixtures.csv', encoding='latin1')
        fixtures_df = fixtures_df[fixtures_df['Div'] == 'E0']
        if fixtures_df.empty: print("No upcoming Premier League fixtures found."); return
        print(f"Found {len(fixtures_df)} upcoming Premier League matches.")
    except Exception as e:
        print(f"❌ Could not fetch fixtures: {e}"); return

    # --- Analyze Each Fixture with the Portfolio ---
    value_bets = []
    for index, fixture in tqdm(fixtures_df.iterrows(), total=len(fixtures_df), desc="Analyzing Live Fixtures"):
        probs_alpha, probs_bravo, prob_draw_charlie, probs_delta = predict_with_portfolio(fixture, brain, historical_df)
        final_probs = {
            'H': (probs_alpha['H'] * 0.5 + probs_bravo['H'] * 0.3) * 0.8 + probs_delta['H'] * 0.2,
            'A': (probs_alpha['A'] * 0.5 + probs_bravo['A'] * 0.3) * 0.8 + probs_delta['A'] * 0.2,
            'D': (probs_alpha['D'] * 0.4 + probs_bravo['D'] * 0.2 + prob_draw_charlie * 0.2) * 0.8 + probs_delta['D'] * 0.2
        }
        total_prob = sum(final_probs.values())
        final_probs = {k: v / total_prob for k, v in final_probs.items()}
        for outcome, odds_col in [('H', 'B365H'), ('D', 'B365D'), ('A', 'B365A')]:
            if pd.notna(fixture[odds_col]) and fixture[odds_col] > 0:
                edge = (final_probs[outcome] * fixture[odds_col]) - 1
                if edge > 0.05:
                    kelly_fraction = edge / (fixture[odds_col] - 1) if fixture[odds_col] > 1 else 0
                    bet_info = {'Match': f"{fixture['HomeTeam']} vs {fixture['AwayTeam']}", 'Bet': {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome], 'Odds': fixture[odds_col], 'Edge': edge, 'Confidence': final_probs[outcome], 'Stake (Kelly/4)': kelly_fraction / 4}
                    bet_info['Profile'] = get_risk_profile(bet_info, brain)
                    value_bets.append(bet_info)
    
    # --- Executive Summary ---
    print("\n\n--- 📝 EXECUTIVE SUMMARY 📝 ---")
    if value_bets:
        sorted_bets = sorted(value_bets, key=lambda x: x['Edge'], reverse=True)
        bet_of_the_week = sorted([b for b in sorted_bets if b['Confidence'] > 0.5], key=lambda x: x['Edge'], reverse=True)
        top_underdog = sorted([b for b in sorted_bets if b['Odds'] > 3.0], key=lambda x: x['Edge'], reverse=True)
        if bet_of_the_week: print(f"🎯 Bet of the Week: {bet_of_the_week[0]['Bet']} in '{bet_of_the_week[0]['Match']}' @ {bet_of_the_week[0]['Odds']}")
        else: print("🎯 Bet of the Week: No high-confidence favorite found.")
        if top_underdog: print(f"⚡ Top Underdog Play: {top_underdog[0]['Bet']} in '{top_underdog[0]['Match']}' @ {top_underdog[0]['Odds']}")
        else: print("⚡ Top Underdog Play: No significant underdog value found.")
    else:
        print("No recommendations meet the criteria this week.")

    print("\n\n--- 📈 VALUE DASHBOARD (w/ Risk Profile) 📈 ---")
    if value_bets:
        value_df = pd.DataFrame(value_bets).sort_values('Edge', ascending=False)
        for col in ['Edge', 'Confidence', 'Stake (Kelly/4)']: value_df[col] = value_df[col].map('{:.2%}'.format)
        print(value_df[['Match', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake (Kelly/4)', 'Profile']].to_string(index=False))
    else:
        print("No significant value bets found (Edge > 5%) in the current fixtures.")
        
    # --- Full Backtesting Module ---
    print("\n\n--- 📊 BACKTESTING & EQUITY CURVE 📊 ---")
    try:
        backtest_df = historical_df.copy()
        split_point = int(len(backtest_df) * 0.75)
        test_set = backtest_df.iloc[split_point:]
        X_test_alpha = test_set[brain['model_alpha']['features']]
        X_test_scaled = brain['model_alpha']['scaler'].transform(X_test_alpha)
        test_probs = brain['model_alpha']['model'].predict_proba(X_test_scaled)
        test_set['H_prob'], test_set['D_prob'], test_set['A_prob'] = test_probs[:,0], test_probs[:,1], test_probs[:,2]
        test_set['edge_H'] = test_set['H_prob'] * test_set['B365H'] - 1
        test_set['edge_D'] = test_set['D_prob'] * test_set['B365D'] - 1
        test_set['edge_A'] = test_set['A_prob'] * test_set['B365A'] - 1
        def calculate_profit(row):
            profit, bets = 0, 0
            if row['edge_H'] > 0.05: profit += np.where(row['FTR'] == 'H', row['B365H'] - 1, -1); bets += 1
            if row['edge_D'] > 0.05: profit += np.where(row['FTR'] == 'D', row['B365D'] - 1, -1); bets += 1
            if row['edge_A'] > 0.05: profit += np.where(row['FTR'] == 'A', row['B365A'] - 1, -1); bets += 1
            return pd.Series([profit, bets])
        test_set[['profit', 'bets']] = test_set.apply(calculate_profit, axis=1)
        test_set['cumulative_profit'] = test_set['profit'].cumsum()
        peak = test_set['cumulative_profit'].expanding(min_periods=1).max()
        drawdown = (test_set['cumulative_profit'] - peak)
        max_drawdown = drawdown.min()
        total_profit = test_set['cumulative_profit'].iloc[-1]
        total_bets = test_set['bets'].sum()
        roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0
        print(f"Backtest Results ({len(test_set)} matches, {int(total_bets)} bets):")
        print(f"  - Total Profit: {total_profit:.2f} units")
        print(f"  - ROI: {roi:.2f}%")
        print(f"  - Max Drawdown: {max_drawdown:.2f} units")
        plt.figure(figsize=(12, 6))
        plt.plot(test_set['Date'], test_set['cumulative_profit'], label='Equity Curve')
        plt.title('Backtest Equity Curve (Profit Over Time)')
        plt.xlabel('Date'); plt.ylabel('Cumulative Profit (Units)'); plt.grid(True); plt.legend(); plt.show()
    except Exception as e:
        print(f"Could not run backtest: {e}")

    # --- System Health Check ---
    print("\n\n--- 🩺 SYSTEM HEALTH CHECK 🩺 ---")
    try:
        last_10_games = historical_df.tail(10)
        X_health = last_10_games[brain['model_alpha']['features']]
        X_health_scaled = brain['model_alpha']['scaler'].transform(X_health)
        y_health_true = brain['model_alpha']['le'].transform(last_10_games['FTR'])
        health_probs = brain['model_alpha']['model'].predict_proba(X_health_scaled)
        logloss = log_loss(y_health_true, health_probs)
        print(f"Model Alpha Log Loss on last 10 completed games: {logloss:.4f}")
        if logloss > 1.0: print("⚠️ WARNING: Model performance may be degrading. Consider retraining the brain.")
        else: print("✅ Model performance is stable.")
    except Exception as e:
        print(f"Could not run health check: {e}")

    print("\n=============================================")
    print("✅ Co-Pilot Analysis Complete")
    print("=============================================")

# --- Run the Co-Pilot ---
run_the_copilot()
