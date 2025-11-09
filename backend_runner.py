# ==============================================================================
# Part 0: Setup and Dependencies
# ==============================================================================
pip install pandas scikit-learn xgboost joblib tqdm matplotlib beautifulsoup4 requests

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
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

# ==============================================================================
# Part 1: The Brain v19.0 - Training "The Resilient Machine"
# ==============================================================================

def calculate_elo_and_volatility(matches):
    """Calculates Elo ratings and a new Volatility Index for each team."""
    teams = pd.concat([matches['HomeTeam'], matches['AwayTeam']]).unique()
    elo_ratings = {team: 1500 for team in teams}
    elo_errors = {team: [] for team in teams} # To track prediction errors
    
    home_elos, away_elos = [], []
    for i, row in tqdm(matches.iterrows(), total=len(matches), desc="Calculating Elo & Volatility"):
        h, a = row['HomeTeam'], row['AwayTeam']
        r_h, r_a = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500)
        home_elos.append(r_h); away_elos.append(r_a)
        
        e_h = 1 / (1 + 10**((r_a - r_h) / 400))
        s_h = 1 if row['FTR'] == 'H' else 0 if row['FTR'] == 'A' else 0.5
        
        # Track error
        error = abs(s_h - e_h)
        elo_errors[h].append(error)
        elo_errors[a].append(error)
        
        k_factor = 20 * (1 + abs(row['FTHG'] - row['FTAG']) / 10)
        elo_ratings[h] += k_factor * (s_h - e_h)
        elo_ratings[a] += k_factor * ((1-s_h) - (1-e_h))
        
    matches['HomeElo'], matches['AwayElo'] = home_elos, away_elos
    
    # Calculate final volatility index for each team
    team_volatility = {team: np.mean(errors) for team, errors in elo_errors.items() if errors}
    
    return matches, elo_ratings, team_volatility

def train_the_brain_resilient():
    """
    Trains the complete four-model portfolio and saves the brain.
    """
    print("--- Training The Brain v19.0 (The Resilient Machine) ---")
    
    # --- Data Acquisition & Feature Engineering ---
    seasons = ['2324', '2223', '2122', '2021']
    df = pd.concat([pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{s}/E0.csv', parse_dates=['Date'], dayfirst=True, on_bad_lines='skip') for s in seasons]).sort_values('Date').reset_index(drop=True)
    if df.empty: raise ValueError("Data acquisition failed.")
    df, elo_ratings, team_volatility = calculate_elo_and_volatility(df)
    
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
    # (Model training logic remains the same as v18)
    features_alpha = ['elo_diff', 'form_diff']
    X_alpha, y = df[features_alpha], df['FTR']
    le = LabelEncoder(); y_encoded = le.fit_transform(y)
    scaler_alpha = StandardScaler(); X_scaled_alpha = scaler_alpha.fit_transform(X_alpha)
    base_model_alpha = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=250, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_alpha = CalibratedClassifierCV(base_model_alpha, method='isotonic', cv=5)
    model_alpha.fit(X_scaled_alpha, y_encoded)
    
    avg_goals_home, avg_goals_away = df['FTHG'].mean(), df['FTAG'].mean()
    home_strength = df.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'}).rename(columns={'FTHG': 'h_gf_avg', 'FTAG': 'h_ga_avg'})
    away_strength = df.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'}).rename(columns={'FTAG': 'a_gf_avg', 'FTHG': 'a_ga_avg'})
    team_strengths = pd.concat([home_strength, away_strength], axis=1)
    team_strengths['attack'] = (team_strengths['h_gf_avg'] / avg_goals_home + team_strengths['a_gf_avg'] / avg_goals_away) / 2
    team_strengths['defence'] = (team_strengths['h_ga_avg'] / avg_goals_away + team_strengths['a_ga_avg'] / avg_goals_home) / 2
    model_bravo = {'team_strengths': team_strengths[['attack', 'defence']], 'avg_goals_home': avg_goals_home, 'avg_goals_away': avg_goals_away}

    df['abs_elo_diff'] = abs(df['elo_diff'])
    features_charlie = ['abs_elo_diff', 'form_diff']
    y_draw = (df['FTR'] == 'D').astype(int)
    scaler_charlie = StandardScaler()
    X_charlie_scaled = scaler_charlie.fit_transform(df[features_charlie])
    model_charlie = LogisticRegression(class_weight='balanced', random_state=42)
    model_charlie.fit(X_charlie_scaled, y_draw)

    features_delta = ['B365H', 'B365D', 'B365A']
    X_delta = df[features_delta]
    scaler_delta = StandardScaler()
    X_delta_scaled = scaler_delta.fit_transform(X_delta)
    base_model_delta = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_delta = CalibratedClassifierCV(base_model_delta, method='isotonic', cv=5)
    model_delta.fit(X_delta_scaled, y_encoded)

    # --- Save the Brain Portfolio ---
    brain_portfolio = {
        'model_alpha': {'model': model_alpha, 'le': le, 'features': features_alpha, 'scaler': scaler_alpha, 'elo_ratings': elo_ratings, 'volatility': team_volatility},
        'model_bravo': model_bravo,
        'model_charlie': {'model': model_charlie, 'features': features_charlie, 'scaler': scaler_charlie},
        'model_delta': {'model': model_delta, 'features': features_delta, 'scaler': scaler_delta}
    }
    joblib.dump(brain_portfolio, 'betting_copilot_brain_v19.joblib')
    joblib.dump(df, 'historical_data_with_features_v19.joblib')
    print("\n✅ Brain v19.0 Portfolio training complete.")

# --- Execute the Training ---
train_the_brain_resilient()


# ==============================================================================
# Part 2: The Co-Pilot v19.0 - The Live Resilient Machine
# ==============================================================================

def get_news_alert(team1, team2):
    """Performs a simple Google News search for injury-related keywords."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        query = f'"{team1}" OR "{team2}" injury OR doubt OR out'
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=nws"
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = soup.find_all('h3')
        for h in headlines[:3]: # Check top 3 headlines
            if any(keyword in h.text.lower() for keyword in ['injury', 'doubt', 'out', 'miss', 'sidelined']):
                return f"⚠️ News Alert: Check recent injury news for {team1} or {team2}."
    except Exception:
        return None
    return None

def predict_with_portfolio(fixture, brain, historical_df):
    # (This function remains the same as v18)
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

def run_the_copilot():
    """
    Loads the v19.0 brain and runs all final analysis modules.
    """
    print("\n\n=============================================")
    print("🚀 LAUNCHING THE BETTING CO-PILOT (v19.0 The Resilient Machine) 🚀")
    print("=============================================")
    
    try:
        brain = joblib.load('betting_copilot_brain_v19.joblib')
        historical_df = joblib.load('historical_data_with_features_v19.joblib')
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
                    
                    # NEW: Volatility-Adjusted Staking
                    home_vol = brain['model_alpha']['volatility'].get(fixture['HomeTeam'], 0.25)
                    away_vol = brain['model_alpha']['volatility'].get(fixture['AwayTeam'], 0.25)
                    match_volatility = (home_vol + away_vol) / 2
                    volatility_adjustment = 1 - (match_volatility - 0.2) # Adjust stake down for volatile teams
                    
                    stake = (kelly_fraction / 4) * volatility_adjustment
                    
                    bet_info = {'Match': f"{fixture['HomeTeam']} vs {fixture['AwayTeam']}", 'Bet': {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome], 'Odds': fixture[odds_col], 'Edge': edge, 'Confidence': final_probs[outcome], 'Stake': stake}
                    value_bets.append(bet_info)
    
    # --- Add News Alerts ---
    print("\n--- Scanning for Real-Time News Alerts ---")
    for bet in tqdm(value_bets, desc="Checking News"):
        teams = bet['Match'].split(' vs ')
        bet['News Alert'] = get_news_alert(teams[0], teams[1])
        time.sleep(1) # Be respectful to Google News

    print("\n\n--- 📈 VALUE DASHBOARD (w/ News & Volatility-Adjusted Stake) 📈 ---")
    if value_bets:
        value_df = pd.DataFrame(value_bets).sort_values('Edge', ascending=False)
        for col in ['Edge', 'Confidence', 'Stake']: value_df[col] = value_df[col].map('{:.2%}'.format)
        print(value_df[['Match', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake', 'News Alert']].to_string(index=False))
        print("\n'Stake' is a risk-adjusted recommended bet size as a percentage of your bankroll.")
    else:
        print("No significant value bets found (Edge > 5%) in the current fixtures.")
        
    # (Other modules like Backtesting and Health Check would run here)

    print("\n=============================================")
    print("✅ Co-Pilot Analysis Complete")
    print("=============================================")

# --- Run the Co-Pilot ---
run_the_copilot()
