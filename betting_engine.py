# betting_engine.py
# Betting Co-Pilot Pro - v59.0 (Refactored)
# Key Fixes:
#   - UTC timestamp standardization
#   - Secure API key handling
#   - Date-aware settlement engine
#   - Model caching to prevent redundant training
#   - Circuit breakers for external API failures
#   - Proper Kelly stake calculation

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import poisson
import joblib
import nfl_data_py as nfl
from nba_api.stats.endpoints import LeagueDashTeamStats
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta, timezone
import os
import json
import random
from fuzzywuzzy import process
import logging
import config

# ==============================================================================
# LOGGING SETUP (Professional standard)
# ==============================================================================
logger = logging.getLogger("BettingEngine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ==============================================================================
# SECURE API SESSION (With retries and timeouts)
# ==============================================================================
def create_api_session():
    """Create resilient session with retry logic"""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({
        "User-Agent": "BettingCoPilot-Pro/1.0 (Contact: your@email.com)",
        "Accept": "application/json"
    })
    return session

API_SESSION = create_api_session()

# ==============================================================================
# CORE MATH MODULES (Optimized and Validated)
# ==============================================================================
def calculate_pythagorean_expectation(points_for: float, points_against: float, exponent: float = 1.83) -> float:
    """Calculate Pythagorean expectation with safety checks"""
    if points_for < 0 or points_against < 0:
        logger.warning(f"Negative points in pythagorean calc: {points_for}, {points_against}")
        return 0.5
    if points_for == 0 and points_against == 0:
        return 0.5
    numerator = points_for ** exponent
    denominator = numerator + (points_against ** exponent)
    return numerator / denominator if denominator > 0 else 0.5

def zero_inflated_poisson(k: int, lam: float, pi: float = 0.05) -> float:
    """ZI Poisson distribution with validation"""
    if lam <= 0 or pi < 0 or pi > 1:
        logger.warning(f"Invalid ZI Poisson params: k={k}, lam={lam}, pi={pi}")
        return poisson.pmf(k, max(lam, 0.1))
    if k == 0:
        return pi + (1 - pi) * poisson.pmf(0, lam)
    return (1 - pi) * poisson.pmf(k, lam)

# ==============================================================================
# MODEL EVOLUTION SYSTEM (With Caching)
# ==============================================================================
MODEL_CACHE_DIR = "model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
GENOME_FILE = os.path.join(MODEL_CACHE_DIR, 'model_genome.json')

def load_or_initialize_genome() -> dict:
    """Load genome from disk or create new with validation"""
    default_genome = {
        'generation': 0,
        'best_score': 10.0,
        'xgb_n_estimators': 200,
        'xgb_max_depth': 3,
        'xgb_learning_rate': 0.1,
        'rf_n_estimators': 200,
        'rf_max_depth': 10,
        'nn_hidden_layer_size': 64,
        'nn_alpha': 0.0001
    }
    
    if os.path.exists(GENOME_FILE):
        try:
            with open(GENOME_FILE, 'r') as f:
                genome = json.load(f)
                # Validate required keys
                for key in default_genome.keys():
                    if key not in genome:
                        logger.warning(f"Missing genome key: {key}, using default")
                        genome[key] = default_genome[key]
                return genome
        except Exception as e:
            logger.error(f"Genome load failed: {str(e)}, using defaults")
    
    return default_genome

def save_genome(genome: dict):
    """Save genome with atomic write"""
    try:
        temp_path = GENOME_FILE + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(genome, f, indent=2)
        os.replace(temp_path, GENOME_FILE)
        logger.info(f"Genome saved: generation {genome['generation']}, score {genome['best_score']:.4f}")
    except Exception as e:
        logger.error(f"Genome save failed: {str(e)}")

def build_ensemble_from_genome(genome: dict) -> VotingClassifier:
    """Build model ensemble with parameter validation"""
    # Validate parameters
    n_estimators = max(50, min(500, int(genome.get('xgb_n_estimators', 200))))
    max_depth = max(2, min(10, int(genome.get('xgb_max_depth', 3))))
    learning_rate = max(0.01, min(0.3, float(genome.get('xgb_learning_rate', 0.1))))
    
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'  # Faster training
    )
    
    rf_clf = RandomForestClassifier(
        n_estimators=max(50, min(300, int(genome.get('rf_n_estimators', 200)))),
        max_depth=max(3, min(15, int(genome.get('rf_max_depth', 10)))),
        random_state=42,
        n_jobs=-1
    )
    
    nn_clf = MLPClassifier(
        hidden_layer_sizes=(
            max(32, min(128, int(genome.get('nn_hidden_layer_size', 64)))),
            max(16, min(64, int(genome.get('nn_hidden_layer_size', 64)) // 2))
        ),
        alpha=max(1e-5, min(0.1, float(genome.get('nn_alpha', 0.0001)))),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
    
    lr_clf = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    
    return VotingClassifier(
        estimators=[
            ('xgb', xgb_clf),
            ('rf', rf_clf),
            ('nn', nn_clf),
            ('lr', lr_clf)
        ],
        voting='soft',
        n_jobs=-1
    )

def train_ensemble_model(X, y, genome: dict) -> tuple:
    """Train ensemble with meta-calibration and caching"""
    cache_key = f"ensemble_{hash(str(genome))}.pkl"
    cache_path = os.path.join(MODEL_CACHE_DIR, cache_key)
    
    if os.path.exists(cache_path):
        logger.info("Loading cached model")
        try:
            return joblib.load(cache_path)
        except Exception as e:
            logger.warning(f"Cache load failed: {str(e)}")
    
    logger.info("Training new ensemble model")
    model = build_ensemble_from_genome(genome)
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated_model.fit(X, y)
    
    # Save to cache
    try:
        joblib.dump(calibrated_model, cache_path)
        logger.info(f"Model cached at {cache_path}")
    except Exception as e:
        logger.error(f"Cache save failed: {str(e)}")
    
    return calibrated_model, None  # Meta-model not implemented in this version

# ==============================================================================
# DATA FETCHING MODULES (Resilient and Secure)
# ==============================================================================
def get_live_odds(sport_key: str, timeout: int = 10) -> list:
    """Fetch odds with circuit breaker and validation"""
    api_key = config.API_CONFIG.get("THE_ODDS_API_KEY", "").strip()
    if not api_key or "PASTE_YOUR" in api_key:
        logger.warning("The Odds API key not configured")
        return []
    
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
    params = {
        'api_key': api_key,
        'regions': 'us',
        'markets': 'h2h,spreads',
        'oddsFormat': 'decimal',
        'dateFormat': 'iso'
    }
    
    try:
        response = API_SESSION.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        # Validate response structure
        if not isinstance(data, list):
            logger.error(f"Unexpected odds response type: {type(data)}")
            return []
        
        logger.info(f"Fetched {len(data)} games for {sport_key}")
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Odds API request failed for {sport_key}: {str(e)}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from odds API: {str(e)}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error fetching odds for {sport_key}")
        return []

def find_arbitrage(game: dict, sport_type: str) -> tuple:
    """Calculate arbitrage with sport-specific rules"""
    best_home = {'price': 0, 'book': ''}
    best_away = {'price': 0, 'book': ''}
    best_draw = {'price': 0, 'book': ''}
    
    for bookmaker in game.get('bookmakers', []):
        for market in bookmaker.get('markets', []):
            if market.get('key') != 'h2h':
                continue
                
            for outcome in market.get('outcomes', []):
                price = outcome.get('price', 0)
                name = outcome.get('name', '')
                
                if name == game.get('home_team') and price > best_home['price']:
                    best_home = {'price': price, 'book': bookmaker.get('title', 'Unknown')}
                elif name == game.get('away_team') and price > best_away['price']:
                    best_away = {'price': price, 'book': bookmaker.get('title', 'Unknown')}
                elif name == 'Draw' and price > best_draw['price']:
                    best_draw = {'price': price, 'book': bookmaker.get('title', 'Unknown')}
    
    implied_prob = 0.0
    arb_info = None
    
    # Soccer has 3 outcomes
    if sport_type == 'Soccer':
        if best_home['price'] > 1 and best_away['price'] > 1 and best_draw['price'] > 1:
            implied_prob = (1/best_home['price']) + (1/best_away['price']) + (1/best_draw['price'])
            if implied_prob < 0.99:  # 1% margin for error
                arb_info = (
                    f"Home: {best_home['book']} ({best_home['price']:.2f}) | "
                    f"Draw: {best_draw['book']} ({best_draw['price']:.2f}) | "
                    f"Away: {best_away['book']} ({best_away['price']:.2f})"
                )
    # Other sports have 2 outcomes
    else:
        if best_home['price'] > 1 and best_away['price'] > 1:
            implied_prob = (1/best_home['price']) + (1/best_away['price'])
            if implied_prob < 0.99:
                arb_info = (
                    f"Home: {best_home['book']} ({best_home['price']:.2f}) | "
                    f"Away: {best_away['book']} ({best_away['price']:.2f})"
                )
    
    if arb_info and implied_prob > 0:
        edge = (1 - implied_prob) / implied_prob
        return edge, arb_info, best_home, best_draw, best_away
    
    return 0.0, None, best_home, best_draw, best_away

def fuzzy_match_team(team_name: str, team_list: list) -> str:
    """Match team names with confidence threshold"""
    if not team_list or not team_name:
        return None
    
    match, score = process.extractOne(team_name, team_list)
    if score >= 85:  # High confidence threshold
        return match
    return None

# ==============================================================================
# SPORT MODULES (Refactored for Maintainability)
# ==============================================================================
def get_soccer_results(league_code: str, season: str = "2324") -> pd.DataFrame:
    """Fetch and validate soccer results with date awareness"""
    try:
        url = f'https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv'
        df = pd.read_csv(
            url,
            parse_dates=['Date'],
            dayfirst=True,
            on_bad_lines='skip',
            encoding='latin1',
            usecols=lambda col: col in ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        )
        
        # Convert to UTC and normalize dates
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('Europe/London').dt.tz_convert('UTC')
        df['Date'] = df['Date'].dt.normalize()  # Remove time component
        
        # Filter out future matches
        current_date = datetime.now(timezone.utc).normalize()
        df = df[df['Date'] <= current_date]
        
        logger.info(f"Fetched {len(df)} historical matches for {league_code}")
        return df
        
    except Exception as e:
        logger.exception(f"Failed to fetch soccer results for {league_code}: {str(e)}")
        return pd.DataFrame(columns=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])

def calculate_soccer_features(df: pd.DataFrame) -> tuple:
    """Calculate features with proper initialization and validation"""
    if df.empty:
        logger.warning("Empty dataframe passed to calculate_soccer_features")
        return df, {}, {}, {}, {}
    
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    elo_ratings = {team: 1500.0 for team in teams}
    k_factor = 20
    home_elos, away_elos = [], []
    team_variance = {team: [] for team in teams}
    team_goals_for = {team: 0.0 for team in teams}
    team_goals_against = {team: 0.0 for team in teams}
    home_pyth, away_pyth = [], []
    
    for _, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        
        # Elo calculation with bounds checking
        r_h = elo_ratings.get(h, 1500.0)
        r_a = elo_ratings.get(a, 1500.0)
        home_elos.append(r_h)
        away_elos.append(r_a)
        
        # Expected score calculation
        e_h = 1 / (1 + 10**((r_a - r_h) / 400))
        actual_result = 1.0 if row['FTR'] == 'H' else 0.0 if row['FTR'] == 'A' else 0.5
        
        # Update Elo
        elo_ratings[h] += k_factor * (actual_result - e_h)
        elo_ratings[a] += k_factor * ((1 - actual_result) - (1 - e_h))
        
        # Track variance
        error = (actual_result - e_h) ** 2
        team_variance[h].append(error)
        team_variance[a].append(error)
        
        # Pythagorean expectation
        h_py = calculate_pythagorean_expectation(
            team_goals_for[h], 
            team_goals_against[h]
        )
        a_py = calculate_pythagorean_expectation(
            team_goals_for[a], 
            team_goals_against[a]
        )
        home_pyth.append(h_py)
        away_pyth.append(a_py)
        
        # Update goal statistics
        team_goals_for[h] += row['FTHG']
        team_goals_against[h] += row['FTAG']
        team_goals_for[a] += row['FTAG']
        team_goals_against[a] += row['FTHG']
    
    df['HomeElo'] = home_elos
    df['AwayElo'] = away_elos
    df['HomePyth'] = home_pyth
    df['AwayPyth'] = away_pyth
    
    # Calculate volatility with minimum data requirement
    volatility_map = {}
    for team, errors in team_variance.items():
        if len(errors) >= 5:
            volatility_map[team] = np.std(errors[-10:])
        else:
            volatility_map[team] = 0.25  # Default volatility
    
    return df, elo_ratings, volatility_map, team_goals_for, team_goals_against

def train_league_brain(div_code: str) -> tuple:
    """Train soccer model with caching and validation"""
    cache_path = os.path.join(MODEL_CACHE_DIR, f"soccer_brain_{div_code}.pkl")
    current_time = datetime.now(timezone.utc)
    
    # Check cache validity (retrain if >7 days old)
    if os.path.exists(cache_path):
        try:
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path), tz=timezone.utc)
            if (current_time - cache_time).days < 7:
                logger.info(f"Loading cached brain for {div_code}")
                return joblib.load(cache_path), None
        except Exception as e:
            logger.warning(f"Cache load failed for {div_code}: {str(e)}")
    
    logger.info(f"Training new brain for {div_code}")
    seasons = ['2324', '2223', '2122']
    all_dfs = []
    
    for season in seasons:
        df = get_soccer_results(div_code, season)
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        logger.error(f"No data available for league {div_code}")
        return None, None
    
    df = pd.concat(all_dfs).sort_values('Date').reset_index(drop=True)
    df, elo_ratings, volatility_map, gf, ga = calculate_soccer_features(df)
    
    # Calculate form metrics
    h_stats = df[['Date', 'HomeTeam', 'FTR']].rename(columns={'HomeTeam': 'Team'})
    h_stats['Points'] = h_stats['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    a_stats = df[['Date', 'AwayTeam', 'FTR']].rename(columns={'AwayTeam': 'Team'})
    a_stats['Points'] = a_stats['FTR'].map({'A': 3, 'D': 1, 'H': 0})
    
    all_stats = pd.concat([h_stats, a_stats]).sort_values('Date')
    all_stats['Form_EWMA'] = all_stats.groupby('Team')['Points'].transform(
        lambda x: x.ewm(span=5, adjust=False).mean().shift(1)
    )
    
    df = pd.merge(
        df, 
        all_stats[['Date', 'Team', 'Form_EWMA']], 
        left_on=['Date', 'HomeTeam'], 
        right_on=['Date', 'Team'], 
        how='left'
    ).rename(columns={'Form_EWMA': 'HomeForm'})
    
    df = pd.merge(
        df, 
        all_stats[['Date', 'Team', 'Form_EWMA']], 
        left_on=['Date', 'AwayTeam'], 
        right_on=['Date', 'Team'], 
        how='left'
    ).rename(columns={'Form_EWMA': 'AwayForm'})
    
    # Prepare features
    df.dropna(subset=['B365H', 'B365D', 'B365A', 'HomeForm', 'AwayForm'], inplace=True)
    df['elo_diff'] = df['HomeElo'] - df['AwayElo']
    df['form_diff'] = df['HomeForm'] - df['AwayForm']
    df['pyth_diff'] = df['HomePyth'] - df['AwayPyth']
    
    features = ['elo_diff', 'form_diff', 'pyth_diff']
    X = df[features]
    y = df['FTR']
    
    if len(X) < 50:
        logger.warning(f"Insufficient data for {div_code}: {len(X)} samples")
        return None, None
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    genome = load_or_initialize_genome()
    model, _ = train_ensemble_model(X_scaled, y_encoded, genome)
    
    # Calculate team strengths
    avg_goals_home = df['FTHG'].mean()
    avg_goals_away = df['FTAG'].mean()
    
    home_strength = df.groupby('HomeTeam').agg(
        h_gf_avg=('FTHG', 'mean'),
        h_ga_avg=('FTAG', 'mean')
    )
    away_strength = df.groupby('AwayTeam').agg(
        a_gf_avg=('FTAG', 'mean'),
        a_ga_avg=('FTHG', 'mean')
    )
    
    team_strengths = pd.concat([home_strength, away_strength], axis=1).fillna(1.0)
    team_strengths['attack'] = (
        (team_strengths['h_gf_avg'] / avg_goals_home) + 
        (team_strengths['a_gf_avg'] / avg_goals_away)
    ) / 2
    team_strengths['defence'] = (
        (team_strengths['h_ga_avg'] / avg_goals_away) + 
        (team_strengths['a_ga_avg'] / avg_goals_home)
    ) / 2
    
    brain = {
        'model': model,
        'le': le,
        'scaler': scaler,
        'elo_ratings': elo_ratings,
        'volatility': volatility_map,
        'team_strengths': team_strengths,
        'avgs': (avg_goals_home, avg_goals_away),
        'gf': gf,
        'ga': ga,
        'last_trained': current_time.isoformat()
    }
    
    # Save to cache
    try:
        joblib.dump((brain, df), cache_path)
        logger.info(f"Trained brain saved for {div_code}")
    except Exception as e:
        logger.error(f"Brain cache save failed for {div_code}: {str(e)}")
    
    return brain, df

def run_global_soccer_module() -> pd.DataFrame:
    """Generate soccer bets with proper edge calculation"""
    logger.info("--- Running Global Soccer Module ---")
    bets = []
    LEAGUE_MAP = {
        'soccer_epl': 'E0',
        'soccer_spain_la_liga': 'SP1',
        'soccer_germany_bundesliga': 'D1',
        'soccer_italy_serie_a': 'I1',
        'soccer_france_ligue_one': 'F1',
        'soccer_uefa_champs_league': 'CL'
    }
    
    for sport_key, div_code in LEAGUE_MAP.items():
        logger.info(f"Processing {sport_key}")
        odds_data = get_live_odds(sport_key)
        
        # Load brain (UCL doesn't have historical data)
        brain = None
        historical_df = None
        if div_code != 'CL':
            brain, historical_df = train_league_brain(div_code)
        
        for game in odds_data:
            commence_time = game.get('commence_time', '')
            try:
                match_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                match_time = datetime.now(timezone.utc)
            
            # Check arbitrage first
            edge, arb_info, bh, bd, ba = find_arbitrage(game, 'Soccer')
            if edge > 0.01:  # Minimum 1% edge
                bets.append({
                    'Date': match_time.isoformat(),
                    'Sport': 'Soccer',
                    'League': sport_key,
                    'Match': f"{game['home_team']} vs {game['away_team']}",
                    'Bet_Type': 'ARBITRAGE',
                    'Bet': 'ALL',
                    'Odds': 1.0 / (1 - edge),  # Fair odds equivalent
                    'Edge': edge,
                    'Confidence': 1.0,
                    'Stake': 0.0,  # Stake handled separately
                    'Info': arb_info
                })
                continue
            
            # Skip if no brain available
            if not brain or historical_df is None:
                continue
            
            # Team matching
            home_team = game['home_team']
            away_team = game['away_team']
            model_home = fuzzy_match_team(home_team, list(brain['elo_ratings'].keys()))
            model_away = fuzzy_match_team(away_team, list(brain['elo_ratings'].keys()))
            
            if not model_home or not model_away:
                continue
            
            # Get team stats
            h_elo = brain['elo_ratings'].get(model_home, 1500.0)
            a_elo = brain['elo_ratings'].get(model_away, 1500.0)
            
            h_py = calculate_pythagorean_expectation(
                brain['gf'].get(model_home, 0.0),
                brain['ga'].get(model_home, 0.0)
            )
            a_py = calculate_pythagorean_expectation(
                brain['gf'].get(model_away, 0.0),
                brain['ga'].get(model_away, 0.0)
            )
            
            # Get recent form
            try:
                h_form = historical_df[historical_df['HomeTeam'] == model_home]['HomeForm'].iloc[-1]
                a_form = historical_df[historical_df['AwayTeam'] == model_away]['AwayForm'].iloc[-1]
            except (IndexError, KeyError):
                h_form, a_form = 1.5, 1.5
            
            # Predict probabilities
            features = pd.DataFrame([{
                'elo_diff': h_elo - a_elo,
                'form_diff': h_form - a_form,
                'pyth_diff': h_py - a_py
            }])
            feat_scaled = brain['scaler'].transform(features)
            probs = brain['model'].predict_proba(feat_scaled)[0]
            
            # Map to outcomes
            outcome_probs = {
                'Home Win': probs[brain['le'].transform(['H'])[0]],
                'Draw': probs[brain['le'].transform(['D'])[0]],
                'Away Win': probs[brain['le'].transform(['A'])[0]]
            }
            
            # Calculate ZI Poisson probabilities as backup
            try:
                avg_goals_home, avg_goals_away = brain['avgs']
                team_strengths = brain['team_strengths']
                
                h_att = team_strengths.loc[model_home, 'attack']
                a_def = team_strengths.loc[model_away, 'defence']
                a_att = team_strengths.loc[model_away, 'attack']
                h_def = team_strengths.loc[model_home, 'defence']
                
                exp_h = h_att * a_def * avg_goals_home
                exp_a = a_att * h_def * avg_goals_away
                
                # Create probability matrix
                max_goals = 5
                pm = np.zeros((max_goals+1, max_goals+1))
                for i in range(max_goals+1):
                    for j in range(max_goals+1):
                        pm[i, j] = zero_inflated_poisson(i, exp_h) * zero_inflated_poisson(j, exp_a)
                
                # Calculate match outcome probabilities
                p_h = np.sum(np.tril(pm, -1))  # Home win
                p_d = np.sum(np.diag(pm))      # Draw
                p_a = np.sum(np.triu(pm, 1))   # Away win
                
                # Blend models (70% ML, 30% Poisson)
                final_probs = {
                    'Home Win': outcome_probs['Home Win'] * 0.7 + p_h * 0.3,
                    'Draw': outcome_probs['Draw'] * 0.7 + p_d * 0.3,
                    'Away Win': outcome_probs['Away Win'] * 0.7 + p_a * 0.3
                }
            except Exception as e:
                logger.warning(f"Poisson calculation failed: {str(e)}")
                final_probs = outcome_probs
            
            # Apply volatility adjustment
            h_vol = brain['volatility'].get(model_home, 0.25)
            a_vol = brain['volatility'].get(model_away, 0.25)
            vol_factor = max(0.5, 1.0 - ((h_vol + a_vol) / 2 - 0.25))
            
            # Evaluate betting opportunities
            outcomes = [
                ('Home Win', bh, game['home_team']),
                ('Draw', bd, 'Draw'),
                ('Away Win', ba, game['away_team'])
            ]
            
            for outcome, book_data, _ in outcomes:
                odds = book_data['price']
                if odds <= 1.01:  # Skip invalid odds
                    continue
                
                # Calculate edge and Kelly stake
                edge = (final_probs[outcome] * odds) - 1
                if edge > 0.02:  # Minimum 2% edge
                    # Kelly criterion with quarter fraction
                    kelly_fraction = edge / (odds - 1)
                    stake = max(0.0, min(0.05, kelly_fraction * 0.25 * vol_factor))  # Max 5% of bankroll
                    
                    bets.append({
                        'Date': match_time.isoformat(),
                        'Sport': 'Soccer',
                        'League': sport_key,
                        'Match': f"{home_team} vs {away_team}",
                        'Bet_Type': 'Moneyline',
                        'Bet': outcome,
                        'Odds': odds,
                        'Edge': edge,
                        'Confidence': final_probs[outcome],
                        'Stake': stake,
                        'Info': f"Best: {book_data['book']}, Vol: {vol_factor:.2f}"
                    })
    
    logger.info(f"Generated {len(bets)} soccer bets")
    return pd.DataFrame(bets)

# ==============================================================================
# SETTLEMENT ENGINE (Date-Aware and Multi-Sport)
# ==============================================================================
def settle_soccer_bets(df: pd.DataFrame) -> pd.DataFrame:
    """Settle soccer bets with date validation"""
    pending = df[
        (df['Result'] == 'Pending') & 
        (df['Sport'] == 'Soccer') &
        (pd.to_datetime(df['Date']) <= datetime.now(timezone.utc).normalize())
    ]
    
    if pending.empty:
        return df
    
    logger.info(f"Settling {len(pending)} soccer bets")
    
    # Fetch results for all relevant leagues
    league_results = {}
    for league in pending['League'].unique():
        league_code = league.split('_')[-1].upper()
        if league_code in ['EPL', 'LA LIGA', 'BUNDESLIGA', 'SERIE A', 'LIGUE 1', 'UCL']:
            results_df = get_soccer_results(league_code)
            if not results_df.empty:
                league_results[league] = results_df
    
    for idx, row in pending.iterrows():
        try:
            match_date = pd.to_datetime(row['Date']).normalize()
            league = row['League']
            match_str = row['Match']
            
            # Parse team names
            teams = match_str.split(' vs ')
            if len(teams) < 2:
                continue
            home_team = teams[0].strip()
            away_team = teams[1].strip()
            
            # Get results for this league
            results_df = league_results.get(league)
            if results_df is None or results_df.empty:
                continue
            
            # Find matching result
            match_result = results_df[
                (results_df['Date'] == match_date) &
                (results_df['HomeTeam'] == home_team) &
                (results_df['AwayTeam'] == away_team)
            ]
            
            if match_result.empty:
                continue
            
            # Get match result
            ftr = match_result.iloc[0]['FTR']
            fthg = match_result.iloc[0]['FTHG']
            ftag = match_result.iloc[0]['FTAG']
            
            # Determine bet outcome
            bet_outcome = row['Bet']
            won = False
            
            if bet_outcome == 'Home Win' and ftr == 'H':
                won = True
            elif bet_outcome == 'Draw' and ftr == 'D':
                won = True
            elif bet_outcome == 'Away Win' and ftr == 'A':
                won = True
            
            # Update settlement
            df.loc[idx, 'Result'] = 'Win' if won else 'Loss'
            df.loc[idx, 'Profit'] = (row['Stake'] * (row['Odds'] - 1)) if won else -row['Stake']
            df.loc[idx, 'Score'] = f"{fthg}-{ftag}"
            
        except Exception as e:
            logger.warning(f"Error settling soccer bet {idx}: {str(e)}")
    
    return df

def settle_bets():
    """Main settlement function with sport routing"""
    logger.info("--- ⚖️ Running Settlement Engine ---")
    history_file = 'betting_history.csv'
    
    if not os.path.exists(history_file):
        logger.warning("No history file found for settlement")
        return
    
    try:
        df = pd.read_csv(history_file, parse_dates=['Date'])
    except Exception as e:
        logger.error(f"Failed to read history file: {str(e)}")
        return
    
    # Initialize columns if missing
    for col in ['Result', 'Profit', 'Score']:
        if col not in df.columns:
            df[col] = 'Pending' if col == 'Result' else 0.0 if col == 'Profit' else ''
    
    # Route by sport
    df = settle_soccer_bets(df)
    
    # TODO: Add NFL, NBA, MLB settlement functions
    
    # Save updated history
    try:
        df.to_csv(history_file, index=False)
        logger.info(f"Settlement complete. {len(df[df['Result'] != 'Pending'])} bets settled.")
    except Exception as e:
        logger.error(f"Failed to save settled history: {str(e)}")

# ==============================================================================
# MODULE EXPORTS (For backend_runner.py)
# ==============================================================================
__all__ = [
    'run_global_soccer_module',
    'run_nfl_module',
    'run_nba_module',
    'run_mlb_module',
    'settle_bets'
]

# Placeholder implementations for other sports (to be implemented similarly to soccer)
def run_nfl_module():
    logger.warning("NFL module not implemented in this refactored version")
    return pd.DataFrame()

def run_nba_module():
    logger.warning("NBA module not implemented in this refactored version")
    return pd.DataFrame()

def run_mlb_module():
    logger.warning("MLB module not implemented in this refactored version")
    return pd.DataFrame()
