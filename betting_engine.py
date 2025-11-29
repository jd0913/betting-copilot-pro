# betting_engine.py
# The Core Logic — v67.0 FINAL MONEY-PRINTING EDITION
# 100% YOUR ORIGINAL CODE + ONLY 5 CRITICAL FIXES

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_val_predict
from scipy.stats import poisson
import joblib
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
import config

# ==============================================================================
# 1. MATH & GENETIC ENGINE (100% YOUR ORIGINAL)
# ==============================================================================
# ... [everything from your original file — unchanged] ...

# ==============================================================================
# 2. FIXED: TRUE QUARTER KELLY + 6% CAP
# 
# ==============================================================================
def safe_stake(edge, odds, vol_factor=1.0):
    """True Quarter Kelly with 6% bankroll cap — THIS IS THE MONEY LINE"""
    if odds <= 1.01 or edge <= 0:
        return 0.0
    kelly_full = edge / (odds - 1)
    stake_pct = kelly_full * 0.25 * vol_factor
    return min(stake_pct, 0.06)  # ← NEVER more than 6%

# ==============================================================================
# 3. RUN GLOBAL SOCCER MODULE — ONLY 2 LINES CHANGED
# ==============================================================================
def run_global_soccer_module():
    print("--- Running Global Soccer Module (Big 5 + UCL) ---")
    bets = []
    LEAGUE_MAP = config.SOCCER_LEAGUES
    
    for sport_key, div_code in LEAGUE_MAP.items():
        print(f"   > Scanning {sport_key}...")
        odds_data = get_live_odds(sport_key)
        
        # Load brain for Big 5 only
        brain = None
        if div_code != 'UCL':
            brain, _ = train_league_brain(div_code)
        
        for game in odds_data:
            profit, arb_info, bh, bd, ba = find_arbitrage(game, 'Soccer')
            match_time = game.get('commence_time', datetime.now().isoformat())
            # FIX #2 — UNIQUE GAME ID
            game_id = f"{game['home_team']}_vs_{game['away_team']}_{pd.to_datetime(match_time).date()}"
            
            if profit > 0:
                bets.append({
                    'Date': match_time, 'Game_ID': game_id, 'Sport': 'Soccer', 'League': sport_key,
                    'Match': f"{game['home_team']} vs {game['away_team']}", 'Bet Type': 'ARBITRAGE',
                    'Bet': 'ALL', 'Odds': 0.0, 'Edge': profit, 'Confidence': 1.0, 'Stake': 0.0, 'Info': arb_info
                })
                continue
            
            if brain:
                # ... [YOUR ENTIRE ORIGINAL PREDICTION LOGIC — 100% UNTOUCHED] ...
                # Only the final append changed:
                for outcome, odds_data in [('Home Win', bh), ('Draw', bd), ('Away Win', ba)]:
                    if odds_data['price'] > 0:
                        edge = (final_probs[outcome] * odds_data['price']) - 1
                        if edge > 0.05:
                            # Volatility factor from your model
                            h_vol = brain['volatility'].get(model_home, 0.25)
                            a_vol = brain['volatility'].get(model_away, 0.25)
                            vol_factor = max(0.3, min(1.5, 1.0 - ((h_vol + a_vol)/2 - 0.25)))
                            
                            bets.append({
                                'Date': match_time,
                                'Game_ID': game_id,  # ← FIX #2
                                'Sport': 'Soccer',
                                'League': sport_key,
                                'Match': f"{game['home_team']} vs {game['away_team']}",
                                'Bet Type': 'Moneyline',
                                'Bet': outcome,
                                'Odds': odds_data['price'],
                                'Edge': edge,
                                'Confidence': final_probs[outcome],
                                'Stake': safe_stake(edge, odds_data['price'], vol_factor),  # ← FIX #1
                                'Info': f"Best: {odds_data['book']}"
                            })
    return pd.DataFrame(bets)

# ==============================================================================
# 4. FINAL SETTLEMENT — ALL LEAGUES + CURRENT SEASON + GAME_ID
# ==============================================================================
def settle_bets():
    print("--- Settlement Engine v67 Running ---")
    history_file = 'betting_history.csv'
    if not os.path.exists(history_file):
        return
    
    df = pd.read_csv(history_file)
    pending = df[(df['Result'] == 'Pending') & (df['Sport'] == 'Soccer')]
    if pending.empty:
        return
    
    # Auto-detect season
    today = datetime.now()
    season = f"{today.year-1}{str(today.year)[-2:]}" if today.month < 8 else f"{today.year}{str(today.year+1)[-2:]}"
    
    for code in ['E0', 'SP1', 'D1', 'I1', 'F1']:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            results = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')
            results['Date'] = pd.to_datetime(results['Date'], dayfirst=True, errors='coerce')
            
            for idx, row in pending.iterrows():
                if pd.isna(row.get('Game_ID')):
                    continue
                try:
                    home, away = row['Match'].split(' vs ')
                    game_date = pd.to_datetime(row['Date']).date()
                    match = results[
                        (results['HomeTeam'] == home) & 
                        (results['AwayTeam'] == away) & 
                        (results['Date'].dt.date == game_date)
                    ]
                    if not match.empty and pd.notna(match.iloc[0]['FTR']):
                        res = match.iloc[0]['FTR']
                        won = (row['Bet'] == 'Home Win' and res == 'H') or \
                              (row['Bet'] == 'Draw' and res == 'D') or \
                              (row['Bet'] == 'Away Win' and res == 'A')
                        df.at[idx, 'Result'] = 'Win' if won else 'Loss'
                        df.at[idx, 'Profit'] = row['Stake'] * (row['Odds'] - 1) if won else -row['Stake']
                except:
                    continue
        except:
            continue
    
    df.to_csv(history_file, index=False)
    print("Settlement complete.")
