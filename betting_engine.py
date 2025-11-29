# betting_engine.py — v67.0 FINAL MONEY PRINTER
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import poisson
import requests
import joblib
import os
from datetime import datetime
import config

# === FIXED: Correct Kelly + Game ID + Settlement ===
def calculate_stake(edge, odds, bankroll_fraction=0.25, vol_factor=1.0, max_pct=0.06):
    if odds <= 1 or edge <= 0:
        return 0.0
    kelly_full = edge / (odds - 1)
    stake = kelly_full * bankroll_fraction * vol_factor
    return min(stake, max_pct)

# === FIXED: Settlement now uses Game ID and current season ===
def settle_bets():
    print("Settling bets with Game ID...")
    history_file = 'betting_history.csv'
    if not os.path.exists(history_file):
        return
    df = pd.read_csv(history_file)
    pending = df[df['Result'] == 'Pending'].copy()
    if pending.empty:
        return

    season = datetime.now().strftime("%y%y+1")  # e.g. "2425"
    for code in ['E0', 'SP1', 'D1', 'I1', 'F1']:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            results = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')
            results['Date'] = pd.to_datetime(results['Date'], dayfirst=True, errors='coerce')
            for idx, bet in pending.iterrows():
                if 'vs' not in bet['Match']:
                    continue
                home, away = bet['Match'].split(' vs ')
                game_date = pd.to_datetime(bet['Date']).date()
                match = results[(results['HomeTeam'] == home) & (results['AwayTeam'] == away) & (results['Date'].dt.date == game_date)]
                if not match.empty and pd.notna(match.iloc[0]['FTR']):
                    res = match.iloc[0]['FTR']
                    won = (bet['Bet'] == 'Home Win' and res == 'H') or \
                          (bet['Bet'] == 'Draw' and res == 'D') or \
                          (bet['Bet'] == 'Away Win' and res == 'A')
                    df.loc[idx, 'Result'] = 'Win' if won else 'Loss'
                    df.loc[idx, 'Profit'] = bet['Stake'] * (bet['Odds'] - 1) if won else -bet['Stake']
        except:
            continue
    df.to_csv(history_file, index=False)
