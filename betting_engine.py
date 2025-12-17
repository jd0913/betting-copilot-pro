# betting_engine.py
# The Core Logic: AI Models, Data Fetching, Settlement, Feature Engineering
# v85.1 (API-Only Settlement Integration)
# FIX: Updated settle_bets to call utils.settle_bets_with_api_scores

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
from requests.exceptions import RequestException, HTTPError, ConnectionError
from requests.adapters import HTTPAdapter, Retry 
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta, timezone
import os
import json
import random
from fuzzywuzzy import process
from textblob import TextBlob
import logging
from pathlib import Path
import sys 

# Optional import for config - handle gracefully
try:
    import config
except ImportError:
    # Create a dummy config if not available
    class DummyConfig:
        pass
    config = DummyConfig()

# ==============================================================================
# LOGGING SETUP (Professional standard)
# ==============================================================================
logger = logging.getLogger("BettingEngine")
logger.setLevel(logging.INFO)

# ==============================================================================
# PLACEHOLDER MODEL RUNS (MUST BE IMPLEMENTED BY USER)
# ==============================================================================
def run_global_soccer_module():
    logger.warning("Soccer module not implemented in this refactored version. Returning empty DataFrame.")
    return pd.DataFrame(columns=['Date', 'Date_Generated', 'Sport', 'League', 'Match', 'Bet_Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake'])

def run_nfl_module():
    logger.warning("NFL module not implemented in this refactored version. Returning empty DataFrame.")
    return pd.DataFrame(columns=['Date', 'Date_Generated', 'Sport', 'League', 'Match', 'Bet_Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake'])

def run_nba_module():
    logger.warning("NBA module not implemented in this refactored version. Returning empty DataFrame.")
    return pd.DataFrame(columns=['Date', 'Date_Generated', 'Sport', 'League', 'Match', 'Bet_Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake'])

def run_mlb_module():
    logger.warning("MLB module not implemented in this refactored version. Returning empty DataFrame.")
    return pd.DataFrame(columns=['Date', 'Date_Generated', 'Sport', 'League', 'Match', 'Bet_Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake'])

# ==============================================================================
# SETTLEMENT ENGINE
# ==============================================================================

def settle_bets():
    """Reads betting history, settles pending bets using the API, and saves the updated history."""
    
    # Path to the history file
    history_file = Path("betting_history.csv")
    
    # Load existing history
    try:
        df = pd.read_csv(history_file)
        if 'Date' in df.columns:
            df['Date_Obj'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
    except Exception as e:
        logger.error(f"Failed to read history file: {str(e)}")
        return
    
    # Initialize columns if missing
    for col in ['Result', 'Profit', 'Score', 'Date_Obj']:
        if col not in df.columns:
            df[col] = 'Pending' if col == 'Result' else 0.0 if col == 'Profit' else ''

    # Import and call the new API-ONLY settlement function
    try:
        import utils
        if hasattr(utils, 'settle_bets_with_api_scores'):
            # CRITICAL CHANGE: Call the new API-ONLY function
            df = utils.settle_bets_with_api_scores(df)
            logger.info(f"API-ONLY settlement applied to {len(df)} records.")
        else:
            logger.warning("Warning: settle_bets_with_api_scores function not found in utils. Settlement skipped.")
    except ImportError as e:
        logger.error(f"Warning: utils module import failed: {str(e)}. Settlement skipped.")
    except Exception as e:
        logger.critical(f"Error during API-based settlement: {str(e)}. Settlement skipped.", exc_info=True)
    
    # Save the updated history
    try:
        df.drop(columns=['Date_Obj'], inplace=True, errors='ignore') # Remove temp column before saving
        df.to_csv(history_file, index=False)
        settled_count = len(df[df['Result'].isin(['Win', 'Loss', 'Push', 'Auto-Settled'])])
        logger.info(f"Settlement complete. {settled_count} bets settled using API score lookup.")
    except Exception as e:
        logger.error(f"Failed to save settled history: {str(e)}")

# ==============================================================================
# MODULE EXPORTS
# ==============================================================================
__all__ = [
    'run_global_soccer_module',
    'run_nfl_module',
    'run_nba_module',
    'run_mlb_module',
    'settle_bets'
]
