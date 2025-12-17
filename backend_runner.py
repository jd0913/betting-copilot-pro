# backend_runner-3.py
# Betting Co-Pilot Pro - v85.1 (API-Only Backend - Hardcoded Key)
# FIX: Removed Google/Scraping configurations.
# FIX: Removed API Key check (now hardcoded in utils-2.py).

import pandas as pd
import numpy as np
import betting_engine 
import requests
from datetime import datetime, timedelta, timezone
import os
import logging
import time
import random
from pathlib import Path
import sys 

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BackendRunner")

# --- CONFIGURATION & SECURITY SETUP ---
# Discord webhook - set this in your environment variables or GitHub Secrets
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "").strip() 
if not DISCORD_WEBHOOK or "PASTE_YOUR" in DISCORD_WEBHOOK:
    DISCORD_WEBHOOK = None
    logger.warning("Discord webhook not configured via environment variable.")

logger.info("Settlement is configured for API-Only mode (key hardcoded in utils-2.py).")

# Define consistent schema (avoids Streamlit crashes on missing columns)
BET_SCHEMA = {
    'Date': 'datetime64[ns, UTC]',
    'Date_Generated': 'datetime64[ns, UTC]',
    'Sport': 'string',
    'League': 'string',
    'Match': 'string',
    'Bet_Type': 'string',
    'Bet': 'string',
    'Odds': 'float64',
    'Edge': 'float64',
    'Confidence': 'float64',
    'Stake': 'float64',
    'Score': 'string', # For final score
    'Result': 'string', # Win, Loss, Push, Pending
    'Profit': 'float64', # Profit/Loss amount
}

# --- Shared File I/O Functions (Simulated) ---

def save_latest_bets(df):
    """Saves the DataFrame to latest_bets.csv."""
    try:
        # Assuming Date_Obj is added by utils.load_data or in memory
        df_to_save = df.copy().drop(columns=['Date_Obj'], errors='ignore')
        df_to_save.to_csv("latest_bets.csv", index=False)
        logger.info(f"Saved {len(df_to_save)} latest bets.")
        return df_to_save
    except Exception as e:
        logger.error(f"Failed to save latest bets: {e}")
        return pd.DataFrame()

def load_history():
    """Loads the betting history from betting_history.csv."""
    try:
        df = pd.read_csv("betting_history.csv")
        if 'Date' in df.columns:
            df['Date_Obj'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
        return df
    except FileNotFoundError:
        logger.warning("betting_history.csv not found. Starting new history.")
        return pd.DataFrame(columns=list(BET_SCHEMA.keys()) + ['Date_Obj'])
    except Exception as e:
        logger.error(f"Failed to load history: {e}")
        return pd.DataFrame(columns=list(BET_SCHEMA.keys()) + ['Date_Obj'])

def archive_to_history(latest_df):
    """Placeholder: Relies on betting_engine.settle_bets to handle file I/O."""
    pass 

# --- Discord Notification Function ---

def send_discord_alert(df):
    """Sends a notification to Discord about the daily bets."""
    if not DISCORD_WEBHOOK or df.empty:
        return
    
    new_bets = df[df['Result'] == 'Pending']
    if new_bets.empty:
        return
        
    message = "🚀 **NEW VALUE BETS IDENTIFIED** 🚀\n\n"
    for _, row in new_bets.iterrows():
        message += f"**{row['Sport']}**: {row['Match']}\n"
        message += f"➡️ Bet: {row['Bet']} @ {row['Odds']:.2f}\n"
        message += f"🔥 Edge: {row['Edge']:.2f}%, Conf: {row['Confidence']:.2f}%\n"
        message += f"💰 Stake: ${row['Stake']:.2f} (Kelly)\n"
        message += f"📅 Time: {row['Date']}\n\n"

    try:
        requests.post(DISCORD_WEBHOOK, json={"content": message}, timeout=5)
        logger.info("Discord alert sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send Discord alert: {e}")

# --- Main Runner Logic ---

def run_backend_analysis():
    logger.info(f"--- Betting Co-Pilot Pipeline Starting ({datetime.now(timezone.utc).isoformat()}) ---")
    
    # PHASE 1 & 2: MODEL ANALYSIS AND BET SELECTION
    logger.info("🧠 PHASE 1 & 2: MODEL ANALYSIS AND BET SELECTION (Placeholder)")
    
    all_bets = []
    
    try:
        import betting_engine
        
        # NOTE: Assumes betting_engine functions return DataFrames
        soccer_bets = betting_engine.run_global_soccer_module()
        nfl_bets = betting_engine.run_nfl_module()
        
        all_bets.extend([soccer_bets, nfl_bets])

        all_bets = [df for df in all_bets if not df.empty]
        combined_bets = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error during model run: {e}. Proceeding to settlement with available data.")
        combined_bets = pd.DataFrame() 

    logger.info(f"📝 PHASE 2: COMBINING RESULTS - {len(combined_bets)} new bets found.")
    
    # Save new bets to the 'latest_bets.csv'
    latest_bets = save_latest_bets(combined_bets)
    
    # PHASE 3: Settlement
    logger.info("📅 PHASE 3: SETTLEMENT (API-ONLY)")
    
    # This calls betting_engine.settle_bets, which now uses the API-only logic in utils-2.py
    betting_engine.settle_bets() 
    
    if not latest_bets.empty:
        logger.info(f"🎉 ANALYSIS COMPLETE: {len(latest_bets)} value bets found. Settlement used API-Only strategy.")
    else:
        logger.info("😴 ANALYSIS COMPLETE: No value bets found today.")
    
    # PHASE 4: Notifications
    logger.info("🔔 PHASE 4: SENDING NOTIFICATIONS")
    send_discord_alert(latest_bets)

# ==============================================================================
# EXECUTION GUARD
# ==============================================================================
if __name__ == "__main__":
    try:
        run_backend_analysis()
    except SystemExit:
        logger.info("Pipeline exited normally")
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
    except Exception as e:
        logger.critical(f"UNCAUGHT EXCEPTION: {str(e)}", exc_info=True)
        sys.exit(1)
