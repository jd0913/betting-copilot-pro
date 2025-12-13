# backend_runner.py
# Betting Co-Pilot Pro - v82.0 (Google-Only Backend - NO Config)
# FIX: Removed all config.py dependencies
# FIX: Integrated Google-based data fetching and settlement

import pandas as pd
import numpy as np
import betting_engine # This module now handles all Google scraping
import requests
from datetime import datetime, timedelta, timezone
import os
import logging
import time
import random
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BackendRunner")

# --- HARDCODED PARAMETERS (Replaces config.py) ---
# Discord webhook - set this in your environment variables or Streamlit secrets
# Example for local dev: os.environ["DISCORD_WEBHOOK"] = "your_webhook_url"
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "") # Or get from Streamlit secrets if deployed
if not DISCORD_WEBHOOK or "PASTE_YOUR" in DISCORD_WEBHOOK:
    DISCORD_WEBHOOK = None # Disable if not set or invalid
    logger.warning("Discord webhook not configured via environment variable.")

# Google scraping parameters
GOOGLE_SCRAPE_DELAY_MIN = 1
GOOGLE_SCRAPE_DELAY_MAX = 3
GOOGLE_REQUEST_TIMEOUT = 15
GOOGLE_MAX_RETRIES = 2
GOOGLE_SEARCH_LIMIT_PER_DAY = 100

# Define consistent schema (avoids Streamlit crashes on missing columns)
BET_SCHEMA = {
    'Date': 'datetime64[ns, UTC]',
    'Date_Generated': 'datetime64[ns, UTC]',
    'Sport': 'string',
    'League': 'string',
    'Match': 'string',
    'Bet_Type': 'string', # Renamed from 'Bet Type' for consistency
    'Bet': 'string',
    'Odds': 'float64',
    'Edge': 'float64',
    'Confidence': 'float64',
    'Stake': 'float64',
    'Info': 'string',
    'Result': 'category',
    'Profit': 'float64',
    'Score': 'string'
}

def validate_bet_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe matches expected schema with proper dtypes"""
    for col, dtype in BET_SCHEMA.items():
        if col not in df.columns:
            if "datetime" in dtype:
                df[col] = pd.NaT
            elif "float" in dtype:
                df[col] = 0.0
            elif "category" in dtype:
                df[col] = pd.Categorical([])
            else:
                df[col] = ""
    
    for col, dtype in BET_SCHEMA.items():
        try:
            if "datetime" in dtype:
                df[col] = pd.to_datetime(df[col], utc=True)
            elif "category" in dtype:
                df[col] = pd.Categorical(df[col])
            else:
                df[col] = df[col].astype(dtype)
        except Exception as e:
            logger.warning(f"Schema conversion failed for {col}: {str(e)}")
            if "float" in dtype:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    return df[list(BET_SCHEMA.keys())] # Enforce column order

def send_discord_alert(df: pd.DataFrame):
    """Send formatted alert with safety checks."""
    if not DISCORD_WEBHOOK:
        logger.info("Discord alerts disabled - no webhook configured.")
        return

    try:
        if df.empty or len(df[df['Edge'] > 0.05]) == 0:
            msg = "🤖 **Betting Co-Pilot**\nNo high-value bets found via Google scraping today. Markets look efficient."
        else:
            top_bets = df[df['Edge'] > 0.05].nlargest(5, 'Edge')
            msg = "🚀 **Betting Co-Pilot - Top Value Bets (Google)**\n"
            
            for _, row in top_bets.iterrows():
                sport_icon = {
                    "Soccer": "⚽", "NFL": "🏈", "NBA": "🏀", "MLB": "⚾"
                }.get(row['Sport'], "🎲")
                
                edge_pct = f"{row['Edge']:.1%}"
                stake_pct = f"{row['Stake']:.1%}" if row['Stake'] > 0 else "N/A"
                
                bet_line = (
                    f"{sport_icon} **{row['Match']}**\n"
                    f"→ {row['Bet']} @ {row['Odds']:.2f}\n"
                    f"📈 Edge: {edge_pct} | 💰 Stake: {stake_pct}\n\n"
                )
                
                if len(msg) + len(bet_line) > 1900:
                    msg += "... *(truncated for Discord limits)*"
                    break
                msg += bet_line
        
        # Safety: Never expose internal paths/secrets in alerts
        safe_msg = msg.replace(str(Path.home()), "[USER_HOME]").replace(str(Path.cwd()), "[WORKDIR]")
        
        response = requests.post(
            DISCORD_WEBHOOK,
            json={"content": safe_msg[:1999]},  # Hard truncate
            timeout=5  # Critical: Prevent hanging on failed requests
        )
        response.raise_for_status()
        logger.info(f"Discord alert sent: {len(safe_msg)} chars")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Discord alert failed: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected error in Discord alert")

def archive_to_history(new_bets: pd.DataFrame):
    """Safely append to historical data with duplicate prevention."""
    history_path = Path("betting_history.csv")
    
    try:
        if history_path.exists():
            history = pd.read_csv(history_path, parse_dates=['Date', 'Date_Generated'])
            history = pd.concat([history, new_bets], ignore_index=True)
            
            # Deduplicate using critical fields (prevents re-betting same events)
            dedup_cols = ['Date_Generated', 'Sport', 'Match', 'Bet']
            history = history.sort_values('Date_Generated', ascending=False)
            history = history.drop_duplicates(subset=dedup_cols, keep='first')
        else:
            history = new_bets.copy()
        
        # Validate schema before saving
        history = validate_bet_schema(history)
        history.to_csv(history_path, index=False)
        logger.info(f"Archived {len(new_bets)} bets to history ({len(history)} total)")
        
    except Exception as e:
        logger.exception("History archiving failed - creating backup")
        # Emergency backup to prevent data loss
        new_bets.to_csv(f"emergency_backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv", index=False)
        raise

def save_latest_bets(bets_df: pd.DataFrame):
    """Save current recommendations with schema enforcement."""
    try:
        # Create empty dataframe with schema if no bets
        if bets_df.empty:
            bets_df = pd.DataFrame(columns=BET_SCHEMA.keys())
        
        # Standardize UTC timestamps
        current_time = datetime.now(timezone.utc)
        bets_df['Date_Generated'] = current_time
        if 'Date' not in bets_df.columns:
            bets_df['Date'] = current_time
        
        # Initialize tracking columns
        bets_df['Result'] = bets_df.get('Result', 'Pending')
        bets_df['Profit'] = bets_df.get('Profit', 0.0)
        bets_df['Score'] = bets_df.get('Score', '')
        
        # Validate and save
        bets_df = validate_bet_schema(bets_df)
        bets_df.to_csv("latest_bets.csv", index=False)
        logger.info(f"Saved {len(bets_df)} latest bets")
        return bets_df
        
    except Exception as e:
        logger.exception("Critical failure saving latest bets")
        raise

def run_backend_analysis():
    """Orchestrates full betting analysis pipeline with error isolation."""
    logger.info("="*60)
    logger.info("🚀 STARTING DAILY BETTING ANALYSIS PIPELINE (Google-Only, No Config)")
    logger.info(f"UTC Timestamp: {datetime.now(timezone.utc).isoformat()}")
    
    try:
        # PHASE 1: Settle existing bets (Google-based - relies on betting_engine)
        try:
            logger.info("🏁 PHASE 1: SETTLING EXISTING BETS (Google-based)")
            betting_engine.settle_bets() # This calls the Google-based function in betting_engine.py
        except Exception as e:
            logger.exception("BET SETTLEMENT (Google) FAILED - continuing pipeline")
        
        # PHASE 2: Generate new recommendations (Google-based - relies on betting_engine)
        logger.info("🧠 PHASE 2: GENERATING BET RECOMMENDATIONS (Google-based)")
        sport_modules = {
            "Soccer": betting_engine.run_global_soccer_module, # Updated module in betting_engine.py
            "NFL": betting_engine.run_nfl_module, # Placeholder/updated module in betting_engine.py
            "NBA": betting_engine.run_nba_module, # Placeholder/updated module in betting_engine.py
            "MLB": betting_engine.run_mlb_module  # Placeholder/updated module in betting_engine.py
        }
        
        all_bets = []
        for sport, module_fn in sport_modules.items():
            try:
                logger.info(f"⚽ Running {sport} module (Google-based)...")
                # Add a small delay between sport modules to be respectful to Google
                time.sleep(random.uniform(GOOGLE_SCRAPE_DELAY_MIN, GOOGLE_SCRAPE_DELAY_MAX))
                
                bets = module_fn() # Calls the Google-dependent function in betting_engine.py
                
                # Check if the function returned a valid DataFrame
                if isinstance(bets, pd.DataFrame) and not bets.empty:
                    logger.info(f"✅ {sport}: {len(bets)} bets found via Google scraping.")
                    all_bets.append(bets)
                else:
                    logger.warning(f"⚠️ {sport}: No bets found or invalid return type from Google scraping.")
                    
            except Exception as e:
                logger.exception(f"❌ {sport} MODULE (Google-based) FAILED: {e}")
                # Continue to the next sport module despite the error
                time.sleep(random.uniform(GOOGLE_SCRAPE_DELAY_MIN, GOOGLE_SCRAPE_DELAY_MAX))
        
        # PHASE 3: Consolidate and save results
        logger.info("💾 PHASE 3: SAVING RESULTS")
        combined_bets = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
        
        # Save latest recommendations (always create file for Streamlit)
        latest_bets = save_latest_bets(combined_bets)
        
        # Archive to history (only if we have actual bets)
        if not latest_bets.empty:
            archive_to_history(latest_bets)
            logger.info(f"🎉 ANALYSIS COMPLETE: {len(latest_bets)} value bets found via Google")
        else:
            logger.info("😴 ANALYSIS COMPLETE: No value bets found via Google today")
        
        # PHASE 4: Notifications
        logger.info("🔔 PHASE 4: SENDING NOTIFICATIONS")
        send_discord_alert(latest_bets)
        
    except Exception as e:
        logger.exception("CRITICAL FAILURE IN MAIN PIPELINE")
        # Emergency notification for complete pipeline failure
        if DISCORD_WEBHOOK:
            try:
                requests.post(DISCORD_WEBHOOK, json={
                    "content": f"🚨 **CRITICAL FAILURE** in betting pipeline\n`{str(e)}`"
                }, timeout=3)
            except:
                pass
        raise

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
