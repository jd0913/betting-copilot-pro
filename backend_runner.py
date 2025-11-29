# backend_runner.py
# The Master Orchestrator — Runs Every 30 Minutes
# v67.0 FINAL — NOW BULLETPROOF
# Fixes Applied:
#   1. True Quarter Kelly (you were doing FULL KELLY before)
#   2. Game_ID added to all bets
#   3. Settlement runs BEFORE new bets (prevents double-betting settled games)
#   4. Discord webhook with real-time P&L + bankroll protection

import pandas as pd
import time
from datetime import datetime
import os
import requests
import json
import config
from betting_engine import run_global_soccer_module, settle_bets, safe_stake

# ==============================================================================
# BANKROLL & SETTINGS (EDIT ONCE)
# ==============================================================================
BANKROLL = 25000.0          # ← YOUR CURRENT BANKROLL
KELLY_MULTIPLIER = 1.0      # 1.0 = Quarter Kelly, 0.5 = Eighth Kelly, 2.0 = Half Kelly
DISCORD_WEBHOOK = config.API_CONFIG["DISCORD_WEBHOOK"]

# ==============================================================================
# DISCORD NOTIFIER (REAL-TIME ALERTS)
# ==============================================================================
def send_discord(message):
    if "PASTE_YOUR" in DISCORD_WEBHOOK or not DISCORD_WEBHOOK:
        print(f"[DISCORD] {message}")
        return
    payload = {"content": message, "username": "Betting Co-Pilot Pro v67"}
    try:
        requests.post(DISCORD_WEBHOOK, json=payload)
    except:
        pass

# ==============================================================================
# MAIN LOOP — THE MONEY ENGINE
# ==============================================================================
def run_cycle():
    print(f"\n=== CYCLE STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # 1. SETTLE OLD BETS FIRST (CRITICAL — prevents re-betting settled games)
    settle_bets()
    
    # 2. GENERATE NEW BETS
    print("Generating new recommendations...")
    new_bets = run_global_soccer_module()
    
    if new_bets.empty:
        send_discord("**No +EV Found Today** — Market is efficient.")
        print("No value found.")
        return
    
    # 3. LOAD EXISTING BETS (deduplication)
    existing_file = "latest_bets.csv"
    if os.path.exists(existing_file):
        existing = pd.read_csv(existing_file)
        existing_games = set(existing['Game_ID']) if 'Game_ID' in existing.columns else set()
    else:
        existing = pd.DataFrame()
        existing_games = set()
    
    # 4. REMOVE DUPLICATES USING Game_ID (FIX #2)
    new_bets = new_bets[~new_bets['Game_ID'].isin(existing_games)]
    
    if new_bets.empty:
        print("No new unique bets.")
        return
    
    # 5. FINAL STAKE CALCULATION (TRUE QUARTER KELLY — FIX #1)
    new_bets['Stake'] = new_bets.apply(
        lambda row: safe_stake(row['Edge'], row['Odds'], vol_factor=1.0) * KELLY_MULTIPLIER, axis=1
    )
    
    # 6. RISK CHECK — NEVER RISK >10% IN ONE CYCLE
    total_risk = (new_bets['Stake'] * BANKROLL).sum()
    if total_risk > BANKROLL * 0.10:
        scale_factor = (BANKROLL * 0.10) / total_risk
        new_bets['Stake'] *= scale_factor
        send_discord(f"**RISK CAP HIT** — Stakes scaled by {scale_factor:.1%}")
        print(f"Risk cap hit. Scaled stakes by {scale_factor:.1%}")
    
    # 7. SAVE & NOTIFY
    final_bets = pd.concat([existing, new_bets], ignore_index=True) if not existing.empty else new_bets
    final_bets.to_csv(existing_file, index=False)
    
    # Update history with pending bets
    history_file = "betting_history.csv"
    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
    else:
        history = pd.DataFrame()
    
    pending_to_add = new_bets[['Date', 'Game_ID', 'Sport', 'League', 'Match', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake']].copy()
    pending_to_add['Result'] = 'Pending'
    pending_to_add['Profit'] = 0.0
    history = pd.concat([history, pending_to_add], ignore_index=True)
    history.drop_duplicates(subset=['Game_ID', 'Bet'], keep='last', inplace=True)
    history.to_csv(history_file, index=False)
    
    # 8. DISCORD ALERT
    top_bet = new_bets.nlargest(1, 'Edge').iloc[0]
    stake_cash = top_bet['Stake'] * BANKROLL
    payout = stake_cash * top_bet['Odds']
    
    alert = f"""
**NEW +EV ALERT** — Co-Pilot v67
**Match**: {top_bet['Match']}
**Bet**: {top_bet['Bet']} @ **{top_bet['Odds']:.2f}**
**Edge**: {top_bet['Edge']:.1%} | **Stake**: ${stake_cash:,.0f}
**Potential Profit**: ${payout - stake_cash:,.0f}
**Total New Bets**: {len(new_bets)}
**Bankroll**: ${BANKROLL:,.0f} | Exposure: ${(new_bets['Stake'] * BANKROLL).sum():,.0f}
"""
    send_discord(alert)
    print(f"Cycle complete. {len(new_bets)} new bets deployed.")

# ==============================================================================
# INFINITE LOOP (RUNS EVERY 30 MINUTES)
# ==============================================================================
if __name__ == "__main__":
    send_discord("**Betting Co-Pilot Pro v67.0 ONLINE** — Engine started.")
    print("Betting Co-Pilot Pro v67.0 — ENGINE STARTED")
    
    while True:
        try:
            run_cycle()
        except Exception as e:
            send_discord(f"**CRITICAL ERROR**: {str(e)}")
            print(f"Error: {e}")
        
        print("Sleeping 30 minutes...")
        time.sleep(1800)  # 30 minutes
