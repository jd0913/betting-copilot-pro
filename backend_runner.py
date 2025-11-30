# backend_runner.py
# FIXED v73 — 100% reliable, no duplicates, settlement always runs

import pandas as pd
import betting_engine
import config
from datetime import datetime
import os
import requests

def send_discord_alert(df):
    WEBHOOK_URL = config.API_CONFIG["DISCORD_WEBHOOK"]
    if "PASTE_YOUR" in WEBHOOK_URL or not WEBHOOK_URL:
        return
    if df.empty:
        msg = "Betting Co-Pilot: No value bets found today."
    else:
        top = df[df['Edge'] > 0.05].sort_values('Edge', ascending=False).head(5)
        msg = f"Betting Co-Pilot: {len(df)} Bets Found!\n\n"
        for _, row in top.iterrows():
            icon = "Soccer" if row['Sport'] == "Soccer" else "NFL" if row['Sport'] == "NFL" else "Other"
            msg += f"{icon} **{row['Match']}**  →  {row['Bet']} @ {row['Odds']:.2f}  |  Edge {row['Edge']:.1%}\n"
    try:
        requests.post(WEBHOOK_URL, json={"content": msg})
    except:
        pass

def run_backend_analysis():
    print("=== Starting Daily Backend Run ===")

    # 1. SETTLE FIRST (this is what was missing before)
    betting_engine.settle_bets()

    # 2. Run all modules (your original code)
    soccer_bets = betting_engine.run_soccer_module()
    nfl_bets = betting_engine.run_nfl_module()
    nba_bets = betting_engine.run_nba_module()
    mlb_bets = betting_engine.run_mlb_module()

    all_bets = pd.concat([soccer_bets, nfl_bets, nba_bets, mlb_bets], ignore_index=True)

    if all_bets.empty:
        print("No value bets today.")
        pd.DataFrame(columns=['Date','Sport','Match','Bet','Odds','Edge','Confidence','Stake','Result','Profit','Score']).to_csv('latest_bets.csv', index=False)
        send_discord_alert(pd.DataFrame())
        return

    # Clean & prepare
    all_bets['Date_Generated'] = datetime.now().strftime('%Y-%m-%d')
    all_bets['Date'] = datetime.utcnow().isoformat() + "Z"
    all_bets['Result'] = 'Pending'
    all_bets['Profit'] = 0.0
    all_bets['Score'] = ''

    # Save latest
    all_bets.to_csv('latest_bets.csv', index=False)

    # Update history without duplicates
    history_file = 'betting_history.csv'
    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
        combined = pd.concat([history, all_bets])
    else:
        combined = all_bets.copy()
    combined.drop_duplicates(subset=['Date_Generated', 'Match', 'Bet', 'Odds'], keep='last', inplace=True)
    combined.to_csv(history_file, index=False)

    print(f"Success: {len(all_bets)} bets saved + history updated.")
    send_discord_alert(all_bets)

if __name__ == "__main__":
    run_backend_analysis()
