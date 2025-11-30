# backend_runner.py
# The Execution Script
# FIX: Removed non-existent 'train_soccer_brain' function call.
# FIX: Added specific exception handling for Discord connection errors.

import pandas as pd
import betting_engine
import config
from datetime import datetime
import os
import requests
from requests.exceptions import RequestException # Import specific exception

def send_discord_alert(df):
    WEBHOOK_URL = config.API_CONFIG.get("DISCORD_WEBHOOK", "PASTE_YOUR_WEBHOOK_URL_HERE")
    if "PASTE_YOUR" in WEBHOOK_URL: return
    if df.empty: msg = "🤖 **Betting Co-Pilot:** No value bets found today."
    else:
        top_bets = df[df['Edge'] > 0.05].sort_values('Edge', ascending=False).head(5)
        msg = f"🚀 **Betting Co-Pilot:** {len(df)} Bets Found!\n"
        for i, row in top_bets.iterrows():
            sport_icon = "⚽" if row['Sport'] == "Soccer" else "🏈" if row['Sport'] == "NFL" else "🏀"
            msg += f"{sport_icon} **{row['Match']}**\n   👉 {row['Bet']} @ {row['Odds']:.2f}\n   📈 Edge: {row['Edge']:.2%} | 💰 Stake: {row['Stake']:.2%}\n\n"
    try: 
        requests.post(WEBHOOK_URL, json={"content": msg})
    except RequestException as e: # Catch specific Request errors
        print(f"Error sending Discord alert: {e}") 
        pass

def run_backend_analysis():
    print("--- Starting Daily Global Backend Analysis (Modular) ---")
    
    # 1. Settle Bets 
    betting_engine.settle_bets()
    
    # 2. Train Brains
    # DELETED: soccer_brain, soccer_hist = betting_engine.train_soccer_brain() 
    # The training for soccer is now handled internally by run_global_soccer_module()
    
    # 3. Run Modules
    # The 'run_global_soccer_module' function handles both training and execution.
    soccer_bets = betting_engine.run_global_soccer_module() 
    nfl_bets = betting_engine.run_nfl_module()
    nba_bets = betting_engine.run_nba_module()
    mlb_bets = betting_engine.run_mlb_module()
    
    all_bets = pd.concat([soccer_bets, nfl_bets, nba_bets, mlb_bets], ignore_index=True)
    
    # 4. Save & Archive
    if not all_bets.empty:
        all_bets['Date_Generated'] = datetime.now().strftime('%Y-%m-%d')
        
        # Ensure Date column exists
        if 'Date' not in all_bets.columns:
            all_bets['Date'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            all_bets['Date'] = all_bets['Date'].fillna(datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'))
            
        # Initialize Status Columns
        all_bets['Result'] = 'Pending'
        all_bets['Profit'] = 0.0
        all_bets['Score'] = ''
        
        all_bets.to_csv('latest_bets.csv', index=False)
        
        history_file = 'betting_history.csv'
        if os.path.exists(history_file):
            pd.concat([pd.read_csv(history_file), all_bets]).drop_duplicates(subset=['Date_Generated', 'Match', 'Bet']).to_csv(history_file, index=False)
        else:
            all_bets.to_csv(history_file, index=False)
            
        print(f"\nSuccessfully saved {len(all_bets)} recommendations.")
        send_discord_alert(all_bets)
    else:
        print("\nNo value bets found.")
        # Ensure latest_bets.csv is created even if empty for the Streamlit app to load.
        pd.DataFrame(columns=['Date', 'Date_Generated', 'Sport', 'League', 'Match', 'Bet Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake', 'Info', 'Result', 'Profit', 'Score']).to_csv('latest_bets.csv', index=False)
        send_discord_alert(pd.DataFrame())

if __name__ == "__main__":
    run_backend_analysis()
