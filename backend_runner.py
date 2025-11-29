# backend_runner.py
# The Execution Script

import pandas as pd
import betting_engine
import config
from datetime import datetime
import os
import requests

def send_discord_alert(df):
    WEBHOOK_URL = config.API_CONFIG["DISCORD_WEBHOOK"]
    if "PASTE_YOUR" in WEBHOOK_URL: return
    if df.empty: msg = "🤖 **Betting Co-Pilot:** No value bets found today."
    else:
        top_bets = df[df['Edge'] > 0.05].sort_values('Edge', ascending=False).head(5)
        msg = f"🚀 **Betting Co-Pilot:** {len(df)} Bets Found!\n"
        for i, row in top_bets.iterrows():
            sport_icon = "⚽" if row['Sport'] == "Soccer" else "🏈" if row['Sport'] == "NFL" else "🏀"
            msg += f"{sport_icon} **{row['Match']}**\n   👉 {row['Bet']} @ {row['Odds']:.2f}\n   📈 Edge: {row['Edge']:.2%} | 💰 Stake: {row['Stake']:.2%}\n\n"
    try: requests.post(WEBHOOK_URL, json={"content": msg})
    except: pass

def run_backend_analysis():
    print("--- Starting Daily Global Backend Analysis (US Pro) ---")
    
    # 1. Settle Bets
    settle_bets()
    
    # 2. Train Brains
    soccer_brain, soccer_hist = train_soccer_brain()
    
    # 3. Run Modules
    soccer_bets = run_soccer_module(soccer_brain, soccer_hist)
    nfl_bets = run_nfl_module()
    nba_bets = run_nba_module()
    mlb_bets = run_mlb_module()
    
    all_bets = pd.concat([soccer_bets, nfl_bets, nba_bets, mlb_bets], ignore_index=True)
    
    # 4. Save & Archive
    if not all_bets.empty:
        all_bets['Date_Generated'] = datetime.now().strftime('%Y-%m-%d')
        
        # *** FIX: Initialize Status Columns to prevent NaN ***
        all_bets['Result'] = 'Pending'
        all_bets['Profit'] = 0.0
        all_bets['Score'] = ''
        
        all_bets.to_csv('latest_bets.csv', index=False)
        
        history_file = 'betting_history.csv'
        if os.path.exists(history_file):
            # Ensure we don't duplicate bets if run multiple times a day
            existing_hist = pd.read_csv(history_file)
            updated_history = pd.concat([existing_hist, all_bets]).drop_duplicates(subset=['Date_Generated', 'Match', 'Bet'])
            updated_history.to_csv(history_file, index=False)
        else:
            all_bets.to_csv(history_file, index=False)
            
        print(f"\nSuccessfully saved {len(all_bets)} recommendations.")
        
        # Discord Alert
        WEBHOOK_URL = API_CONFIG["DISCORD_WEBHOOK"]
        if "PASTE_YOUR" not in WEBHOOK_URL:
            try:
                msg = f"🚀 **Betting Co-Pilot:** {len(all_bets)} Bets Found!\n"
                arbs = all_bets[all_bets['Bet Type'] == 'ARBITRAGE']
                if not arbs.empty: msg += f"🚨 **{len(arbs)} ARBITRAGE OPPORTUNITIES!**\n"
                requests.post(WEBHOOK_URL, json={"content": msg})
            except: pass
    else:
        print("\nNo value bets found.")
        # Save empty file with correct headers
        cols = ['Date', 'Date_Generated', 'Sport', 'League', 'Match', 'Bet Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake', 'Info', 'Result', 'Profit', 'Score']
        pd.DataFrame(columns=cols).to_csv('latest_bets.csv', index=False)

if __name__ == "__main__":
    run_backend_analysis()
