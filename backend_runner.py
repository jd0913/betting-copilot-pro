# backend_runner.py
# The Execution Script (Enterprise Edition)
# This script triggers the logic inside betting_engine.py

import pandas as pd
import betting_engine
import config
from datetime import datetime
import os
import requests
import warnings

warnings.filterwarnings('ignore')

def send_discord_alert(df):
    """Sends a summary of value bets to a Discord channel."""
    WEBHOOK_URL = config.API_CONFIG["DISCORD_WEBHOOK"]
    if "PASTE_YOUR" in WEBHOOK_URL: return

    if df.empty:
        msg = "🤖 **Betting Co-Pilot:** No value bets found today."
    else:
        # Filter for top bets to avoid spamming
        top_bets = df[df['Edge'] > 0.05].sort_values('Edge', ascending=False).head(5)
        msg = f"🚀 **Betting Co-Pilot:** {len(df)} Bets Found!\n"
        
        for i, row in top_bets.iterrows():
            sport_icon = "⚽"
            if row['Sport'] == "NFL": sport_icon = "🏈"
            elif row['Sport'] == "NBA": sport_icon = "🏀"
            elif row['Sport'] == "MLB": sport_icon = "⚾"
            
            msg += f"{sport_icon} **{row['Match']}**\n"
            msg += f"   👉 {row['Bet']} @ {row['Odds']:.2f}\n"
            msg += f"   📈 Edge: {row['Edge']:.2%} | 💰 Stake: {row.get('Stake', 0):.2%}\n\n"
            
    try: requests.post(WEBHOOK_URL, json={"content": msg})
    except: pass

def run_backend_analysis():
    print("--- Starting Daily Global Backend Analysis (Modular) ---")
    
    # 1. Settle Bets (Using the Engine)
    # This grades yesterday's bets as Win/Loss
    betting_engine.settle_bets()
    
    # 2. Train Brains (Using the Engine)
    soccer_brain, soccer_hist = betting_engine.train_soccer_brain()
    
    # 3. Run Modules (Using the Engine)
    soccer_bets = betting_engine.run_soccer_module(soccer_brain, soccer_hist)
    nfl_bets = betting_engine.run_nfl_module()
    nba_bets = betting_engine.run_nba_module()
    mlb_bets = betting_engine.run_mlb_module()
    
    # Combine all sports
    all_bets = pd.concat([soccer_bets, nfl_bets, nba_bets, mlb_bets], ignore_index=True)
    
    # 4. Save & Archive
    if not all_bets.empty:
        all_bets['Date_Generated'] = datetime.now().strftime('%Y-%m-%d')
        
        # Initialize Status Columns to prevent NaN in History
        all_bets['Result'] = 'Pending'
        all_bets['Profit'] = 0.0
        all_bets['Score'] = ''
        
        # Save Latest (For Dashboard)
        all_bets.to_csv('latest_bets.csv', index=False)
        
        # Archive History (For Performance Lab)
        history_file = 'betting_history.csv'
        if os.path.exists(history_file):
            existing_hist = pd.read_csv(history_file)
            # Combine and drop duplicates
            updated_history = pd.concat([existing_hist, all_bets]).drop_duplicates(subset=['Date_Generated', 'Match', 'Bet'])
            updated_history.to_csv(history_file, index=False)
        else:
            all_bets.to_csv(history_file, index=False)
            
        print(f"\nSuccessfully saved {len(all_bets)} recommendations.")
        send_discord_alert(all_bets)
    else:
        print("\nNo value bets found.")
        # Save empty file with correct headers so App doesn't crash
        cols = ['Date', 'Date_Generated', 'Sport', 'League', 'Match', 'Bet Type', 'Bet', 'Odds', 'Edge', 'Confidence', 'Stake', 'Info', 'Result', 'Profit', 'Score']
        pd.DataFrame(columns=cols).to_csv('latest_bets.csv', index=False)
        send_discord_alert(pd.DataFrame())

if __name__ == "__main__":
    run_backend_analysis()
