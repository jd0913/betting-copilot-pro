# backend_runner.py
# Daily Execution Engine — Runs models, saves bets, settles results, sends alerts
# v68.0 — Fully fixed, safe, duplicate-proof, error-resilient

import pandas as pd
import betting_engine
import config
from datetime import datetime
import os
import requests
import logging

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# DISCORD ALERTS (Safe — won't crash if webhook missing)
# ==============================================================================
def send_discord_alert(df):
    webhook_url = config.API_CONFIG.get("DISCORD_WEBHOOK", "").strip()
    if not webhook_url or "PASTE" in webhook_url or "YOUR" in webhook_url:
        logger.info("Discord webhook not configured — skipping alert.")
        return

    if df.empty:
        msg = "No value bets found today."
    else:
        top_bets = df[df['Edge'] > 0.05].sort_values('Edge', ascending=False).head(6)
        lines = [f"Found {len(df)} bets today!"]
        for _, row in top_bets.iterrows():
            sport_icon = "⚽" if row['Sport'] == "Soccer" else "NFL" if row['Sport'] == "NFL" else "Other"
            lines.append(f"{sport_icon} **{row['Match']}**")
            lines.append(f"   → {row['Bet']} @ {row['Odds']:.2f} | Edge: {row['Edge']:.1%}")
        msg = "\n".join(lines)

    try:
        requests.post(webhook_url, json={"content": msg}, timeout=10)
        logger.info("Discord alert sent.")
    except Exception as e:
        logger.error(f"Failed to send Discord alert: {e}")

# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def run_backend_analysis():
    logger.info("=== Starting Daily Betting Co-Pilot Run ===")

    try:
        # 1. Settle old bets (Soccer + NFL now work!)
        betting_engine.settle_bets()
        logger.info("Settlement complete.")
    except Exception as e:
        logger.error(f"Settlement failed: {e}")

    try:
        # 2. Run all sport modules
        logger.info("Running prediction modules...")
        soccer_bets = betting_engine.run_soccer_module(None, None)  # model not used yet
        nfl_bets = betting_engine.run_nfl_module()
        nba_bets = betting_engine.run_nba_module()
        mlb_bets = betting_engine.run_mlb_module()

        all_bets = pd.concat([soccer_bets, nfl_bets, nba_bets, mlb_bets], ignore_index=True)

        if all_bets.empty:
            logger.info("No bets generated today.")
            # Still create empty file so frontend doesn't crash
            empty_df = pd.DataFrame(columns=[
                'Date', 'Date_Generated', 'Sport', 'League', 'Match', 'Bet Type',
                'Bet', 'Odds', 'Edge', 'Confidence', 'Stake', 'Info',
                'Result', 'Profit', 'Score'
            ])
            empty_df.to_csv('latest_bets.csv', index=False)
            send_discord_alert(pd.DataFrame())
            return

        # 3. Clean & standardize
        all_bets['Date_Generated'] = datetime.now().strftime('%Y-%m-%d')
        if 'Date' not in all_bets.columns or all_bets['Date'].isna().all():
            all_bets['Date'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

        # Initialize result columns
        for col in ['Result', 'Profit', 'Score']:
            if col not in all_bets.columns:
                all_bets[col] = 'Pending' if col == 'Result' else 0.0 if col == 'Profit' else ''

        # 4. Save latest_bets.csv
        all_bets.to_csv('latest_bets.csv', index=False)
        logger.info(f"Saved {len(all_bets)} new bets to latest_bets.csv")

        # 5. Append to history (NO DUPLICATES!)
        history_file = 'betting_history.csv'
        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
            # Strong deduplication: match on date + match + bet + odds
            combined = pd.concat([history_df, all_bets], ignore_index=True)
            combined.drop_duplicates(
                subset=['Date_Generated', 'Match', 'Bet', 'Odds'],
                keep='last',
                inplace=True
            )
        else:
            combined = all_bets.copy()

        combined.to_csv(history_file, index=False)
        logger.info(f"Updated history → {len(combined)} total bets recorded.")

        # 6. Send alert
        send_discord_alert(all_bets)

        logger.info("=== Daily run completed successfully! ===")

    except Exception as e:
        logger.error(f"CRITICAL ERROR in backend run: {e}")
        # Still create empty file so Streamlit doesn't crash
        pd.DataFrame().to_csv('latest_bets.csv', index=False)

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    run_backend_analysis()
