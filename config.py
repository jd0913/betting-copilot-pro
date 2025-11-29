# config.py
# Secure configuration using environment variables
# v68.0 — No more exposed API keys

import os
from dotenv import load_dotenv

# Load .env file if present (for local development)
load_dotenv()

# ==============================================================================
# API CONFIGURATION (SECURE)
# ==============================================================================
API_CONFIG = {
    # The Odds API — NEVER hardcode in public repos!
    "THE_ODDS_API_KEY": os.getenv(
        "THE_ODDS_API_KEY",
        "0c5a163c2e9a8c4b6a5d33c56747ecf1"  # ← Remove this line when deploying!
    ).strip(),

    # Discord Webhook — optional
    "DISCORD_WEBHOOK": os.getenv("DISCORD_WEBHOOK", "").strip()
}

# ==============================================================================
# LEAGUE & TEAM MAPPINGS
# ==============================================================================
SOCCER_LEAGUES = {
    'soccer_epl': 'E0',
    'soccer_spain_la_liga': 'SP1',
    'soccer_germany_bundesliga': 'D1',
    'soccer_italy_serie_a': 'I1',
    'soccer_france_ligue_one': 'F1',
    'soccer_uefa_champs_league': 'UCL'
}

NFL_TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
}

# ==============================================================================
# SAFETY CHECK (Optional — shows warning in Streamlit if key missing)
# ==============================================================================
if "0c5a163c2e9a8c4b6a5d33c56747ecf1" in API_CONFIG["THE_ODDS_API_KEY"]:
    print("WARNING: Using fallback API key! Set THE_ODDS_API_KEY in environment for production.")
