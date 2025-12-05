# config.py
# Betting Co-Pilot Pro Configuration - Streamlit Cloud Ready
# Secure version that uses Streamlit Secrets and environment variables

import streamlit as st
import os

def get_api_key(key_name: str, default: str = "") -> str:
    """
    Securely get API keys from multiple sources in priority order:
    1. Streamlit Secrets (when running on Streamlit Cloud)
    2. Environment Variables (for local/backend runs)
    3. Default value (if nothing else is available)
    """
    # Try Streamlit Secrets first (for web app)
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return str(st.secrets[key_name])
    except Exception:
        pass
    
    # Try environment variables (for backend scripts)
    env_val = os.getenv(key_name.upper())
    if env_val:
        return env_val
    
    # Fallback to default
    return default

# ==============================================================================
# 🔐 API CONFIGURATION (Secure loading)
# ==============================================================================
API_CONFIG = {
    "THE_ODDS_API_KEY": get_api_key("odds_api_key", "dummy_key_for_github"),
    "DISCORD_WEBHOOK": get_api_key("discord_webhook", "dummy_webhook_for_github")
}

# ==============================================================================
# 🌍 LEAGUE SETTINGS (Fixed codes)
# ==============================================================================
SOCCER_LEAGUES = {
    'soccer_epl': 'E0',           # English Premier League
    'soccer_spain_la_liga': 'SP1', # Spanish La Liga
    'soccer_germany_bundesliga': 'D1', # German Bundesliga
    'soccer_italy_serie_a': 'I1',  # Italian Serie A
    'soccer_france_ligue_one': 'F1', # French Ligue 1
    'soccer_uefa_champs_league': 'CL'  # UEFA Champions League (FIXED from UCL to CL)
}

# ==============================================================================
# 🏈 NFL TEAM MAPPING (Fixed abbreviations)
# ==============================================================================
NFL_TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",  # FIXED: LA → LAR
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
}

# ==============================================================================
# 💰 BETTING PARAMETERS (Professional defaults)
# ==============================================================================
BETTING_CONFIG = {
    "kelly_fraction": 0.25,    # Quarter-Kelly standard
    "min_edge": 0.02,          # Minimum edge (2%) to place bets
    "max_bet_size": 0.05,      # Max 5% of bankroll per bet
    "currency": "USD"
}

# ==============================================================================
# ✅ CONFIGURATION VALIDATION
# ==============================================================================
def is_config_valid() -> bool:
    """Check if essential configuration is available"""
    odds_key = API_CONFIG["THE_ODDS_API_KEY"]
    return odds_key and "dummy" not in odds_key and "PASTE_YOUR" not in odds_key
