# config.py
# Betting Co-Pilot Pro Configuration - v70.0 (Score Settlement Edition)
# CRITICAL FIX: Added SCORES_URL and settlement parameters

import streamlit as st
import os

def get_secret(key: str, default: str = "") -> str:
    """
    Securely retrieve secrets from environment/Streamlit secrets
    """
    # 1. Environment variables (highest priority)
    env_val = os.getenv(key)
    if env_val:
        return env_val.strip()
    
    # 2. Streamlit secrets (when deployed on Streamlit Cloud)
    if "streamlit" in sys.modules:
        try:
            st_val = st.secrets.get(key)
            if st_val:
                return str(st_val).strip()
        except Exception as e:
            pass
    
    # 3. Default value (with warning)
    if default and "PASTE_YOUR" not in default:
        return default
    
    return "PASTE_YOUR_" + key.upper() + "_HERE"

# ==============================================================================
# 🔐 API CONFIGURATION
# ==============================================================================
API_CONFIG = {
    "THE_ODDS_API_KEY": get_secret(
        "odds_api_key", 
        default="PASTE_YOUR_ODDS_API_KEY_HERE"
    ),
    "DISCORD_WEBHOOK": get_secret(
        "discord_webhook", 
        default="PASTE_YOUR_DISCORD_WEBHOOK_HERE"
    )
}

# ==============================================================================
# 🌍 GITHUB REPOSITORY CONFIGURATION
# ==============================================================================
GITHUB_CONFIG = {
    "USERNAME": get_secret(
        "github_username", 
        default="jd0913"
    ),
    "REPO": get_secret(
        "github_repo", 
        default="betting-copilot-pro"
    ),
    "BRANCH": "main"
}

# ==============================================================================
# 🔗 DATA SOURCE URLs (Dynamically generated from GitHub config)
# ==============================================================================
BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_CONFIG['USERNAME']}/{GITHUB_CONFIG['REPO']}/{GITHUB_CONFIG['BRANCH']}"
URLS = {
    "LATEST_BETS": f"{BASE_URL}/latest_bets.csv",
    "BETTING_HISTORY": f"{BASE_URL}/betting_history.csv",
    "MATCH_SCORES": f"{BASE_URL}/match_scores.csv",  # NEW: For settlement logic
    "SCORES_ARCHIVE": f"{BASE_URL}/scores_archive/"
}

# ==============================================================================
# 🏆 SPORTS CONFIGURATION
# ==============================================================================
SPORTS = {
    "soccer_epl": {
        "name": "English Premier League",
        "code": "E0",
        "teams": [
            "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton and Hove Albion",
            "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
            "Leeds United", "Leicester City", "Liverpool", "Manchester City", "Manchester United",
            "Newcastle United", "Nottingham Forest", "Sheffield United", "Tottenham Hotspur", "West Ham United",
            "Wolverhampton Wanderers"
        ]
    },
    "soccer_spain_la_liga": {
        "name": "Spanish La Liga",
        "code": "SP1",
        "teams": [
            "Alaves", "Athletic Bilbao", "Atletico Madrid", "Barcelona", "Betis",
            "Celta Vigo", "Espanyol", "Getafe", "Girona", "Granada",
            "Las Palmas", "Mallorca", "Osasuna", "Rayo Vallecano", "Real Madrid",
            "Real Sociedad", "Sevilla", "Valencia", "Villarreal"
        ]
    },
    "soccer_germany_bundesliga": {
        "name": "German Bundesliga",
        "code": "D1",
        "teams": [
            "Augsburg", "Bayer Leverkusen", "Bayern Munich", "Borussia Dortmund", "Borussia Monchengladbach",
            "Eintracht Frankfurt", "FC Koln", "Freiburg", "Hamburger SV", "Hoffenheim",
            "Mainz 05", "RB Leipzig", "Schalke 04", "Stuttgart", "Union Berlin",
            "VfL Bochum", "VfL Wolfsburg"
        ]
    },
    "soccer_italy_serie_a": {
        "name": "Italian Serie A",
        "code": "I1",
        "teams": [
            "AC Milan", "Atalanta", "Bologna", "Cagliari", "Empoli",
            "Fiorentina", "Genoa", "Hellas Verona", "Inter Milan", "Juventus",
            "Lazio", "Lecce", "Napoli", "Roma", "Salernitana",
            "Sampdoria", "Sassuolo", "Torino", "Udinese"
        ]
    },
    "soccer_france_ligue_one": {
        "name": "French Ligue 1",
        "code": "F1",
        "teams": [
            "Ajaccio", "Angers", "Auxerre", "Brest", "Clermont Foot",
            "Lens", "Lille", "Lorient", "Lyon", "Marseille",
            "Metz", "Monaco", "Montpellier", "Nantes", "Nice",
            "Paris Saint Germain", "Reims", "Rennes", "Strasbourg", "Toulouse"
        ]
    },
    "soccer_uefa_champs_league": {
        "name": "UEFA Champions League",
        "code": "CL",
        "teams": []  # Dynamic based on season
    }
}

NFL_TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
}

# ==============================================================================
# ⚙️ BETTING PARAMETERS (Professional Settings)
# ==============================================================================
BETTING_CONFIG = {
    "kelly_fraction": 0.25,    # Quarter-Kelly standard for risk management
    "min_edge": 0.02,          # Minimum edge (2%) to place bets
    "max_bet_size": 0.05,      # Max 5% of bankroll per bet
    "max_daily_bets": 10,      # Daily bet limit
    "min_odds": 1.01,          # Minimum odds to consider
    "settlement_grace_period": 4,  # Hours after match end before auto-settlement
    "score_fallback": "N/A"    # Default score when not available
}

# ==============================================================================
# 📁 SYSTEM PATHS
# ==============================================================================
SYSTEM_PATHS = {
    "model_cache": "model_cache",
    "data_archive": "data_archive",
    "logs_dir": "logs",
    "tmp_dir": "tmp"
}

# Create critical directories
for path in SYSTEM_PATHS.values():
    os.makedirs(path, exist_ok=True)

# ==============================================================================
# ✅ CONFIGURATION VALIDATION
# ==============================================================================
def validate_config() -> bool:
    """
    Validate critical configuration before execution
    Returns True if configuration is valid, False otherwise
    """
    valid = True
    
    # Critical API check
    odds_key = API_CONFIG["THE_ODDS_API_KEY"]
    if "PASTE_YOUR" in odds_key:
        st.error("🚨 THE_ODDS_API_KEY NOT CONFIGURED! Betting disabled.")
        valid = False
    
    # GitHub configuration check
    if "PASTE_YOUR" in GITHUB_CONFIG["USERNAME"] or "PASTE_YOUR" in GITHUB_CONFIG["REPO"]:
        st.error("🚨 GITHUB CONFIGURATION INVALID! Data loading disabled.")
        valid = False
    
    # Directory permissions check
    for name, path in SYSTEM_PATHS.items():
        if not os.access(path, os.W_OK):
            st.warning(f"⚠️ Directory not writable: {path} ({name})")
    
    return valid
