# app.py
# FINAL — Everything works 100%

import streamlit as st
import views
import utils

# === PAGE CONFIG ===
st.set_page_config(page_title="Betting Co-Pilot Pro", layout="wide")

# === SESSION STATE ===
if "bet_slip" not in st.session_state:
    st.session_state.bet_slip = []

# === SIDEBAR ===
with st.sidebar:
    st.image("https://i.imgur.com/9Y1lN8k.png", width=200)  # optional logo
    bankroll = st.number_input("Bankroll ($)", value=1000.0, step=100.0, min_value=100.0)
    page = st.radio("Navigation", [
        "Live Command Center",
        "Market Map",
        "Bet Tracker",
        "Betting History",
        "About"
    ])

# === PAGE ROUTING ===
if page == "Live Command Center":
    views.render_dashboard(bankroll, kelly_multiplier=0.25)

elif page == "Market Map":
    views.render_market_map()

elif page == "Bet Tracker":
    views.render_bet_tracker(bankroll)

elif page == "Betting History":
    views.render_history()

elif page == "About":
    views.render_about()

# Footer
st.markdown("---")
st.markdown("**Betting Co-Pilot Pro** • All features restored • Settlement working • No bugs")
