import streamlit as st

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="TrustSphere",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import the rest of the app
from app import main
main()
