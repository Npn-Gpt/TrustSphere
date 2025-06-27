import os
import sys
import warnings
import logging
import time
import webbrowser
from threading import Thread
import socket

# Suppress all warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

# Configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'  # Disable file watching
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'  # Run in headless mode

# Initialize PyTorch before Streamlit
import torch
torch.set_warn_always(False)

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run_fastapi():
    import uvicorn
    # Try to start FastAPI
    try:
        uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=False)
    except Exception as e:
        print(f"FastAPI Error: {str(e)}")
        sys.exit(1)

def run_streamlit():
    import streamlit.web.cli as stcli
    sys.argv = [
        "streamlit", "run", "main.py",
        "--server.fileWatcherType=none",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.port=8505"
    ]
    sys.exit(stcli.main())

if __name__ == "__main__":
    try:
        # Check if ports are in use
        if is_port_in_use(8001):
            print("Port 8001 is in use. Please close any running instances of the application.")
            sys.exit(1)
        if is_port_in_use(8505):
            print("Port 8505 is in use. Please close any running instances of the application.")
            sys.exit(1)

        # Start FastAPI in a separate thread
        fastapi_thread = Thread(target=run_fastapi)
        fastapi_thread.daemon = True
        fastapi_thread.start()
        
        # Wait a moment for FastAPI to start
        time.sleep(2)
        
        # Open browser
        webbrowser.open('http://localhost:8505')
        
        # Run Streamlit in main thread
        run_streamlit()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 