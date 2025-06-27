from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from schemas import TrustRequest, TrustResponse
from services.trust_engine import evaluate_trust

app = FastAPI(title="TrustSphere API", version="1.0.0")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Simple user store
def authenticate_user(username: str, password: str):
    return username == "user" and password == "1234"

def get_current_user(token: str = Depends(oauth2_scheme)):
    # For demo, token is just 'user' if authenticated
    if token != "user":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if authenticate_user(form_data.username, form_data.password):
        return {"access_token": "user", "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/")
def root():
    return {"status": "running", "message": "TrustSphere backend is live"}

@app.post("/analyze", response_model=TrustResponse)
async def analyze_text(request: TrustRequest, user: str = Depends(get_current_user)):
    return evaluate_trust(request.text, request.facial_emotion, request.audio_emotion, request.audio_weight) 