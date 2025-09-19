from fastapi import FastAPI

from .routes import router

app = FastAPI(title="nba-predictor")
app.include_router(router)
