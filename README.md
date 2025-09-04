# 🏀 NBA Predictor

A small end-to-end project for predicting NBA game outcomes.  
This repo is about practicing the full pipeline: fetch → features → train → serve.  
Right now the model is very simple, but the focus is on having something reproducible and deployable.

---

## 📂 Project Layout

```
nba-predictor/
  src/
    data/
      fetch.py         # pulls game logs from Basketball-Reference
    model/
    service/           # (coming soon) FastAPI app for /predict
  data_cache/
    games.csv          # raw game data (generated)
  artifacts/
```

---

## 🚀 Pipeline

### 1. Fetch Data
Scrape Basketball-Reference and save to `data_cache/games.csv`.

```bash
python -m src.data.fetch
```

### 2. Build Features
Rolling 10-game averages per team → matchup deltas.

```bash
python -m src.data.features
```

Outputs: `data_cache/features.csv`

### 3. Train Model
Baseline logistic regression classifier.

```bash
python -m src.model.train
```

Outputs:  
- `artifacts/model.joblib` (serialized model)  
- `artifacts/metrics.json` (accuracy, ROC-AUC, sample counts)

---

## 📊 Current Baseline
- Features: difference in rolling 10-game offensive/defensive averages  
- Model: Logistic Regression (scikit-learn)  
- Typical metrics (varies by season): ~X% accuracy, ~Y ROC-AUC  
  *(fill these in after you train on your machine)*

---

## 🔜 Next Steps
- Add more features (3P%, turnovers, rest days, home/away splits)  
- Swap in RandomForest/XGBoost  
- Build FastAPI service (`/health`, `/predict`)  
- Dockerize and deploy  

---

## ⚙️ Requirements
See [requirements.txt](requirements.txt).  
Key packages: `pandas`, `scikit-learn`, `fastapi`, `uvicorn`, `lxml`, `beautifulsoup4`.

Create venv + install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ✏️ Notes
- Data source: [Basketball-Reference](https://www.basketball-reference.com/) (scraped game logs).  
- The official `nba_api` route exists but is flaky; this project defaults to BR for stability.  
- This is a work in progress—right now it’s a baseline model with a clean pipeline, not a production predictor.