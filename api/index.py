from fastapi import FastAPI, HTTPException
import os
import numpy as np
import pandas as pd
from typing import Dict, List
import requests
from datetime import datetime, timezone

# Import the local model
from model.data_fetcher import fetch_klines_sync, klines_to_arrays
from model.forecaster import BTCForecaster

app = FastAPI()

# In serverless, we initialize the forecaster per request or globally.
# Note: Calibration state won't persist across different lambdas easily 
# without a DB, but it will work for the forecast.
forecaster = BTCForecaster()

@app.get("/api/price")
async def get_price():
    base = "https://api.binance.com"
    try:
        p_res = requests.get(f"{base}/api/v3/ticker/price?symbol=BTCUSDT", timeout=10).json()
        s_res = requests.get(f"{base}/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=10).json()
        
        return {
            "price": float(p_res["price"]),
            "change_24h": float(s_res["priceChangePercent"]),
            "high_24h": float(s_res["highPrice"]),
            "low_24h": float(s_res["lowPrice"]),
            "volume_24h": float(s_res["volume"])
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/forecast")
async def get_forecast():
    try:
        klines = fetch_klines_sync(limit=900)
        opens, highs, lows, closes = klines_to_arrays(klines)
        
        res = forecaster.predict(closes, highs, lows)
        
        return {
            "lower": res.lower,
            "upper": res.upper,
            "mu": res.mu_est,
            "sigma": res.sigma,
            "df": res.df,
            "regime": res.regime,
            "calib": res.calib,
            "last_close": float(closes[-1])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtest")
async def get_backtest():
    try:
        klines = fetch_klines_sync(limit=900)
        opens, highs, lows, closes = klines_to_arrays(klines)
        
        metrics = forecaster.backtest(closes, highs, lows, n_test=120, warmup=100)
        
        return {
            "coverage": metrics.coverage,
            "avg_width": metrics.avg_width,
            "winkler": metrics.winkler,
            "samples": metrics.n_samples,
            "details": metrics.details[-50:] 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history():
    try:
        klines = fetch_klines_sync(limit=50)
        return klines
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
