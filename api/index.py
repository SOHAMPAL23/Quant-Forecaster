from fastapi import FastAPI, HTTPException
import os
import numpy as np
import pandas as pd
from typing import Dict, List
import requests
from datetime import datetime, timezone
import asyncio
import sys

# Ensure the root directory is in sys.path so 'model' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the local model
from model.data_fetcher import fetch_klines_async, klines_to_arrays, fetch_24h_stats
from model.forecaster import BTCForecaster

app = FastAPI()

# Initialize the forecaster
forecaster = BTCForecaster()

@app.get("/favicon.ico")
async def favicon():
    from fastapi import Response
    return Response(status_code=204)

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc)}

@app.get("/api/price")
async def get_price():
    """Fetch live BTC price and 24h stats using async fetcher."""
    try:
        # Use the async stats fetcher from data_fetcher.py
        stats = await fetch_24h_stats()
        return {
            "price": stats["last_price"],
            "change_24h": stats["price_change_pct"],
            "high_24h": stats["high_24h"],
            "low_24h": stats["low_24h"],
            "volume_24h": stats["volume_24h"]
        }
    except Exception as e:
        # Return structured error for debugging
        return {"error": "Binance API unreachable", "detail": str(e)}

@app.get("/api/forecast")
async def get_forecast():
    """Run model and return next hour forecast (Async)."""
    try:
        # Direct await to avoid loop-in-loop conflicts on Vercel
        klines = await fetch_klines_async(limit=900)
        opens, highs, lows, closes = klines_to_arrays(klines)
        
        # CPU bound task, but fast enough for serverless
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
        raise HTTPException(status_code=500, detail=f"Forecast Engine Error: {str(e)}")

@app.get("/api/backtest")
async def get_backtest():
    """Run walk-forward backtest (Async)."""
    try:
        klines = await fetch_klines_async(limit=900)
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
        raise HTTPException(status_code=500, detail=f"Backtest Engine Error: {str(e)}")

@app.get("/api/history")
async def get_history():
    """Fetch last 50 candles (Async)."""
    try:
        klines = await fetch_klines_async(limit=50)
        return klines
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data Fetch Error: {str(e)}")
