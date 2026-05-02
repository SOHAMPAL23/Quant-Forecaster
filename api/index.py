from fastapi import FastAPI, HTTPException
import os
import numpy as np
import sys
from typing import Dict, List
import asyncio
from datetime import datetime, timezone

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from model.data_fetcher import fetch_klines_async, klines_to_arrays, fetch_24h_stats
from model.forecaster import BTCForecaster

app = FastAPI()
forecaster = BTCForecaster()

@app.get("/favicon.ico")
async def favicon():
    from fastapi import Response
    return Response(status_code=204)

@app.get("/api/price")
async def get_price():
    try:
        stats = await fetch_24h_stats()
        return {
            "price": stats["last_price"],
            "price_change_pct": stats["price_change_pct"]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/forecast")
async def get_forecast():
    """Aggregated endpoint for the new UI."""
    try:
        # Fetch everything in parallel
        klines_task = fetch_klines_async(limit=900)
        stats_task = fetch_24h_stats()
        
        klines, stats = await asyncio.gather(klines_task, stats_task)
        opens, highs, lows, closes = klines_to_arrays(klines)
        
        # Core forecast
        res = forecaster.predict(closes, highs, lows)
        
        # Backtest metrics
        bt = forecaster.backtest(closes, highs, lows, n_test=120, warmup=100)
        
        return {
            "live_price": stats["last_price"],
            "forecast": {
                "lower": res.lower,
                "upper": res.upper,
                "mu": res.mu_est,
                "sigma": res.sigma,
                "df": res.df,
                "regime": res.regime,
                "calib": res.calib
            },
            "backtest": {
                "coverage": bt.coverage,
                "avg_width": bt.avg_width,
                "winkler": bt.winkler,
                "n_samples": bt.n_samples
            },
            "stats_24h": {
                "price_change_pct": stats["price_change_pct"],
                "high_24h": stats["high_24h"],
                "low_24h": stats["low_24h"],
                "volume_24h": stats["volume_24h"]
            },
            "candles": klines[-50:], # Last 50 candles for the chart
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")

@app.get("/api/backtest_details")
async def get_backtest_details(limit: int = 100):
    """Detailed backtest history for the lower chart."""
    try:
        klines = await fetch_klines_async(limit=900)
        opens, highs, lows, closes = klines_to_arrays(klines)
        metrics = forecaster.backtest(closes, highs, lows, n_test=limit, warmup=100)
        return {"details": metrics.details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history():
    try:
        klines = await fetch_klines_async(limit=50)
        return klines
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
