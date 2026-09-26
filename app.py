"""
FastAPI Cloud AI Model Server for Render / Linux Cloud
Hosts the Golden DoubleEnsemble (XGBoost + CatBoost) AI Trading Brain 24/7.
Provides high-performance REST API endpoints for live market inference and opportunity scanning.
"""

import os
import sys
import time
import datetime
import json
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure GPUDoubleEnsemble is registered in __main__ for unpickling
class GPUDoubleEnsemble:
    def __init__(self, random_state=42, max_epistemic_disagreement=0.08):
        self.random_state = random_state
        self.max_epistemic_disagreement = max_epistemic_disagreement
        self.xgb_base = None
        self.xgb_reweighted = None
        self.cat_model = None
        self.calibrator = None
        self.weights = [0.50, 0.50]
        self.feature_names = None
        self.pair_thresholds = {}
        self.daily_top_k = 5

    def predict_with_uncertainty(self, X_input):
        p_xgb = self.xgb_reweighted.predict_proba(X_input)[:, 1]
        p_cat = self.cat_model.predict_proba(X_input)[:, 1]
        u_epistemic = np.abs(p_xgb - p_cat)
        p_blend = self.weights[0] * p_xgb + self.weights[1] * p_cat
        if hasattr(self, 'calibrator') and self.calibrator is not None:
            try:
                p_blend = self.calibrator.predict(p_blend)
            except Exception:
                pass
        is_hallucination = u_epistemic > self.max_epistemic_disagreement
        return p_blend, u_epistemic, is_hallucination

    def predict_proba(self, X_input):
        p_blend, _, _ = self.predict_with_uncertainty(X_input)
        return np.column_stack([1.0 - p_blend, p_blend])

    def predict(self, X_input, threshold=0.65):
        p_blend, _, is_hal = self.predict_with_uncertainty(X_input)
        return np.where((p_blend > threshold) & (~is_hal), 1, 0)

sys.modules['__main__'].GPUDoubleEnsemble = GPUDoubleEnsemble

from trading_bot_engine import UnifiedTradingBotEngine
from telegram_bot import TelegramNotifier

app = FastAPI(
    title="AI Quantitative MT5 Model Server",
    description="24/7 Cloud AI Decision Brain for MetaTrader 5 execution",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[UnifiedTradingBotEngine] = None

# Telegram notifier configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8643518675:AAG_T9rUW2c2SPM8K9jjBu5Iy6gplStWe_E")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5336298229")
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

@app.on_event("startup")
def startup_event():
    global engine
    try:
        print("[INIT] Loading AI Trading Engine and Ensemble Models...")
        engine = UnifiedTradingBotEngine(model_dir=".")
        print(f"[READY] AI Engine Loaded! Universal Threshold: {engine.universal_threshold}")
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load engine: {e}")
        engine = None

class Candle(BaseModel):
    time: Optional[Any] = None
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = 0.0

class PredictRequest(BaseModel):
    pair: str = Field(..., example="EURUSD")
    timeframe: str = Field(default="15M", example="15M")
    candles: List[Candle]
    account_balance: Optional[float] = 100.0

class OpenPositionInfo(BaseModel):
    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    price_open: float
    price_current: float
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    volume: float = 0.01

class BotCycleRequest(BaseModel):
    timeframe: str = Field(default="15M")
    market_data: Dict[str, List[Candle]]
    open_positions: List[OpenPositionInfo] = []
    account_balance: float = 100.0
    account_equity: float = 100.0
    daily_pnl: float = 0.0
    today_trades: int = 0
    max_daily_trades: int = 5
    daily_max_loss: float = -10.0
    max_concurrent_positions: int = 2

class ScanRequest(BaseModel):
    timeframe: str = Field(default="15M", example="15M")
    market_data: Dict[str, List[Candle]]
    account_balance: Optional[float] = 100.0

class NotifyRequest(BaseModel):
    message: str

def parse_candles_to_df(candles: List[Candle]) -> pd.DataFrame:
    rows = []
    for c in candles:
        rows.append({
            'Open': c.open,
            'High': c.high,
            'Low': c.low,
            'Close': c.close,
            'Volume': c.volume if c.volume is not None else 0.0,
            'Time': c.time
        })
    df = pd.DataFrame(rows)
    if 'Time' in df.columns and df['Time'].iloc[0] is not None:
        try:
            df.index = pd.to_datetime(df['Time'], unit='s' if isinstance(df['Time'].iloc[0], (int, float)) and df['Time'].iloc[0] > 1e9 else None)
        except Exception:
            df.index = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq='15min')
    else:
        df.index = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq='15min')
    return df

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ONLINE",
        "service": "AI Quantitative MT5 Model Server",
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "model_loaded": engine is not None,
        "universal_threshold": engine.universal_threshold if engine else None,
        "active_features": len(engine.feature_cols) if engine else 0,
        "supported_pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"],
        "supported_timeframes": ["15M", "1H"]
    }

@app.post("/predict", tags=["Inference"])
def predict_single_pair(req: PredictRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="AI Model Engine is not loaded.")
    if len(req.candles) < 50:
        raise HTTPException(status_code=400, detail=f"Insufficient candles provided ({len(req.candles)} < 50 required for TA computation)")

    df = parse_candles_to_df(req.candles)
    res = engine.predict_opportunity(req.pair.upper(), req.timeframe.upper(), df)

    p_win = float(res.get('confidence', 0.5))
    u_epi = float(res.get('u_epistemic', 0.0))
    conv = p_win - 0.5 * u_epi
    
    if conv >= 0.57 and p_win >= 0.62:
        lot = 0.03
        tier = "ULTRA"
    elif conv >= 0.52:
        lot = 0.02
        tier = "STANDARD"
    else:
        lot = 0.01
        tier = "DEFENSIVE"

    bal_ratio = max(0.5, (req.account_balance or 100.0) / 100.0)
    lot = round(lot * bal_ratio, 2)
    lot = max(0.01, min(lot, 0.05))

    res['recommended_lot'] = lot
    res['conviction_tier'] = tier
    res['effective_conviction'] = round(conv, 4)
    res['win_probability'] = p_win
    res['epistemic_uncertainty'] = u_epi
    res['is_hallucination'] = u_epi > 0.08
    res['confidence_passed'] = res.get('status') == 'CONFIRMED'
    return res

@app.post("/scan", tags=["Inference"])
def scan_all_pairs(req: ScanRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="AI Model Engine is not loaded.")

    opportunities = []
    t_start = time.time()

    for pair, candles in req.market_data.items():
        if len(candles) < 50:
            continue
        try:
            df = parse_candles_to_df(candles)
            pair_tf = "1H" if pair.upper() == "XAUUSD" else req.timeframe.upper()
            res = engine.predict_opportunity(pair.upper(), pair_tf, df)
            
            p_win = float(res.get('confidence', 0.5))
            u_epi = float(res.get('u_epistemic', 0.0))
            conv = p_win - 0.5 * u_epi

            if conv >= 0.57 and p_win >= 0.62:
                lot = 0.03
                tier = "ULTRA"
            elif conv >= 0.52:
                lot = 0.02
                tier = "STANDARD"
            else:
                lot = 0.01
                tier = "DEFENSIVE"

            bal_ratio = max(0.5, (req.account_balance or 100.0) / 100.0)
            lot = round(lot * bal_ratio, 2)
            lot = max(0.01, min(lot, 0.05))

            res['recommended_lot'] = lot
            res['conviction_tier'] = tier
            res['effective_conviction'] = round(conv, 4)
            res['win_probability'] = p_win
            res['epistemic_uncertainty'] = u_epi
            res['is_hallucination'] = u_epi > 0.08
            res['confidence_passed'] = res.get('status') == 'CONFIRMED'
            opportunities.append(res)
        except Exception as e:
            print(f"[ERROR] Failed scanning pair {pair}: {e}")

    # Sort opportunities: confirmed first, then by highest confidence
    opportunities.sort(key=lambda x: (x.get('confidence_passed', False), x.get('confidence', 0.0)), reverse=True)

    actionable = [o for o in opportunities if o.get('confidence_passed', False) and not o.get('is_hallucination', False) and o.get('action') in ('BUY', 'SELL')]

    return {
        "status": "SUCCESS",
        "scanned_pairs": len(req.market_data),
        "total_opportunities": len(opportunities),
        "actionable_signals": len(actionable),
        "total_scan_latency_ms": round((time.time() - t_start) * 1000, 2),
        "top_opportunities": opportunities[:5],
        "actionable": actionable
    }

@app.post("/bot/cycle", tags=["Bot Engine"])
def run_cloud_bot_cycle(req: BotCycleRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="AI Model Engine is not loaded.")

    t_start = time.time()
    opportunities = []

    for pair, candles in req.market_data.items():
        if len(candles) < 50:
            continue
        try:
            df = parse_candles_to_df(candles)
            pair_tf = "1H" if pair.upper() == "XAUUSD" else req.timeframe.upper()
            res = engine.predict_opportunity(pair.upper(), pair_tf, df)

            p_win = float(res.get('confidence', 0.5))
            u_epi = float(res.get('u_epistemic', 0.0))
            conv = p_win - 0.5 * u_epi

            res['effective_conviction'] = round(conv, 4)
            res['win_probability'] = p_win
            res['epistemic_uncertainty'] = u_epi
            res['is_hallucination'] = u_epi > 0.08
            res['confidence_passed'] = res.get('status') == 'CONFIRMED'
            opportunities.append(res)
        except Exception as e:
            print(f"[ERROR] Pair {pair}: {e}")

    opportunities.sort(key=lambda x: (x.get('confidence_passed', False), x.get('confidence', 0.0)), reverse=True)
    opps_dict = {o['pair']: o for o in opportunities}
    actionable = [o for o in opportunities if o.get('confidence_passed', False) and not o.get('is_hallucination', False) and o.get('action') in ('BUY', 'SELL')]

    orders_to_open = []
    sl_tp_to_modify = []
    orders_to_close = []

    # 1. AI ACTIVE TRADE GUARDIAN: Manage Open Positions
    open_symbols = set()
    for pos in req.open_positions:
        open_symbols.add(pos.symbol)
        pair = pos.symbol
        pos_type = pos.type
        open_price = pos.price_open
        curr_price = pos.price_current
        sl = pos.sl
        tp = pos.tp
        profit = pos.profit
        ticket = pos.ticket
        is_gold = (pair == "XAUUSD")

        price_delta = (curr_price - open_price) if pos_type == 0 else (open_price - curr_price)
        init_sl_dist = abs(open_price - sl) if sl > 0 else (4.50 if is_gold else 0.0016)

        # 1a. Missing SL/TP -> Auto-guard with 1:2 ATR barriers
        if sl == 0.0 or tp == 0.0:
            opp_info = opps_dict.get(pair, {})
            s_dist = float(opp_info.get('sl_dist', init_sl_dist))
            t_dist = float(opp_info.get('tp_dist', s_dist * 2.0))
            new_sl = round(open_price - s_dist if pos_type == 0 else open_price + s_dist, 5 if not is_gold else 2)
            new_tp = round(open_price + t_dist if pos_type == 0 else open_price - t_dist, 5 if not is_gold else 2)
            sl_tp_to_modify.append({
                "ticket": ticket, "symbol": pair, "sl": new_sl, "tp": new_tp, "reason": "AUTO_GUARD_1TO2"
            })

        # 1b. Position is in profit -> Breakeven Lock and Trailing
        elif price_delta > 0 and profit > 0.20:
            # Stage 1: Breakeven lock at +50% SL distance
            if price_delta >= (init_sl_dist * 0.50):
                be_sl = round(open_price + (0.50 if is_gold else 0.00010) if pos_type == 0 else open_price - (0.50 if is_gold else 0.00010), 5 if not is_gold else 2)
                should_be = (pos_type == 0 and (sl == 0.0 or sl < open_price)) or (pos_type == 1 and (sl == 0.0 or sl > open_price))
                if should_be:
                    sl_tp_to_modify.append({
                        "ticket": ticket, "symbol": pair, "sl": be_sl, "tp": tp, "reason": "BREAKEVEN_LOCK_STAGE1"
                    })
            # Stage 2: Trailing stop at +100% SL distance
            if price_delta >= init_sl_dist:
                trail_gap = init_sl_dist * 0.65
                trail_sl = round(curr_price - trail_gap if pos_type == 0 else curr_price + trail_gap, 5 if not is_gold else 2)
                should_trail = (pos_type == 0 and trail_sl > sl) or (pos_type == 1 and trail_sl < sl)
                if should_trail:
                    sl_tp_to_modify.append({
                        "ticket": ticket, "symbol": pair, "sl": trail_sl, "tp": tp, "reason": "TRAILING_PROFIT_STAGE2"
                    })

        # 1c. Position is in loss / drawdown -> Adaptive Loss Cut & Breakeven Recovery
        else:
            opp_curr = opps_dict.get(pair)
            if opp_curr and opp_curr.get('status') == 'CONFIRMED' and not opp_curr.get('is_hallucination', False):
                ai_action = opp_curr.get('action')
                ai_conf = opp_curr.get('confidence', 0.0) * 100
                is_opposite = (pos_type == 0 and ai_action == "SELL") or (pos_type == 1 and ai_action == "BUY")
                drawdown_ratio = abs(price_delta) / (init_sl_dist + 1e-6)

                # Confirmed structural breakdown with >= 72% conviction & at least 50% drawdown
                if is_opposite and ai_conf >= 72.0 and drawdown_ratio >= 0.50:
                    orders_to_close.append({
                        "ticket": ticket, "symbol": pair, "volume": pos.volume, "reason": "AI_ADAPTIVE_LOSS_CUT",
                        "comment": f"Cut early at {drawdown_ratio*100:.0f}% risk | Reversal {ai_action} {ai_conf:.1f}%"
                    })

    # 2. DECIDE NEW TRADES (Strict limits: max 5 daily, max 2 concurrent, -$10 drawdown limit)
    is_drawdown_limit = (req.daily_pnl <= req.daily_max_loss)
    can_open = (req.today_trades < req.max_daily_trades) and (not is_drawdown_limit) and (len(req.open_positions) < req.max_concurrent_positions)

    if actionable and can_open:
        for opp in actionable:
            pair = opp['pair']
            if pair in open_symbols:
                continue

            # Calculate Fixed Dollar Risk Lot ($2.00 - $2.50)
            if pair == "XAUUSD":
                lot = 0.01
            else:
                sl_pips = float(opp.get('sl_pips', 18.0))
                loss_per_001 = max(1.0, sl_pips * 0.10)
                lot = round(2.50 / loss_per_001 * 0.01, 2)
                lot = max(0.01, min(0.03, lot))
                if len(req.open_positions) >= 1:
                    lot = 0.01

            orders_to_open.append({
                "symbol": pair,
                "action": opp['action'],
                "volume": lot,
                "sl": opp['stop_loss'],
                "tp": opp['take_profit'],
                "sl_pips": opp['sl_pips'],
                "tp_pips": opp['tp_pips'],
                "confidence": opp['confidence'],
                "strategy": opp['strategy'],
                "comment": "AI-Cloud-1to2"
            })
            break # Open at most 1 new position per cycle

    top_pair = opportunities[0] if opportunities else {}
    summary = f"Cloud Scan: {len(opportunities)} pairs | Best: {top_pair.get('pair', 'None')} ({top_pair.get('action', 'NEUTRAL')} {top_pair.get('confidence', 0)*100:.1f}%) | Open: {len(orders_to_open)}, Modify: {len(sl_tp_to_modify)}, Close: {len(orders_to_close)}"

    return {
        "status": "SUCCESS",
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "total_scan_latency_ms": round((time.time() - t_start) * 1000, 2),
        "cycle_summary": summary,
        "orders_to_open": orders_to_open,
        "sl_tp_to_modify": sl_tp_to_modify,
        "orders_to_close": orders_to_close,
        "top_opportunities": opportunities[:5]
    }

@app.post("/notify", tags=["Alerts"])
def send_telegram_alert(req: NotifyRequest):
    try:
        ok = notifier.send_message(req.message)
        return {"status": "SUCCESS" if ok else "FAILED", "sent": ok}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
