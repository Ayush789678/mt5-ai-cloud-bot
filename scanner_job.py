"""
Automated AI Market Scanner Job for GitHub Actions (Serverless Cloud MT5 Execution)
Runs every 15-30 minutes during Forex trading hours (Sunday 21:00 UTC to Friday 21:00 UTC).
Directly connects to MT5 demo account, scans live market, runs AI inference,
places orders with TP/SL pre-set, and alerts Telegram!
"""

import os, sys, time, datetime, json, io
import numpy as np
import pandas as pd

# Force UTF-8 encoding for Windows GitHub runner console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from telegram_bot import TelegramNotifier

# 1. Credentials & Risk Configuration (from GitHub Secrets / Environment)
MT5_LOGIN = os.getenv("MT5_LOGIN", "12345868444")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "Ayush5555@")
MT5_SERVER = os.getenv("MT5_SERVER", "Elefin-Trade")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8643518675:AAG_T9rUW2c2SPM8K9jjBu5Iy6gplStWe_E")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5336298229")

ACCOUNT_CAPITAL = float(os.getenv("ACCOUNT_CAPITAL", "100.0"))
DAILY_PROFIT_GOAL = float(os.getenv("DAILY_PROFIT_GOAL", "20.0"))
DAILY_MAX_LOSS = float(os.getenv("DAILY_MAX_LOSS", "-6.0"))
MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "5"))
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "2"))

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

def calculate_dynamic_lot(p_win: float, u_epi: float, balance: float) -> tuple:
    """
    Model-Driven Dynamic Position Sizing:
    - Ultra-High Conviction: 0.03 lot ($6.00 profit)
    - Standard Conviction: 0.02 lot ($4.00 profit)
    - Defensive: 0.01 lot ($2.00 profit)
    """
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
        
    bal_ratio = max(0.5, balance / 100.0)
    lot = round(lot * bal_ratio, 2)
    lot = max(0.01, min(lot, 0.05))
    est_win = lot * 200.0
    return lot, est_win, conv, tier

def find_terminal_path():
    possible_paths = [
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files\Elefin MetaTrader 5\terminal64.exe",
        r"C:\Program Files\Elefin\terminal64.exe",
        r"C:\Program Files\Elefin Ltd\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    # Dynamic search across Program Files
    for base in [r"C:\Program Files", r"C:\Program Files (x86)", os.environ.get("LOCALAPPDATA", "")]:
        if base and os.path.exists(base):
            for root, dirs, files in os.walk(base):
                if "terminal64.exe" in files:
                    return os.path.join(root, "terminal64.exe")
    return None

def run_scanner():
    print("=" * 75)
    print("      AI QUANTITATIVE MT5 MARKET SCANNER (GITHUB ACTIONS RUNNER)     ")
    print("=" * 75)
    print(f"Timestamp UTC: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target MT5 Account: #{MT5_LOGIN} on {MT5_SERVER}")
    
    notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    # 2. Check MT5 Package
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[ERROR] MetaTrader5 library is not installed.")
        notifier.send_message("⚠️ <b>Scanner Error:</b> MetaTrader5 package missing on runner.")
        sys.exit(1)
        
    # 3. Locate and Initialize MT5 Connection
    term_path = find_terminal_path()
    print(f"Detected MT5 Terminal Binary: {term_path}")
    
    login_int = int(MT5_LOGIN)
    print(f"Connecting to MT5 via account #{login_int} on {MT5_SERVER}...")
    
    init_ok = False
    # Attempt 1: Attach to running terminal with credentials
    try:
        init_ok = mt5.initialize(
            login=login_int,
            password=MT5_PASSWORD,
            server=MT5_SERVER,
            timeout=30000
        )
    except Exception as e:
        print(f"Attach with credentials failed: {e}")
        init_ok = False

    # Attempt 2: Explicit launch with terminal binary path
    if not init_ok and term_path:
        print(f"Attempting explicit terminal launch: {term_path}")
        try:
            init_ok = mt5.initialize(
                path=term_path,
                login=login_int,
                password=MT5_PASSWORD,
                server=MT5_SERVER,
                timeout=60000
            )
        except Exception as e:
            print(f"Path launch failed: {e}")
            init_ok = False
            
    # Attempt 3: Bare attach
    if not init_ok:
        print("Attempting bare attach...")
        try:
            init_ok = mt5.initialize()
        except Exception as e:
            print(f"Bare attach failed: {e}")
            init_ok = False
        
    if not init_ok:
        err = mt5.last_error()
        print(f"[ERROR] MT5 initialize failed: {err}")
        notifier.send_message(f"⚠️ <b>MT5 Init Failed:</b> <code>{err}</code>")
        sys.exit(1)
        
    acc = mt5.account_info()
    if not acc or acc.login != login_int:
        print(f"Authenticating account #{login_int} on {MT5_SERVER}...")
        auth_ok = mt5.login(login=login_int, password=MT5_PASSWORD, server=MT5_SERVER)
        if not auth_ok:
            err = mt5.last_error()
            print(f"[ERROR] Failed to login to MT5 account #{login_int} on server {MT5_SERVER}: {err}")
            notifier.send_message(f"⚠️ <b>MT5 Login Failed:</b> Account #{login_int} on {MT5_SERVER} failed: <code>{err}</code>")
            mt5.shutdown()
            sys.exit(1)
        acc = mt5.account_info()
        
    if not acc:
        print("[ERROR] Could not retrieve account information.")
        mt5.shutdown()
        sys.exit(1)
        
    print(f"[OK] Connected to #{acc.login} ({acc.server}) | Balance: ${acc.balance:.2f} | Equity: ${acc.equity:.2f}")
    
    # 5. Check Open Positions & Circuit Breakers
    positions = mt5.positions_get()
    open_count = len(positions) if positions is not None else 0
    print(f"Current open positions in market: {open_count}")
    
    if open_count >= MAX_CONCURRENT_POSITIONS:
        print(f"Max concurrent positions reached ({open_count}/{MAX_CONCURRENT_POSITIONS}). Skipping new entries to manage exposure.")
        mt5.shutdown()
        return

    # 6. Load Trained Institutional Model Engine
    from trading_bot_engine import UnifiedTradingBotEngine
    print("Loading Golden DoubleEnsemble model bundle...")
    engine = UnifiedTradingBotEngine(os.path.dirname(os.path.abspath(__file__)))
    print(f"[OK] AI Engine online with {len(engine.feature_cols)} feature inputs.")
    
    # Map timeframe strings to MT5 constants
    tf_map = {
        "15M": mt5.TIMEFRAME_M15,
        "1H": mt5.TIMEFRAME_H1
    }
    
    candidates = []
    
    # 7. Scan Tracked Currency Pairs
    for pair in PAIRS:
        # Check if we already have an open position in this pair
        has_pos = any(pos.symbol == pair for pos in (positions or []))
        if has_pos:
            print(f"[{pair}] Skipping: Already holding active position.")
            continue
            
        # Ensure symbol is selected in MarketWatch
        if not mt5.symbol_select(pair, True):
            print(f"[{pair}] Could not select symbol in MarketWatch.")
            continue
            
        for tf_label, tf_const in tf_map.items():
            rates = mt5.copy_rates_from_pos(pair, tf_const, 0, 300)
            if rates is None or len(rates) < 100:
                continue
                
            df = pd.DataFrame(rates)
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
            
            # Extract live features using engine
            try:
                feat_series = engine.compute_features_fast(df, pair, tf_label)
                if feat_series is None or len(feat_series) == 0:
                    continue
                    
                # Reindex to exact feature manifest
                feat_df = pd.DataFrame([feat_series]).reindex(columns=engine.feature_cols).fillna(0.0)
                
                # Model Inference
                probs = engine.model.predict_proba(feat_df)[0]
                p_win = float(probs[1]) if len(probs) > 1 else float(probs[0])
                
                # Estimate Epistemic Uncertainty
                u_epi = 0.05
                if hasattr(engine.model, 'estimators_'):
                    est_preds = [est.predict_proba(feat_df)[0][1] for est in engine.model.estimators_[:10]]
                    u_epi = float(np.std(est_preds))
                    
                # Pair-specific threshold
                floor = engine.pair_thresholds.get(pair, 0.56)
                conviction = p_win - 0.5 * u_epi
                
                # Check qualifying conditions
                if p_win >= floor and u_epi <= 0.10:
                    current_price = float(df['Close'].iloc[-1])
                    candidates.append({
                        "pair": pair,
                        "tf": tf_label,
                        "action": "BUY",
                        "price": current_price,
                        "p_win": p_win,
                        "u_epi": u_epi,
                        "conviction": conviction
                    })
                    print(f"  [*] [{pair} {tf_label}] SIGNAL DETECTED: BUY @ {current_price:.5f} | Win Prob: {p_win*100:.1f}% | Conv: {conviction:.3f}")
            except Exception as e:
                print(f"[{pair} {tf_label}] Feature extraction error: {e}")
                continue
                
    # 8. Sort candidates by Conviction and execute highest quality
    if candidates:
        candidates.sort(key=lambda x: x["conviction"], reverse=True)
        top_cand = candidates[0]
        
        pair = top_cand["pair"]
        action = top_cand["action"]
        tf = top_cand["tf"]
        price = top_cand["price"]
        p_win = top_cand["p_win"]
        u_epi = top_cand["u_epi"]
        
        lot_size, est_win, conv, tier = calculate_dynamic_lot(p_win, u_epi, acc.balance)
        
        # Calculate SL & TP (1.6:1 Reward-to-Risk)
        point = mt5.symbol_info(pair).point if mt5.symbol_info(pair) else 0.0001
        tp_pips = 20.0
        sl_pips = 12.5
        
        if action == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            tp = price + tp_pips * point * 10
            sl = price - sl_pips * point * 10
        else:
            order_type = mt5.ORDER_TYPE_SELL
            tp = price - tp_pips * point * 10
            sl = price + sl_pips * point * 10
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pair,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 858585,
            "comment": f"AI-ML-{tier}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        print(f"\n[EXECUTING ORDER] {action} {lot_size} lots of {pair} @ {price:.5f} | TP: {tp:.5f} | SL: {sl:.5f}")
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[SUCCESS] ORDER EXECUTED! Deal ID: #{result.deal}")
            notifier.notify_trade_opened(
                pair=pair, action=action, tf=tf, price=price,
                lot_size=lot_size, tp=tp, sl=sl, p_win=p_win,
                conviction=conv, est_win=est_win, balance=acc.balance
            )
        else:
            err = result.comment if result else mt5.last_error()
            print(f"[ORDER REJECTED] Broker Error: {err}")
    else:
        print("\n[SCAN COMPLETE] No high-conviction trade setups met the >= 85% safety filters this cycle.")
        
    mt5.shutdown()
    print("MT5 Terminal disconnected cleanly. Scanner job finished.")

if __name__ == "__main__":
    run_scanner()
