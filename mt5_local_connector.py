"""
Local MetaTrader 5 AI Bridge Connector
Runs locally on your Windows PC alongside MetaTrader 5.
1. Connects to your local MT5 terminal (Elefin Demo or standard MT5)
2. Streams live market data to your Cloud AI Model Server (on Render or localhost)
3. Receives institutional AI predictions (XGBoost + CatBoost ensemble)
4. Automatically executes orders on MT5 with Stop Loss & Take Profit
5. Sends real-time Telegram confirmations!
"""

import os
import sys
import time
import datetime
import argparse
import requests
import numpy as np
import pandas as pd

# Force UTF-8 terminal output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    import MetaTrader5 as mt5
except ImportError:
    print("[ERROR] MetaTrader5 package is not installed in this environment.")
    print("Please run: .\\.venv\\Scripts\\pip install -r requirements.txt")
    sys.exit(1)

from telegram_bot import TelegramNotifier

# Default Configuration
DEFAULT_RENDER_URL = os.getenv("RENDER_API_URL", "http://127.0.0.1:8000")
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "12345868444"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "Ayush5555@")
MT5_SERVER = os.getenv("MT5_SERVER", "Elefin-Trade")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8643518675:AAG_T9rUW2c2SPM8K9jjBu5Iy6gplStWe_E")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5336298229")

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "XAUUSD"]
TIMEFRAME = "15M"
TIMEFRAME_MT5 = mt5.TIMEFRAME_M15

MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "2"))
MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "5"))
DAILY_MAX_LOSS = float(os.getenv("DAILY_MAX_LOSS", "-6.0"))
DAILY_PROFIT_GOAL = float(os.getenv("DAILY_PROFIT_GOAL", "20.0"))
MAGIC_NUMBER = 888999

class LocalMT5Connector:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self.notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.daily_trade_count = 0
        self.current_date = datetime.date.today()
        self.last_traded_bar = {}

    def connect_mt5(self) -> bool:
        print("=" * 65)
        print("          LOCAL MT5 <-> CLOUD AI BRIDGE CONNECTOR            ")
        print("=" * 65)
        print(f"Cloud AI Brain Endpoint: {self.api_url}")
        print(f"Target MT5 Account      : #{MT5_LOGIN} ({MT5_SERVER})")
        print(f"Active Pairs            : {', '.join(PAIRS)}")
        print(f"Timeframe               : 15M (Forex) / 1H (Gold)")
        print("=" * 65)

        # Attempt 1: Attach to running terminal
        print("\n[1/3] Connecting to MetaTrader 5...")
        if mt5.initialize():
            term_info = mt5.terminal_info()
            acc_info = mt5.account_info()
            if acc_info and acc_info.login == MT5_LOGIN:
                print(f"[OK] Connected directly to running terminal! Balance: ${acc_info.balance:.2f}")
                return True
            else:
                # Connected to wrong account or unauthenticated, re-login
                print(f"[AUTH] Logging into #{MT5_LOGIN} on {MT5_SERVER}...")
                if mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                    print(f"[OK] Successfully logged in! Account #{MT5_LOGIN}")
                    return True

        # Attempt 2: Initialize with explicit credentials
        if mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            acc_info = mt5.account_info()
            print(f"[OK] Initialized with credentials! Balance: ${acc_info.balance:.2f}")
            return True

        print(f"[FAIL] Could not connect to MT5. Error: {mt5.last_error()}")
        return False

    def check_cloud_health(self) -> bool:
        print("[CLOUD] Pinging Cloud AI Brain (waking up if asleep, please wait ~30s)...")
        for attempt in range(1, 4):
            try:
                r = requests.get(f"{self.api_url}/", timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    print(f"✅ [CLOUD AI ONLINE] Status: {data.get('status')} | Universal Threshold: {data.get('universal_threshold')}")
                    return True
                else:
                    print(f"[WARN] Cloud AI Server returned HTTP {r.status_code}")
            except requests.exceptions.Timeout:
                print(f"[WAKING UP] Server is still booting from sleep (attempt {attempt}/3)...")
                time.sleep(5)
            except Exception as e:
                print(f"[RETRY {attempt}/3] Connecting to Cloud AI at {self.api_url}: {e}")
                time.sleep(4)
        return False

    def fetch_pair_candles(self, pair: str, count: int = 150) -> list:
        # Ensure symbol selected in Market Watch
        mt5.symbol_select(pair, True)
        tf = mt5.TIMEFRAME_H1 if pair == "XAUUSD" else TIMEFRAME_MT5
        rates = mt5.copy_rates_from_pos(pair, tf, 0, count)
        if rates is None or len(rates) < 50:
            return []

        candles = []
        for r in rates:
            candles.append({
                "time": int(r['time']),
                "open": float(r['open']),
                "high": float(r['high']),
                "low": float(r['low']),
                "close": float(r['close']),
                "volume": float(r['tick_volume'])
            })
        return candles

    def execute_trade(self, opp: dict):
        pair = opp['pair']
        action = opp['action']
        lot = float(opp.get('recommended_lot', 0.01))
        conviction = opp.get('conviction_tier', 'STANDARD')
        p_win = opp.get('win_probability', 0.5)

        # Check existing positions on this symbol
        open_positions = mt5.positions_get(symbol=pair)
        if open_positions and len(open_positions) > 0:
            print(f"[SKIP] Position already open for {pair}. Skipping duplicate order.")
            return

        all_positions = mt5.positions_get()
        if all_positions and len(all_positions) >= MAX_CONCURRENT_POSITIONS:
            print(f"[SKIP] Max concurrent positions limit ({MAX_CONCURRENT_POSITIONS}) reached.")
            return

        tick = mt5.symbol_info_tick(pair)
        sym_info = mt5.symbol_info(pair)
        if not tick or not sym_info:
            print(f"[ERROR] Could not fetch tick or symbol info for {pair}")
            return

        digits = sym_info.digits
        point = sym_info.point

        if pair == "XAUUSD":
            lot = 0.01  # Strict defensive lot size for Gold on small accounts
            sl_dist = 6.00  # $6.00 stop loss distance
            tp_dist = 4.00  # $4.00 take profit ($4.00 profit on 0.01 lot)
            if action == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                sl = round(price - sl_dist, digits)
                tp = round(price + tp_dist, digits)
            elif action == "SELL":
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                sl = round(price + sl_dist, digits)
                tp = round(price - tp_dist, digits)
            else:
                return
        else:
            if action == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                sl_pips = 30 * (point * 10 if digits in (3, 5) else point)
                tp_pips = 20 * (point * 10 if digits in (3, 5) else point)
                sl = round(price - sl_pips, digits)
                tp = round(price + tp_pips, digits)
            elif action == "SELL":
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                sl_pips = 30 * (point * 10 if digits in (3, 5) else point)
                tp_pips = 20 * (point * 10 if digits in (3, 5) else point)
                sl = round(price + sl_pips, digits)
                tp = round(price - tp_pips, digits)
            else:
                return

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pair,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": f"AI-{conviction[:4]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        print(f"\n[ORDER EXECUTION] Sending {action} {lot} lots on {pair} @ {price:.5f} (SL: {sl}, TP: {tp})...")
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ [SUCCESS] Order placed! Ticket: #{result.order} | Retcode: {result.retcode}")
            self.daily_trade_count += 1
            
            # Send Telegram confirmation
            msg = (
                f"⚡ <b>AI ORDER EXECUTED!</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"• <b>Symbol:</b> {pair} ({TIMEFRAME})\n"
                f"• <b>Action:</b> {action}\n"
                f"• <b>Volume:</b> {lot} lots ({conviction})\n"
                f"• <b>Entry Price:</b> {price:.5f}\n"
                f"• <b>Take Profit:</b> {tp:.5f}\n"
                f"• <b>Stop Loss:</b> {sl:.5f}\n"
                f"• <b>Win Probability:</b> {p_win*100:.1f}%\n"
                f"• <b>Ticket:</b> #{result.order}\n"
                f"━━━━━━━━━━━━━━━━━━"
            )
            self.notifier.send_message(msg)
        else:
            err = result.comment if result else mt5.last_error()
            print(f"❌ [ORDER REJECTED] Code: {result.retcode if result else 'None'} | Reason: {err}")

    def run_loop(self, poll_interval: int = 15):
        if not self.connect_mt5():
            print("[FATAL] MT5 connection failed. Exiting.")
            sys.exit(1)

        self.check_cloud_health()
        print(f"\n[ACTIVE] Starting continuous scanner loop (checking every {poll_interval}s)...")
        print("Press Ctrl+C anytime to stop.\n")

        while True:
            try:
                # Reset daily counter on date change
                if datetime.date.today() != self.current_date:
                    self.current_date = datetime.date.today()
                    self.daily_trade_count = 0

                acc = mt5.account_info()
                balance = acc.balance if acc else 100.0
                equity = acc.equity if acc else 100.0

                # 1. Collect candle data across all pairs
                market_data = {}
                for p in PAIRS:
                    c = self.fetch_pair_candles(p, count=150)
                    if c:
                        market_data[p] = c

                if not market_data:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Waiting for market data...")
                    time.sleep(poll_interval)
                    continue

                # 2. Call Cloud AI Server
                payload = {
                    "timeframe": TIMEFRAME,
                    "market_data": market_data,
                    "account_balance": balance
                }

                t0 = time.time()
                try:
                    resp = requests.post(f"{self.api_url}/scan", json=payload, timeout=25)
                    scan_res = resp.json()
                except Exception as e:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Cloud AI request error: {e}")
                    time.sleep(poll_interval)
                    continue

                latency = round((time.time() - t0) * 1000, 1)
                total_scanned = scan_res.get("scanned_pairs", 0)
                actionable = scan_res.get("actionable", [])
                top_opps = scan_res.get("top_opportunities", [])

                timestamp_str = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp_str}] Scanned {total_scanned} pairs | Cloud Latency: {latency}ms | Equity: ${equity:.2f} | Actionable Signals: {len(actionable)}")

                # Print top opportunity
                if top_opps:
                    top = top_opps[0]
                    passed = "PASS" if top.get('confidence_passed') else "WAIT"
                    print(f"   -> Top: {top.get('pair')} | Action: {top.get('action')} | WinProb: {top.get('win_probability',0)*100:.1f}% | Uncertainty: {top.get('epistemic_uncertainty',0):.3f} | [{passed}]")

                # 3. Execute any actionable signals
                if actionable and self.daily_trade_count < MAX_DAILY_TRADES:
                    for opp in actionable:
                        pair = opp['pair']
                        candles = market_data.get(pair, [])
                        last_bar_time = candles[-1]['time'] if candles else 0
                        # Avoid repeated trades on the same candle
                        if self.last_traded_bar.get(pair) == last_bar_time:
                            continue

                        self.execute_trade(opp)
                        self.last_traded_bar[pair] = last_bar_time

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                print("\n[STOPPED] Local MT5 Connector terminated by user.")
                mt5.shutdown()
                break
            except Exception as e:
                print(f"[UNEXPECTED ERROR] {e}")
                time.sleep(poll_interval)

def main():
    parser = argparse.ArgumentParser(description="Local MT5 AI Bridge Connector")
    parser.add_argument("--url", default=DEFAULT_RENDER_URL, help="Cloud AI Server URL (e.g. https://your-app.onrender.com)")
    parser.add_argument("--interval", type=int, default=15, help="Scan interval in seconds (default: 15)")
    args = parser.parse_args()

    connector = LocalMT5Connector(api_url=args.url)
    connector.run_loop(poll_interval=args.interval)

if __name__ == "__main__":
    main()
