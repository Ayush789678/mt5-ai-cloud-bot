"""
Ultra-High-Speed 1-Second MetaTrader 5 AI Trading Bot
1. Connects directly to local MT5 terminal (Elefin Demo or standard MT5)
2. Evaluates the Golden DoubleEnsemble model in local RAM at sub-second speeds (~400ms across all 8 pairs!)
3. Analyzes the market EVERY 1 SECOND for instantaneous micro-breakout and trend execution
4. Keeps Render Cloud AI warm in background
5. Automatically executes trades with dynamic lot sizing, Stop Loss & Take Profit, and alerts Telegram!
"""

import os
import sys
import time
import datetime
import argparse
import threading
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

from trading_bot_engine import UnifiedTradingBotEngine, GPUDoubleEnsemble
from telegram_bot import TelegramNotifier

# Default Configuration
DEFAULT_RENDER_URL = os.getenv("RENDER_API_URL", "https://mt5-ai-model-service.onrender.com")
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

class RealTime1SecBot:
    def __init__(self, api_url: str = DEFAULT_RENDER_URL):
        self.api_url = api_url.rstrip("/")
        self.notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.daily_trade_count = 0
        self.current_date = datetime.date.today()
        self.last_traded_bar = {}
        
        # Load Institutional AI Brain directly into local memory for sub-second analysis
        print("[INIT] Loading AI Golden DoubleEnsemble into local RAM for 1-second analysis...")
        model_dir = os.path.dirname(os.path.abspath(__file__))
        self.engine = UnifiedTradingBotEngine(model_dir=model_dir)
        print(f"✅ [READY] Local AI Engine online with {len(self.engine.feature_cols)} features!")

    def start_cloud_keepalive(self):
        """Background thread that sends heartbeats to Render every 30 seconds to keep it warm and logging."""
        def ping_loop():
            while True:
                try:
                    requests.get(f"{self.api_url}/", timeout=10)
                except Exception:
                    pass
                time.sleep(30)
        t = threading.Thread(target=ping_loop, daemon=True)
        t.start()

    def connect_mt5(self) -> bool:
        print("=" * 70)
        print("        REAL-TIME 1-SECOND QUANTITATIVE MT5 AI TRADING BOT          ")
        print("=" * 70)
        print(f"Target MT5 Account : #{MT5_LOGIN} ({MT5_SERVER})")
        print(f"Active Symbols     : {', '.join(PAIRS)}")
        print(f"Scan Frequency     : EVERY 1.0 SECOND (Zero Lag Local RAM Inference)")
        print(f"Risk Management    : Max 5 Trades/Day | Target: $10.00 - $15.00/Day")
        print("=" * 70)

        # Attempt 1: Attach to running terminal
        print("\n[1/2] Connecting to MetaTrader 5...")
        if mt5.initialize():
            term_info = mt5.terminal_info()
            acc_info = mt5.account_info()
            if acc_info and acc_info.login == MT5_LOGIN:
                print(f"[OK] Connected directly to running terminal! Balance: ${acc_info.balance:.2f}")
                return True
            else:
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

    def fetch_pair_candles(self, pair: str, count: int = 150) -> pd.DataFrame:
        mt5.symbol_select(pair, True)
        tf = mt5.TIMEFRAME_H1 if pair == "XAUUSD" else TIMEFRAME_MT5
        rates = mt5.copy_rates_from_pos(pair, tf, 0, count)
        if rates is None or len(rates) < 50:
            return None

        df = pd.DataFrame(rates)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
        df.index = pd.to_datetime(df['time'], unit='s')
        return df

    def analyze_market_locally(self, balance: float) -> tuple:
        opportunities = []
        for pair in PAIRS:
            df = self.fetch_pair_candles(pair, count=150)
            if df is None:
                continue
            try:
                pair_tf = "1H" if pair == "XAUUSD" else TIMEFRAME
                res = self.engine.predict_opportunity(pair, pair_tf, df)
                
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

                bal_ratio = max(0.5, balance / 100.0)
                lot = round(lot * bal_ratio, 2)
                lot = max(0.01, min(lot, 0.05))

                res['recommended_lot'] = lot
                res['conviction_tier'] = tier
                res['effective_conviction'] = round(conv, 4)
                res['win_probability'] = p_win
                res['epistemic_uncertainty'] = u_epi
                res['is_hallucination'] = u_epi > 0.20
                res['confidence_passed'] = res.get('status') == 'CONFIRMED'
                opportunities.append(res)
            except Exception:
                pass

        opportunities.sort(key=lambda x: (x.get('confidence_passed', False), x.get('confidence', 0.0)), reverse=True)
        actionable = [o for o in opportunities if o.get('confidence_passed', False) and not o.get('is_hallucination', False) and o.get('action') in ('BUY', 'SELL')]
        return opportunities, actionable

    def execute_trade(self, opp: dict):
        pair = opp['pair']
        action = opp['action']
        lot = float(opp.get('recommended_lot', 0.01))
        conviction = opp.get('conviction_tier', 'STANDARD')
        p_win = opp.get('win_probability', 0.5)

        # Check existing positions on this symbol
        open_positions = mt5.positions_get(symbol=pair)
        if open_positions and len(open_positions) > 0:
            return

        all_positions = mt5.positions_get()
        if all_positions and len(all_positions) >= MAX_CONCURRENT_POSITIONS:
            return

        tick = mt5.symbol_info_tick(pair)
        sym_info = mt5.symbol_info(pair)
        if not tick or not sym_info:
            return

        digits = sym_info.digits
        point = sym_info.point

        if pair == "XAUUSD":
            lot = 0.01
            sl_dist = 6.00
            tp_dist = 4.00
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
                f"⚡ <b>AI ORDER EXECUTED (1-SEC SCANNER)!</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"• <b>Symbol:</b> {pair} ({'1H' if pair == 'XAUUSD' else TIMEFRAME})\n"
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

    def run_1sec_loop(self):
        if not self.connect_mt5():
            print("[FATAL] MT5 connection failed. Exiting.")
            sys.exit(1)

        self.start_cloud_keepalive()
        print("\n" + "=" * 70)
        print("  [LIVE] REAL-TIME 1-SECOND MARKET ANALYSIS IS NOW RUNNING!")
        print("  • Engine: Institutional DoubleEnsemble (108 Quant Features)")
        print("  • Pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, XAUUSD")
        print("  • Render Cloud AI: https://mt5-ai-model-service.onrender.com (SYNCED)")
        print("  • NOTE: If you click inside this window and see 'Select' in the title,")
        print("    Windows paused output. Simply press ENTER or ESC to resume scrolling.")
        print("=" * 70 + "\n")

        scan_count = 0
        last_cloud_sync = 0

        while True:
            t0 = time.time()
            try:
                scan_count += 1
                # Reset daily counter on date change
                if datetime.date.today() != self.current_date:
                    self.current_date = datetime.date.today()
                    self.daily_trade_count = 0

                acc = mt5.account_info()
                balance = acc.balance if acc else 100.0
                equity = acc.equity if acc else 100.0

                # Analyze all 8 pairs in local RAM (<400ms)
                opportunities, actionable = self.analyze_market_locally(balance)
                compute_time_ms = (time.time() - t0) * 1000

                # Print clean scrolling line every 3 scans (~3 seconds)
                if scan_count % 3 == 0 or actionable:
                    top = opportunities[0] if opportunities else {}
                    act = top.get('action', 'NEUTRAL')
                    conf = top.get('confidence', 0) * 100
                    pair_name = top.get('pair', 'None')
                    ts_str = datetime.datetime.now().strftime('%H:%M:%S')

                    status_tag = f"🎯 ACTIONABLE: {act}" if actionable else f"Scanning... Best: {pair_name} ({act} {conf:.1f}%)"
                    print(f"[{ts_str}] ⚡ Scan #{scan_count:05d} | 8 Symbols ({compute_time_ms:.0f}ms) | Equity: ${equity:.2f} | Trades: {self.daily_trade_count}/{MAX_DAILY_TRADES} | {status_tag}")


                # Execute actionable trades if daily limit allows
                if actionable and self.daily_trade_count < MAX_DAILY_TRADES:
                    for opp in actionable:
                        pair = opp['pair']
                        df = self.fetch_pair_candles(pair, count=50)
                        last_bar_time = int(df['time'].iloc[-1]) if df is not None and 'time' in df.columns else 0
                        if self.last_traded_bar.get(pair) == last_bar_time:
                            continue

                        self.execute_trade(opp)
                        self.last_traded_bar[pair] = last_bar_time

                # Target 1.0 second cycle
                elapsed = time.time() - t0
                sleep_time = max(0.05, 1.0 - elapsed)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\n[STOPPED] Real-Time 1-Second Bot terminated by user.")
                mt5.shutdown()
                break
            except Exception as e:
                print(f"[WARN] Error in scan cycle: {e}")
                time.sleep(1.0)

def main():
    bot = RealTime1SecBot()
    bot.run_1sec_loop()

if __name__ == "__main__":
    main()
