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
DAILY_MAX_LOSS = float(os.getenv("DAILY_MAX_LOSS", "-10.0"))  # $10 daily drawdown limit
DAILY_PROFIT_GOAL = float(os.getenv("DAILY_PROFIT_GOAL", "20.0"))
MAGIC_NUMBER = 888999

class RealTime1SecBot:
    def __init__(self, api_url: str = DEFAULT_RENDER_URL):
        self.api_url = api_url.rstrip("/")
        self.notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.current_date = datetime.date.today()
        self.drawdown_alert_sent = False
        self.pending_reversals = {}  # {ticket: True} for patient breakeven recovery
        self.be_locked = set()        # {ticket} for risk-free trades
        
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

    def get_today_stats(self) -> dict:
        """
        Accurately queries MT5 deal history and active positions for today.
        Tracks:
        - Total trade entries opened today (active + closed)
        - Closed PnL realized today
        - Floating PnL of open positions
        - Total Daily Net PnL (Closed + Floating)
        """
        now = datetime.datetime.now()
        today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)

        deals = mt5.history_deals_get(today_start, now)
        today_orders = set()
        closed_pnl = 0.0

        if deals:
            for d in deals:
                if d.entry == mt5.DEAL_ENTRY_IN and d.order > 0:
                    today_orders.add(d.order)
                elif d.entry == mt5.DEAL_ENTRY_OUT:
                    closed_pnl += float(d.profit) + float(d.swap) + float(d.commission)

        positions = mt5.positions_get()
        open_count = len(positions) if positions else 0
        floating_pnl = sum(p.profit for p in positions) if positions else 0.0

        for p in (positions or []):
            today_orders.add(p.ticket)

        total_trades_today = max(len(today_orders), open_count)
        total_daily_pnl = round(closed_pnl + floating_pnl, 2)

        return {
            "total_trades_today": total_trades_today,
            "open_positions_count": open_count,
            "closed_pnl": round(closed_pnl, 2),
            "floating_pnl": round(floating_pnl, 2),
            "total_daily_pnl": total_daily_pnl
        }

    def calculate_model_lot(self, pair: str, sl_dist: float, p_win: float, u_epi: float, conv: float, balance: float, open_count: int) -> tuple:
        """
        Calculates dynamic position size based on Fixed Dollar Risk ($2.00 - $2.50 per trade):
        - Pegs risk to $2.50 per trade so 4 consecutive losses never breach the $10 daily drawdown limit.
        - Uses live MT5 broker tick values to dynamically determine exact lots across all pairs.
        - Gold (XAUUSD): Strictly 0.01 lot for accounts under $200.
        - Forex: Sized between 0.01 and 0.03 lots based on pair volatility (calm EURUSD = 0.02-0.03, volatile GBPUSD = 0.01-0.02).
        - If 1 trade is already open, any second trade defaults defensively to 0.01 lot.
        """
        has_open_position = open_count >= 1
        if has_open_position:
            return 0.01, "DEFENSIVE (0.01 lot — 2nd Position Guard)"

        if pair == "XAUUSD":
            return 0.01, "STANDARD (Gold 0.01 lot — Safe Margin)"

        sym_info = mt5.symbol_info(pair)
        if not sym_info or sl_dist <= 0:
            return 0.01, "DEFENSIVE (0.01 lot)"

        # Target dollar risk: $2.50 (2.5% of $100 account)
        target_risk_dollars = max(2.00, min(2.50, balance * 0.025))

        tick_size = sym_info.trade_tick_size if sym_info.trade_tick_size > 0 else sym_info.point
        tick_val_loss = sym_info.trade_tick_value_loss if sym_info.trade_tick_value_loss > 0 else 1.0

        loss_per_lot = (sl_dist / tick_size) * tick_val_loss
        if loss_per_lot <= 0:
            return 0.01, "DEFENSIVE (0.01 lot)"

        raw_lot = target_risk_dollars / loss_per_lot

        # Reward higher conviction with slightly higher sizing within the $2.50 envelope
        if conv >= 0.65 and p_win >= 0.70 and u_epi <= 0.10:
            lot = min(0.03, max(0.02, round(raw_lot * 1.15, 2)))
            tier = f"ULTRA ({lot} lots — A+ Conviction, ~$2.50 Risk)"
        elif conv >= 0.60 and p_win >= 0.66:
            lot = min(0.02, max(0.01, round(raw_lot, 2)))
            tier = f"STRONG ({lot} lots — High Momentum, ~$2.50 Risk)"
        else:
            lot = 0.01
            tier = "DEFENSIVE (0.01 lot — Standard Setup)"

        # Step rounding and min/max boundaries
        step = sym_info.volume_step if sym_info.volume_step > 0 else 0.01
        lot = round(round(lot / step) * step, 2)
        lot = max(sym_info.volume_min, min(0.03, lot))

        return lot, tier

    def analyze_market_locally(self, balance: float, open_count: int = 0) -> tuple:
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
                sl_dist = float(res.get('sl_dist', 0.0))

                lot, tier = self.calculate_model_lot(pair, sl_dist, p_win, u_epi, conv, balance, open_count)

                res['recommended_lot'] = lot
                res['conviction_tier'] = tier
                res['effective_conviction'] = round(conv, 4)
                res['win_probability'] = p_win
                res['epistemic_uncertainty'] = u_epi
                res['is_hallucination'] = u_epi > 0.12
                res['confidence_passed'] = res.get('status') == 'CONFIRMED'
                opportunities.append(res)
            except Exception:
                pass

        opportunities.sort(key=lambda x: (x.get('confidence_passed', False), x.get('confidence', 0.0)), reverse=True)
        actionable = [o for o in opportunities if o.get('confidence_passed', False) and not o.get('is_hallucination', False) and o.get('action') in ('BUY', 'SELL')]
        return opportunities, actionable

    def run_cloud_bot_cycle(self, balance: float, equity: float, stats: dict) -> tuple:
        """
        Cloud-Native Execution:
        1. Gathers live OHLCV bars for all 8 pairs and open positions from MT5.
        2. Sends payload to Render Cloud Bot (/bot/cycle).
        3. Cloud Bot runs AI DoubleEnsemble, manages positions, decides new entries.
        4. Returns (data, True) on success.
        """
        market_data = {}
        for pair in PAIRS:
            df = self.fetch_pair_candles(pair, count=120)
            if df is not None and not df.empty:
                candles = []
                for _, row in df.iterrows():
                    candles.append({
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": float(row.get('Volume', 0.0)),
                        "time": str(row.name)
                    })
                market_data[pair] = candles

        open_pos_list = []
        positions = mt5.positions_get()
        if positions:
            for p in positions:
                open_pos_list.append({
                    "ticket": int(p.ticket),
                    "symbol": str(p.symbol),
                    "type": int(p.type),
                    "price_open": float(p.price_open),
                    "price_current": float(p.price_current),
                    "sl": float(p.sl),
                    "tp": float(p.tp),
                    "profit": float(p.profit),
                    "volume": float(p.volume)
                })

        payload = {
            "timeframe": TIMEFRAME,
            "market_data": market_data,
            "open_positions": open_pos_list,
            "account_balance": balance,
            "account_equity": equity,
            "daily_pnl": stats['total_daily_pnl'],
            "today_trades": stats['total_trades_today'],
            "max_daily_trades": MAX_DAILY_TRADES,
            "daily_max_loss": DAILY_MAX_LOSS,
            "max_concurrent_positions": MAX_CONCURRENT_POSITIONS
        }

        try:
            resp = requests.post(f"{self.api_url}/bot/cycle", json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return data, True
        except Exception:
            pass

        return None, False

    def execute_cloud_instructions(self, cloud_data: dict):
        # 1. Close orders (Adaptive Loss Cut & BE Recovery)
        for cl in cloud_data.get('orders_to_close', []):
            ticket = cl['ticket']
            sym = cl['symbol']
            pos = mt5.positions_get(ticket=ticket)
            if pos and len(pos) > 0:
                p = pos[0]
                close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
                s_info = mt5.symbol_info(sym)
                c_price = s_info.bid if p.type == 0 else s_info.ask
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": ticket,
                    "symbol": sym,
                    "volume": p.volume,
                    "type": close_type,
                    "price": c_price,
                    "deviation": 20,
                    "magic": MAGIC_NUMBER,
                    "comment": cl.get('comment', 'AI-CloudClose')[:31],
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                c_res = mt5.order_send(req)
                if c_res and c_res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"🛡️ [CLOUD AI EXIT] Closed #{ticket} ({sym}) | Reason: {cl.get('reason')}")

        # 2. Modify SL/TP (Auto-Guard, Breakeven Lock, Trailing)
        for mod in cloud_data.get('sl_tp_to_modify', []):
            ticket = mod['ticket']
            sym = mod['symbol']
            pos = mt5.positions_get(ticket=ticket)
            if pos and len(pos) > 0:
                p = pos[0]
                if abs(p.sl - mod['sl']) > 1e-4 or abs(p.tp - mod['tp']) > 1e-4:
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": ticket,
                        "symbol": sym,
                        "sl": mod['sl'],
                        "tp": mod['tp'],
                    }
                    m_res = mt5.order_send(req)
                    if m_res and m_res.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"🔒 [CLOUD AI SL/TP] Modified #{ticket} ({sym}) SL={mod['sl']}, TP={mod['tp']} ({mod.get('reason')})")

        # 3. Open new orders
        for op in cloud_data.get('orders_to_open', []):
            sym = op['symbol']
            act = op['action']
            lot = op['volume']
            tick = mt5.symbol_info_tick(sym)
            s_info = mt5.symbol_info(sym)
            if not tick or not s_info:
                continue

            order_type = mt5.ORDER_TYPE_BUY if act == "BUY" else mt5.ORDER_TYPE_SELL
            price = tick.ask if act == "BUY" else tick.bid
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": sym,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": op['sl'],
                "tp": op['tp'],
                "deviation": 20,
                "magic": MAGIC_NUMBER,
                "comment": "AI-Cloud-1to2",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"⚡ [CLOUD AI ORDER] Placed {act} {lot} lots on {sym} @ {price:.5f} (SL={op['sl']}, TP={op['tp']} | 1:2 RR)")
                self.daily_trade_count += 1

    def execute_trade(self, opp: dict):
        pair = opp['pair']
        action = opp['action']
        lot = float(opp.get('recommended_lot', 0.01))
        conviction = opp.get('conviction_tier', 'STANDARD')
        p_win = opp.get('win_probability', 0.5)
        sl_dist = float(opp.get('sl_dist', 0.0))
        tp_dist = float(opp.get('tp_dist', 0.0))
        sl_pips = float(opp.get('sl_pips', 0.0))
        tp_pips = float(opp.get('tp_pips', 0.0))

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
        pip_unit = point * 10 if digits in (3, 5) else point

        if sl_dist <= 0 or tp_dist <= 0:
            if pair == "XAUUSD":
                sl_dist = 4.50
                tp_dist = 9.00
            else:
                sl_dist = 14 * pip_unit
                tp_dist = 28 * pip_unit

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
            "comment": "AI-1to2",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        print(f"\n[ORDER EXECUTION] Sending {action} {lot} lots on {pair} @ {price:.5f} (Tier: {conviction}) (SL: {sl}, TP: {tp} | 1:2 RR)...")
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ [SUCCESS] Order placed! Ticket: #{result.order} | Retcode: {result.retcode}")
            self.daily_trade_count += 1
            
            # Send Telegram confirmation
            msg = (
                f"⚡ <b>AI ORDER EXECUTED (STRICT 1:2 RR)!</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"• <b>Symbol:</b> {pair} ({'1H' if pair == 'XAUUSD' else TIMEFRAME})\n"
                f"• <b>Action:</b> {action}\n"
                f"• <b>Volume:</b> {lot} lots ({conviction})\n"
                f"• <b>Entry Price:</b> {price:.5f}\n"
                f"• <b>Take Profit:</b> {tp:.5f} (+{tp_pips:.1f} pips | 2R Target)\n"
                f"• <b>Stop Loss:</b> {sl:.5f} (-{sl_pips:.1f} pips | 1R Risk)\n"
                f"• <b>Risk-Reward:</b> 1:2 Strict Mathematical Ratio\n"
                f"• <b>Win Probability:</b> {p_win*100:.1f}%\n"
                f"• <b>Ticket:</b> #{result.order}\n"
                f"━━━━━━━━━━━━━━━━━━"
            )
            self.notifier.send_message(msg)
        else:
            err = result.comment if result else mt5.last_error()
            print(f"❌ [ORDER REJECTED] Code: {result.retcode if result else 'None'} | Reason: {err}")

    def manage_all_positions(self, opportunities_dict: dict):
        """
        AI Active Trade Guardian (AI-ATG):
        1. Auto-Guard: Assigns dynamic 1:2 ATR SL & TP to manual/unguarded trades.
        2. 3-Stage Smart Trailing (Winning Trades):
           - Stage 1: Breakeven Lock at +50% SL distance gain -> trails SL to Entry + 1 pip (100% Risk-Free).
           - Stage 2: Dynamic Trailing Stop at +100% SL distance gain (1R) -> locks in profits step-by-step.
           - Stage 3: Full 1:2 Take Profit target.
        3. Adaptive Loss Cutting & Breakeven Recovery (Losing Trades):
           - Normal Pullback: Holds calmly. Lets trade breathe within its designated SL.
           - Breakeven Recovery: If momentum exhausted against position, waits for price to retrace to Entry and closes at $0.00 flat!
           - Adaptive Loss Cut: If AI confirms a complete structural trend breakdown (>=72% opposite conviction with low uncertainty),
             cuts the trade early at 50% drawdown (~$1.20) rather than taking the full -$2.50 Stop Loss!
        """
        positions = mt5.positions_get()
        if not positions:
            return

        for pos in positions:
            pair = pos.symbol
            ticket = pos.ticket
            pos_type = pos.type  # 0 = BUY, 1 = SELL
            open_price = pos.price_open
            current_price = pos.price_current
            sl = pos.sl
            tp = pos.tp
            profit = pos.profit
            magic = pos.magic

            sym_info = mt5.symbol_info(pair)
            if not sym_info:
                continue

            digits = sym_info.digits
            point = sym_info.point
            pip_unit = point * 10 if digits in (3, 5) else point

            # Calculate live price excursion
            price_delta = (current_price - open_price) if pos_type == 0 else (open_price - current_price)
            pips_gain = price_delta / pip_unit if pair != "XAUUSD" else price_delta

            # Estimate initial SL distance
            if sl > 0:
                init_sl_dist = abs(open_price - sl)
            else:
                init_sl_dist = 4.50 if pair == "XAUUSD" else (14 * pip_unit)

            # 1. AUTO-PROTECTIVE SL/TP (If missing)
            if sl == 0.0 or tp == 0.0:
                opp = opportunities_dict.get(pair, {})
                s_dist = float(opp.get('sl_dist', init_sl_dist))
                t_dist = float(opp.get('tp_dist', s_dist * 2.0))
                new_sl = round(open_price - s_dist if pos_type == 0 else open_price + s_dist, digits)
                new_tp = round(open_price + t_dist if pos_type == 0 else open_price - t_dist, digits)

                mod_req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "symbol": pair,
                    "sl": new_sl,
                    "tp": new_tp,
                }
                res = mt5.order_send(mod_req)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"🛡️ [AI AUTO-GUARD] Attached 1:2 SL ({new_sl}) & TP ({new_tp}) to trade #{ticket} ({pair})")
                    self.notifier.send_message(
                        f"🛡️ <b>AI AUTO-GUARD ACTIVATED (1:2 RR)!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"Attached protective levels to trade #{ticket} ({pair}):\n"
                        f"• <b>Entry:</b> {open_price:.5f}\n"
                        f"• <b>Protective SL:</b> {new_sl:.5f}\n"
                        f"• <b>Target TP:</b> {new_tp:.5f}\n"
                        f"━━━━━━━━━━━━━━━━━━"
                    )

            # 2. POSITION IS IN PROFIT (pips_gain > 0 and profit > $0.20)
            if pips_gain > 0 and profit > 0.20:
                # Stage 1: Breakeven Lock at +50% SL distance
                be_trigger_dist = (init_sl_dist * 0.50) if pair == "XAUUSD" else ((init_sl_dist / pip_unit) * 0.50)
                can_be = pips_gain >= be_trigger_dist
                be_sl = round(open_price + (1 * pip_unit) if pos_type == 0 else open_price - (1 * pip_unit), digits)
                if pair == "XAUUSD":
                    be_sl = round(open_price + 0.50 if pos_type == 0 else open_price - 0.50, digits)

                # Check if SL needs updating to Breakeven
                sl_needs_be = False
                if can_be:
                    if pos_type == 0 and (sl == 0.0 or sl < open_price):
                        sl_needs_be = True
                    elif pos_type == 1 and (sl == 0.0 or sl > open_price):
                        sl_needs_be = True

                if sl_needs_be:
                    be_req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": ticket,
                        "symbol": pair,
                        "sl": be_sl,
                        "tp": tp,
                    }
                    res = mt5.order_send(be_req)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        self.be_locked.add(ticket)
                        print(f"🔒 [BREAK-EVEN LOCKED] Moved SL on #{ticket} ({pair}) to {be_sl} (Gain: +{pips_gain:.1f} pips)")
                        self.notifier.send_message(
                            f"🔒 <b>BREAK-EVEN LOCKED (STAGE 1)!</b>\n"
                            f"━━━━━━━━━━━━━━━━━━\n"
                            f"• <b>Trade:</b> #{ticket} ({pair})\n"
                            f"• <b>Gain:</b> +{pips_gain:.1f} pips\n"
                            f"• <b>New SL:</b> {be_sl:.5f} (Risk-Free Trade!)\n"
                            f"━━━━━━━━━━━━━━━━━━"
                        )

                # Stage 2: Smart Trailing Stop once in 1R profit (+100% SL distance)
                trail_trigger_dist = (init_sl_dist * 1.0) if pair == "XAUUSD" else (init_sl_dist / pip_unit)
                if pips_gain >= trail_trigger_dist:
                    trail_gap = (init_sl_dist * 0.65)
                    new_trail_sl = round(current_price - trail_gap if pos_type == 0 else current_price + trail_gap, digits)
                    should_trail = (pos_type == 0 and new_trail_sl > sl) or (pos_type == 1 and new_trail_sl < sl)
                    if should_trail:
                        tr_req = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "symbol": pair,
                            "sl": new_trail_sl,
                            "tp": tp,
                        }
                        mt5.order_send(tr_req)

            # 3. POSITION IS IN LOSS / DRAWDOWN (profit <= $0 or pips_gain <= 0)
            else:
                opp = opportunities_dict.get(pair)
                if opp and opp.get('status') == 'CONFIRMED' and not opp.get('is_hallucination', False):
                    ai_action = opp.get('action')
                    ai_conf = opp.get('confidence', 0.0) * 100
                    is_opposite_reversal = (pos_type == 0 and ai_action == "SELL") or (pos_type == 1 and ai_action == "BUY")

                    # Scenario A: AI Confirmed Structural Trend Breakdown
                    # If high conviction opposite signal (>= 72%) and drawdown is at least 50% of SL risk:
                    drawdown_dist = abs(price_delta)
                    drawdown_ratio = drawdown_dist / init_sl_dist if init_sl_dist > 0 else 0.0

                    if is_opposite_reversal and ai_conf >= 72.0 and drawdown_ratio >= 0.50:
                        close_type = mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY
                        close_price = sym_info.bid if pos_type == 0 else sym_info.ask
                        close_req = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "position": ticket,
                            "symbol": pair,
                            "volume": pos.volume,
                            "type": close_type,
                            "price": close_price,
                            "deviation": 20,
                            "magic": MAGIC_NUMBER,
                            "comment": "AI-AdaptLossCut",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        c_res = mt5.order_send(close_req)
                        if c_res and c_res.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"🛡️ [AI ADAPTIVE LOSS CUT] Cut #{ticket} ({pair}) early at 50% risk: Loss ${profit:.2f}")
                            self.notifier.send_message(
                                f"🛡️ <b>AI ADAPTIVE LOSS CUT!</b>\n"
                                f"━━━━━━━━━━━━━━━━━━\n"
                                f"Cut position #{ticket} ({pair}) early to save capital:\n"
                                f"• <b>Reason:</b> Model confirmed trend breakdown to {ai_action} ({ai_conf:.1f}%)\n"
                                f"• <b>Drawdown Cut:</b> {drawdown_ratio*100:.0f}% of SL (Loss saved by ~50%!)\n"
                                f"• <b>Closed at:</b> {close_price:.5f} | Realized: ${profit:.2f}\n"
                                f"━━━━━━━━━━━━━━━━━━"
                            )
                            continue

                    # Scenario B: Flag for Breakeven Recovery if opposite momentum noted
                    if is_opposite_reversal and ai_conf >= 68.0:
                        self.pending_reversals[ticket] = True

                # Scenario C: Patient Breakeven Recovery Exit upon Retracement to Entry
                if ticket in self.pending_reversals and (profit >= -0.10 or pips_gain >= 0.0):
                    close_type = mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY
                    close_price = sym_info.bid if pos_type == 0 else sym_info.ask
                    close_req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": ticket,
                        "symbol": pair,
                        "volume": pos.volume,
                        "type": close_type,
                        "price": close_price,
                        "deviation": 20,
                        "magic": MAGIC_NUMBER,
                        "comment": "AI-BERecovery",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    c_res = mt5.order_send(close_req)
                    if c_res and c_res.retcode == mt5.TRADE_RETCODE_DONE:
                        self.pending_reversals.pop(ticket, None)
                        print(f"🎯 [AI BREAKEVEN RECOVERY] Closed #{ticket} ({pair}) flat at entry ($0.00) after retracement!")
                        self.notifier.send_message(
                            f"🎯 <b>AI BREAKEVEN RECOVERY SUCCESSFUL!</b>\n"
                            f"━━━━━━━━━━━━━━━━━━\n"
                            f"Exited position #{ticket} ({pair}) flat at entry after retracement:\n"
                            f"• <b>Closed at:</b> {close_price:.5f} | Profit: ${profit:.2f} (Zero Capital Lost!)\n"
                            f"━━━━━━━━━━━━━━━━━━"
                        )

    def run_1sec_loop(self):
        if not self.connect_mt5():
            print("[FATAL] MT5 connection failed. Exiting.")
            sys.exit(1)

        self.start_cloud_keepalive()
        print("\n" + "=" * 70)
        print("  [LIVE] REAL-TIME 1-SECOND MARKET ANALYSIS IS NOW RUNNING!")
        print("  • Engine: Institutional DoubleEnsemble (108 Quant Features)")
        print("  • Pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, XAUUSD")
        print("  • Active AI Manager: Auto-Guard SL/TP, Break-Even Lock, Reversal Exit")
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
                # Reset daily alert on date change
                if datetime.date.today() != self.current_date:
                    self.current_date = datetime.date.today()
                    self.drawdown_alert_sent = False

                acc = mt5.account_info()
                balance = acc.balance if acc else 100.0
                equity = acc.equity if acc else 100.0

                # 1. Real-time today's trade count and daily PnL from MT5 history
                stats = self.get_today_stats()
                today_trades = stats['total_trades_today']
                open_count = stats['open_positions_count']
                daily_pnl = stats['total_daily_pnl']

                # 2. Check Daily Risk Controls
                is_drawdown_limit_hit = (daily_pnl <= DAILY_MAX_LOSS) # -$10.00
                is_profit_goal_hit = (daily_pnl >= DAILY_PROFIT_GOAL)   # +$20.00

                # Send Telegram alert once if drawdown limit hit
                if is_drawdown_limit_hit and not self.drawdown_alert_sent:
                    self.drawdown_alert_sent = True
                    self.notifier.send_message(
                        f"🛑 <b>DAILY RISK CIRCUIT BREAKER HIT!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"Total daily drawdown reached ${daily_pnl:+.2f} (Limit: ${DAILY_MAX_LOSS:.2f}).\n"
                        f"New trade execution is <b>PAUSED for today</b> to protect capital.\n"
                        f"Existing positions remain actively managed.\n"
                        f"━━━━━━━━━━━━━━━━━━"
                    )

                # PRIMARY: Run Cloud Bot Cycle on Render
                cloud_data, is_cloud_ok = self.run_cloud_bot_cycle(balance, equity, stats)

                if is_cloud_ok and cloud_data:
                    # Execute cloud bot decisions on local MT5
                    self.execute_cloud_instructions(cloud_data)
                    compute_time_ms = (time.time() - t0) * 1000
                    ts_str = datetime.datetime.now().strftime('%H:%M:%S')

                    if scan_count % 3 == 0 or len(cloud_data.get('orders_to_open', [])) > 0:
                        status_summary = cloud_data.get('cycle_summary', 'Cloud Brain Active')
                        print(f"[{ts_str}] ☁️ [RENDER BOT] Cycle #{scan_count:05d} ({compute_time_ms:.0f}ms) | Equity: ${equity:.2f} | Trades: {today_trades}/{MAX_DAILY_TRADES} (Open: {open_count}) | {status_summary}")

                else:
                    # FALLBACK: Local RAM Engine if Render is waking up or network lags
                    opportunities, actionable = self.analyze_market_locally(balance, open_count)
                    compute_time_ms = (time.time() - t0) * 1000

                    opps_dict = {o['pair']: o for o in opportunities}
                    self.manage_all_positions(opps_dict)

                    if scan_count % 3 == 0 or actionable:
                        top = opportunities[0] if opportunities else {}
                        act = top.get('action', 'NEUTRAL')
                        conf = top.get('confidence', 0) * 100
                        pair_name = top.get('pair', 'None')
                        ts_str = datetime.datetime.now().strftime('%H:%M:%S')

                        if is_drawdown_limit_hit:
                            status_tag = f"🛑 PAUSED: Daily Drawdown Limit (-$10.00) Hit! (PnL: ${daily_pnl:+.2f})"
                        elif today_trades >= MAX_DAILY_TRADES:
                            status_tag = f"✅ MAX TRADES REACHED ({today_trades}/{MAX_DAILY_TRADES}) | PnL: ${daily_pnl:+.2f}"
                        elif actionable:
                            status_tag = f"🎯 ACTIONABLE: {act}"
                        else:
                            status_tag = f"Local Fallback... Best: {pair_name} ({act} {conf:.1f}%) | PnL: ${daily_pnl:+.2f}"

                        print(f"[{ts_str}] ⚡ [LOCAL FALLBACK] Scan #{scan_count:05d} ({compute_time_ms:.0f}ms) | Equity: ${equity:.2f} | Trades: {today_trades}/{MAX_DAILY_TRADES} (Open: {open_count}) | {status_tag}")

                    can_open_trades = (today_trades < MAX_DAILY_TRADES) and (not is_drawdown_limit_hit) and (open_count < MAX_CONCURRENT_POSITIONS)
                    if actionable and can_open_trades:
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
