"""
Telegram Notification Engine for 24/7 Cloud MT5 Trading Bot
Sends instant real-time alerts to your personal Telegram for:
- Trade Entries with Dynamic Lot Sizing, TP, SL, and Confidence
- Trade Exits (Take-Profit or Stop-Loss hit)
- Daily Profit / Consistency Milestone Updates
"""

import requests
import json
import logging
from datetime import datetime

logger = logging.getLogger("TelegramAlerts")

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        self.enabled = bool(bot_token and chat_id and bot_token != "YOUR_BOT_TOKEN")
        if not self.enabled:
            logger.warning("Telegram Bot is unconfigured. Alerts will only be logged locally.")

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            print(f"[TELEGRAM LOG ONLY]:\n{text}\n")
            return False
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            res = requests.post(self.base_url, json=payload, timeout=8)
            return res.status_code == 200
        except Exception as e:
            logger.error(f"Failed to dispatch Telegram alert: {e}")
            return False

    def notify_trade_opened(self, pair: str, action: str, tf: str, price: float,
                            lot_size: float, tp: float, sl: float, p_win: float,
                            conviction: float, est_win: float, balance: float):
        action_emoji = "🟢 <b>BUY</b>" if action == "BUY" else "🔴 <b>SELL</b>"
        p_win_pct = p_win * 100
        
        msg = (
            f"🚀 <b>NEW LIVE TRADE EXECUTED (MT5)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Pair:</b> <code>{pair}</code> ({tf})\n"
            f"<b>Action:</b> {action_emoji} @ <code>{price:.5f}</code>\n"
            f"<b>Model Win Probability:</b> <code>{p_win_pct:.1f}%</code>\n"
            f"<b>Conviction Score:</b> <code>{conviction:.3f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ <b>Auto Lot Size:</b> <code>{lot_size:.2f} Lots</code>\n"
            f"🎯 <b>Take Profit (TP):</b> <code>{tp:.5f}</code> (+${est_win:.2f})\n"
            f"🛑 <b>Stop Loss (SL):</b> <code>{sl:.5f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💼 <b>Account Equity:</b> <code>${balance:.2f}</code>\n"
            f"⏰ <i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
        return self.send_message(msg)

    def notify_trade_closed(self, pair: str, action: str, pnl_dollars: float,
                            pips: float, new_balance: float, reason: str = "TP Hit"):
        win = pnl_dollars > 0
        status_emoji = "✅ <b>PROFIT TARGET HIT!</b>" if win else "❌ <b>STOP LOSS HIT</b>"
        pnl_str = f"+${pnl_dollars:.2f}" if win else f"-${abs(pnl_dollars):.2f}"
        
        msg = (
            f"{status_emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Pair:</b> <code>{pair}</code> ({action})\n"
            f"<b>Realized PnL:</b> <code>{pnl_str}</code> ({pips:+.1f} pips)\n"
            f"<b>Close Reason:</b> <i>{reason}</i>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💼 <b>Updated Account Balance:</b> <code>${new_balance:.2f}</code>\n"
            f"⏰ <i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
        return self.send_message(msg)

    def notify_daily_summary(self, date_str: str, trades: int, wins: int,
                             losses: int, net_pnl: float, balance: float):
        wr = (wins / trades * 100) if trades > 0 else 0.0
        goal_status = "🎉 <b>DAILY $10–$20 TARGET ACHIEVED!</b>" if net_pnl >= 10.0 else "📊 <b>DAILY TRADING SUMMARY</b>"
        
        msg = (
            f"{goal_status}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Date:</b> {date_str}\n"
            f"<b>Trades Executed:</b> {trades}\n"
            f"<b>Wins / Losses:</b> {wins}W / {losses}L\n"
            f"<b>Win Rate:</b> <code>{wr:.1f}%</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Net Daily PnL:</b> <code>${net_pnl:+.2f}</code>\n"
            f"💼 <b>Closing Balance:</b> <code>${balance:.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )
        return self.send_message(msg)
