import MetaTrader5 as mt5
import requests
import time
from mt5_local_connector import LocalMT5Connector, PAIRS, TIMEFRAME

c = LocalMT5Connector('https://mt5-ai-model-service.onrender.com')
mt5.initialize()

print('=' * 75)
print('               LIVE AI MODEL MARKET DECISION AUDIT                     ')
print('=' * 75)

market_data = {}
for p in PAIRS:
    candles = c.fetch_pair_candles(p, count=150)
    if candles:
        market_data[p] = candles

payload = {'timeframe': TIMEFRAME, 'market_data': market_data, 'account_balance': 97.25}
t0 = time.time()
res = requests.post('https://mt5-ai-model-service.onrender.com/scan', json=payload, timeout=25)
latency = round((time.time() - t0) * 1000, 1)

data = res.json()
print(f"Scan Completed in {latency}ms across {data.get('scanned_pairs')} pairs")
print('-' * 75)
print(f"{'PAIR':<10} | {'TF':<4} | {'DECISION':<8} | {'CONFIDENCE':<10} | {'STATUS':<15} | {'LOT':<5}")
print('-' * 75)

for opp in data.get('top_opportunities', []):
    pair = opp.get('pair', '')
    tf = opp.get('timeframe', '15M')
    action = opp.get('action', 'NEUTRAL')
    conf = f"{opp.get('confidence', 0)*100:.1f}%"
    status = opp.get('status', 'NEUTRAL')
    lot = opp.get('recommended_lot', 0.01)
    print(f"{pair:<10} | {tf:<4} | {action:<8} | {conf:<10} | {status:<15} | {lot:<5}")

print('=' * 75)
acc = mt5.account_info()
pos = mt5.positions_get()
print(f"Current MT5 Balance: ${acc.balance:.2f} | Open Market Positions: {len(pos) if pos else 0}")
print('=' * 75)
