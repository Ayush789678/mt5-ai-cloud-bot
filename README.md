# 🤖 24/7 AI-Powered Serverless MT5 Trading Bot

Fully automated, 100% free cloud-hosted algorithmic trading engine powered by the **Golden DoubleEnsemble AI Model** (500+700 XGBoost trees + 700 CatBoost trees) running on **GitHub Actions Windows Runners**.

---

## 🚀 Key Features

* **100% Free Cloud Hosting:** Runs on Microsoft Azure Windows runners via GitHub Actions. **No credit cards, no VPS subscriptions, and no MetaAPI payments required.**
* **Laptop Can Be Closed:** Trades trigger in GitHub's cloud 24/7 without your computer being open.
* **Model-Driven Dynamic Position Sizing:**
  * Ultra-High Conviction ($p \ge 62\%$): **0.03 Lots** (+~$6.00 target)
  * Standard Conviction ($p \ge 56\%$): **0.02 Lots** (+~$4.00 target)
  * Defensive Conviction: **0.01 Lots** (+~$2.00 target, -$1.25 max loss)
* **Capital & Daily Goals:** Configured for **$100.00 capital**, targeting **$10.00 to $20.00 / day** net profit.
* **Hardcoded Risk Controls:** Automated Stop-Loss (12.5 pips) and Take-Profit (20.0 pips) are executed natively by the broker datacenter.
* **Real-Time Telegram Alerts:** Dispatched via `@tradeinforbot` with exact probabilities, conviction score, lot size, and execution price.

---

## ⚡ How It Works

1. **Scheduled Cloud Execution:** The workflow runs automatically every 30 minutes during Forex trading sessions (Monday through Friday).
2. **Manual 1-Click Trigger:** You can manually trigger a market scan anytime by going to **Actions → 24/7 AI MT5 Trade Scanner → Run workflow**.
3. **Institutional Precision:** The bot pulls real-time tick and candle data for `EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`, `USDCAD`, `USDCHF`, and `NZDUSD`, computes 108 quantitative indicators, runs model inference, vetoes hallucinations, and places qualifying orders directly into MT5.
