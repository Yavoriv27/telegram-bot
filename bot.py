import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import requests

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

import oandapyV20
from oandapyV20.endpoints import instruments

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
OANDA_KEY = os.getenv("OANDA_API_KEY")
NEWS_KEY = os.getenv("NEWS_API_KEY")

client = oandapyV20.API(access_token=OANDA_KEY)

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]
AUTO = False

# ================= NEWS =================
def check_news():
    try:
        url = f"https://newsapi.org/v2/everything?q=forex OR usd OR eur&apiKey={NEWS_KEY}"
        r = requests.get(url, timeout=5).json()

        for a in r.get("articles", [])[:5]:
            t = a["title"].lower()
            if any(x in t for x in ["fed", "cpi", "rate", "inflation", "nfp", "ecb"]):
                return True
    except:
        pass
    return False

# ================= SESSION =================
def is_market_time():
    h = datetime.utcnow().hour
    return (8 <= h <= 11) or (13 <= h <= 17)

# ================= DATA =================
def get_candles(pair, tf="M1", count=50):
    params = {"granularity": tf, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)

    candles = []
    for c in r.response["candles"]:
        if c["complete"]:
            candles.append({
                "open": float(c["mid"]["o"]),
                "close": float(c["mid"]["c"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"])
            })
    return candles

# ================= STRATEGY =================
def analyze(pair):
    m5 = get_candles(pair, "M5", 30)
    m1 = get_candles(pair, "M1", 30)

    if len(m5) < 10 or len(m1) < 10:
        return None

    closes_m5 = [c["close"] for c in m5]
    closes_m1 = [c["close"] for c in m1]

    # TREND
    ema_fast = sum(closes_m5[-5:]) / 5
    ema_slow = sum(closes_m5[-20:]) / 20

    if abs(ema_fast - ema_slow) < 0.00005:
        return None

    trend = "BUY" if ema_fast > ema_slow else "SELL"

    # LEVEL
    support = min(closes_m1[-20:])
    resistance = max(closes_m1[-20:])
    price = closes_m1[-1]

    near_support = abs(price - support) < (resistance - support) * 0.25
    near_resistance = abs(price - resistance) < (resistance - support) * 0.25

    # PULLBACK
    last3 = closes_m1[-3:]
    pullback_buy = last3[0] > last3[1] > last3[2]
    pullback_sell = last3[0] < last3[1] < last3[2]

    # TRIGGER
    last2 = m1[-2:]
    bullish = last2[0]["close"] > last2[0]["open"] and last2[1]["close"] > last2[1]["open"]
    bearish = last2[0]["close"] < last2[0]["open"] and last2[1]["close"] < last2[1]["open"]

    score = 0

    if trend == "BUY":
        score += 2
    else:
        score += 2

    if near_support or near_resistance:
        score += 2

    if bullish or bearish:
        score += 2

    if trend == "BUY" and near_support and pullback_buy and bullish:
        return {"pair": pair, "direction": "BUY", "score": score,
                "reason": "Trend ↑ + Pullback ↓ + Support + Trigger"}

    if trend == "SELL" and near_resistance and pullback_sell and bearish:
        return {"pair": pair, "direction": "SELL", "score": score,
                "reason": "Trend ↓ + Pullback ↑ + Resistance + Trigger"}

    return None

# ================= SIGNAL =================
def generate_signal():
    if check_news():
        return None, "📰 News filter"

    if not is_market_time():
        return None, "⏰ Not active session"

    best = None

    for pair in PAIRS:
        s = analyze(pair)
        if s:
            if not best or s["score"] > best["score"]:
                best = s

    if not best:
        return None, "❌ No setup"

    return best, None

# ================= UI =================
def main_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")]
    ])

# ================= HANDLERS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 MAX TRADER BOT", reply_markup=main_kb())

async def btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s, reason = generate_signal()

        if not s:
            await q.message.reply_text(reason, reply_markup=main_kb())
            return

        sec = datetime.utcnow().second
        entry_in = 60 - sec

        msg = f"""
📊 {s['pair']}
{'🟢 BUY' if s['direction']=='BUY' else '🔴 SELL'}

📌 {s['reason']}

⏱ Вхід через: {entry_in} сек
🕒 {datetime.utcnow().strftime('%H:%M:%S')}
"""
        await q.message.reply_text(msg, reply_markup=main_kb())

    elif q.data == "auto":
        AUTO = not AUTO
        await q.message.reply_text(f"AUTO: {AUTO}", reply_markup=main_kb())

# ================= AUTO =================
async def auto_job(context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    if not AUTO:
        return

    s, _ = generate_signal()

    if s:
        await context.bot.send_message(
            chat_id=CHAT_ID,
            text=f"{s['pair']} {s['direction']} | {s['reason']}"
        )

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(btn))

    app.job_queue.run_repeating(auto_job, interval=120, first=10)

    print("🚀 MAX TRADER BOT STARTED")
    app.run_polling()

if __name__ == "__main__":
    main()
