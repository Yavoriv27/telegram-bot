# -*- coding: utf-8 -*-

import os
from datetime import datetime
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

import oandapyV20
from oandapyV20.endpoints import instruments

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OANDA_KEY = os.getenv("OANDA_API_KEY")

client = oandapyV20.API(access_token=OANDA_KEY)

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]
CHAT_IDS = set()

# ================= DATA =================
def get_candles(pair, tf, count=120):
    params = {"granularity": tf, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)

    candles = []

    for c in r.response["candles"]:
        if c["complete"]:
            candles.append({
                "o": float(c["mid"]["o"]),
                "c": float(c["mid"]["c"]),
                "h": float(c["mid"]["h"]),
                "l": float(c["mid"]["l"]),
                "v": c.get("volume", 1)
            })

    return candles

# ================= CORE =================

def trend(data):
    return "UP" if data[-1]["c"] > data[-10]["c"] else "DOWN"

def volatility(c):
    return sum(abs(x["h"] - x["l"]) for x in c[-10:]) / 10

def is_news_time():
    now = datetime.now()
    minutes = now.hour * 60 + now.minute
    news = [570, 930, 1020]
    return any(abs(minutes - t) < 30 for t in news)

def too_clean_trend(c):
    last5 = c[-5:]
    return all(x["c"] > x["o"] for x in last5) or all(x["c"] < x["o"] for x in last5)

def liquidity_sweep(c):
    last = c[-1]
    prev_high = max(x["h"] for x in c[-10:-1])
    prev_low = min(x["l"] for x in c[-10:-1])

    if last["l"] < prev_low and last["c"] > prev_low:
        return "BUY"
    if last["h"] > prev_high and last["c"] < prev_high:
        return "SELL"
    return None

def fake_breakout(c):
    last = c[-1]
    prev = c[-2]

    if last["h"] > prev["h"] and last["c"] < prev["h"]:
        return "SELL"
    if last["l"] < prev["l"] and last["c"] > prev["l"]:
        return "BUY"
    return None

def volume_spike(c):
    vols = [x["v"] for x in c[-10:]]
    avg = sum(vols[:-1]) / (len(vols) - 1)
    return vols[-1] > avg * 1.5

def predict_move(c):
    power = 0
    for x in c[-5:]:
        if x["c"] > x["o"]:
            power += abs(x["c"] - x["o"])
        else:
            power -= abs(x["c"] - x["o"])

    if power > 0.001:
        return "BUY", power
    if power < -0.001:
        return "SELL", power
    return "NEUTRAL", power

def best_entry(c):
    last = c[-1]
    body = abs(last["c"] - last["o"])
    full = last["h"] - last["l"]

    if full == 0:
        return "wait"

    ratio = body / full

    if ratio > 0.7:
        return "wait"

    if 0.3 < ratio < 0.7:
        return "enter"

    return "wait"

# ================= FINAL BOSS =================

def analyze(pair):
    c1 = get_candles(pair, "M1")
    c15 = get_candles(pair, "M15")

    if is_news_time():
        return None

    if volatility(c1) < 0.0004:
        return None

    if too_clean_trend(c1):
        return None

    direction = "BUY" if trend(c15) == "UP" else "SELL"

    score = 0
    reasons = []

    sweep = liquidity_sweep(c1)
    fake = fake_breakout(c1)
    vol = volume_spike(c1)

    if sweep == direction:
        score += 30
        reasons.append("Liquidity")

    if fake == direction:
        score += 20
        reasons.append("Fake")

    if vol:
        score += 15
        reasons.append("Volume")

    pred, _ = predict_move(c1)
    if pred == direction:
        score += 15
        reasons.append("Momentum")

    entry = best_entry(c1)
    if entry == "enter":
        score += 10
    else:
        score -= 10

    if score < 65:
        return None

    return {
        "pair": pair,
        "dir": direction,
        "score": score,
        "prob": min(score, 95),
        "entry": entry,
        "reasons": ", ".join(reasons)
    }

# ================= SIGNAL =================

def get_signal():
    best = None

    for pair in PAIRS:
        try:
            s = analyze(pair)
        except:
            continue

        if s:
            if not best or s["prob"] > best["prob"]:
                best = s

    return best

# ================= UI =================

def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")]
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)

    await update.message.reply_text(
        "🔥 FINAL BOSS BOT\nГотовий до роботи",
        reply_markup=keyboard()
    )

async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s = get_signal()

        if not s:
            await q.edit_message_text("❌ Нема сильного сигналу", reply_markup=keyboard())
            return

        msg = f"""
🔥 FINAL BOSS

📊 {s['pair']}
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

📊 Ймовірність: {s['prob']}%
📈 Score: {s['score']}

🧠 {s['reasons']}

{'🔔 Входити зараз' if s['prob']>80 else '⏳ Краще зачекати'}
"""

        await q.edit_message_text(msg, reply_markup=keyboard())

# ================= MAIN =================

def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(buttons))

    print("🔥 FINAL BOSS STARTED")
    app.run_polling()

if __name__ == "__main__":
    main()
