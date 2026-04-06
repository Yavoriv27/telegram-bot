# -*- coding: utf-8 -*-

import os
from datetime import datetime
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

import oandapyV20
from oandapyV20.endpoints import instruments

import yfinance as yf

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OANDA_KEY = os.getenv("OANDA_API_KEY")

client = oandapyV20.API(access_token=OANDA_KEY)

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]
CHAT_IDS = set()
AUTO = True

user_data = {}

# ===== REAL VOLUME (FUTURES) =====
def get_real_volume():
    try:
        data = yf.download("6E=F", interval="1m", period="1d")
        vols = data["Volume"].tail(10).tolist()

        if len(vols) < 5:
            return False

        avg = sum(vols[:-1]) / (len(vols) - 1)
        return vols[-1] > avg * 1.5
    except:
        return False

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
                "l": float(c["mid"]["l"])
            })
    return candles

# ================= CORE =================

def trend(data):
    return "UP" if data[-1]["c"] > data[-10]["c"] else "DOWN"

def volatility(c):
    return sum(abs(x["h"] - x["l"]) for x in c[-10:]) / 10

def is_news_time():
    now = datetime.utcnow()
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

def predict_move(c):
    power = 0
    for x in c[-5:]:
        if x["c"] > x["o"]:
            power += abs(x["c"] - x["o"])
        else:
            power -= abs(x["c"] - x["o"])

    if power > 0.001:
        return "BUY"
    if power < -0.001:
        return "SELL"
    return "NEUTRAL"

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
    real_vol = get_real_volume()

    if sweep == direction:
        score += 35
        reasons.append("Liquidity")

    if fake == direction:
        score += 20
        reasons.append("Fake")

    if real_vol:
        score += 25
        reasons.append("REAL VOLUME 🔥")

    pred = predict_move(c1)
    if pred == direction:
        score += 15
        reasons.append("Momentum")

    entry = best_entry(c1)
    if entry == "enter":
        score += 10
    else:
        score -= 10

    if score < 70:
        return None

    return {
        "pair": pair,
        "dir": direction,
        "score": score,
        "prob": min(score, 95),
        "reasons": ", ".join(reasons)
    }

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
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")]
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    CHAT_IDS.add(chat_id)

    await update.message.reply_text(
        "🔥 FINAL BOSS REAL VOLUME",
        reply_markup=keyboard()
    )

async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s = get_signal()

        if not s:
            await q.edit_message_text("❌ Нема сигналу", reply_markup=keyboard())
            return

        msg = f"""
🔥 FINAL BOSS

📊 {s['pair']}
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

📊 {s['prob']}%
📈 {s['score']}

🧠 {s['reasons']}
"""

        await q.edit_message_text(msg, reply_markup=keyboard())

    elif q.data == "auto":
        AUTO = not AUTO
        await q.edit_message_text(f"🤖 AUTO: {AUTO}", reply_markup=keyboard())

# ================= AUTO =================

async def auto_signal(context: ContextTypes.DEFAULT_TYPE):
    if not AUTO:
        return

    s = get_signal()
    if not s:
        return

    msg = f"""
🚀 AUTO SIGNAL

📊 {s['pair']}
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}
📊 {s['prob']}%
"""

    for chat_id in CHAT_IDS:
        try:
            await context.bot.send_message(chat_id, msg)
        except:
            pass

# ================= MAIN =================

def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(buttons))

    app.job_queue.run_repeating(auto_signal, interval=300, first=10)

    print("🔥 FINAL BOSS REAL VOLUME RUNNING")
    app.run_polling()

if __name__ == "__main__":
    main()
