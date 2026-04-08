# -*- coding: utf-8 -*-

import os
import asyncio
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

PAIRS = ["EUR_USD"]
CHAT_IDS = set()
AUTO = True

LAST_SIGNAL_TIME = {}
COOLDOWN = 120

# ================= DATA =================

def get_candles(pair, tf, count=120):
    try:
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
    except:
        return []

# ================= AI CORE =================

def strong_trend(c):
    up = sum(1 for x in c[-5:] if x["c"] > x["o"])
    down = sum(1 for x in c[-5:] if x["c"] < x["o"])

    if up >= 4:
        return "UP"
    if down >= 4:
        return "DOWN"
    return "FLAT"

def is_flat(c):
    moves = [abs(x["c"] - x["o"]) for x in c[-10:]]
    return sum(moves) / len(moves) < 0.0002

def sniper_entry(c):
    last = c[-1]
    prev = c[-2]
    prev2 = c[-3]

    if prev2["l"] < prev["l"] and prev["c"] > prev["o"]:
        if last["c"] > last["o"] and last["c"] > prev["c"]:
            return "BUY"

    if prev2["h"] > prev["h"] and prev["c"] < prev["o"]:
        if last["c"] < last["o"] and last["c"] < prev["c"]:
            return "SELL"

    return None

def candle_power(c):
    last = c[-1]
    body = abs(last["c"] - last["o"])
    full = last["h"] - last["l"]

    if full == 0:
        return None

    strength = body / full

    if last["c"] > last["o"] and strength > 0.6:
        return "BUY"
    if last["c"] < last["o"] and strength > 0.6:
        return "SELL"

    return None

def get_volume():
    try:
        data = yf.download("6E=F", interval="1m", period="1d", progress=False)
        vols = data["Volume"].tail(5).tolist()
        return vols[-1] > sum(vols[:-1]) / len(vols[:-1])
    except:
        return False

# ================= ANALYZE =================

def analyze(pair):
    c1 = get_candles(pair, "M1")
    c15 = get_candles(pair, "M15")

    if not c1 or not c15:
        return None

    if is_flat(c1):
        return None

    trend_dir = strong_trend(c15)

    if trend_dir == "FLAT":
        return None

    direction = "BUY" if trend_dir == "UP" else "SELL"

    score = 0
    reasons = []

    sniper = sniper_entry(c1)
    if sniper == direction:
        score += 30
        reasons.append("Sniper")

    cp = candle_power(c1)
    if cp == direction:
        score += 20
        reasons.append("Candle")

    if get_volume():
        score += 20
        reasons.append("Volume")

    if score < 70:
        return None

    return {
        "pair": pair,
        "dir": direction,
        "prob": min(score, 95),
        "reasons": ", ".join(reasons)
    }

# ================= UI =================

def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")]
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("🔥 FINAL AI CORE", reply_markup=keyboard())

async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s = analyze("EUR_USD")

        if not s:
            await q.edit_message_text("❌ Нема сигналу", reply_markup=keyboard())
            return

        msg = f"""
🔥 FINAL AI CORE

📊 EUR/USD
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

📊 {s['prob']}%
🧠 {s['reasons']}
"""
        await q.edit_message_text(msg, reply_markup=keyboard())

    elif q.data == "auto":
        AUTO = not AUTO
        await q.edit_message_text(f"🤖 AUTO: {AUTO}", reply_markup=keyboard())

# ================= INSTANT =================

async def instant_loop(app):
    while True:
        if AUTO:
            now = datetime.utcnow().timestamp()

            last = LAST_SIGNAL_TIME.get("EUR_USD")
            if last and now - last < COOLDOWN:
                await asyncio.sleep(10)
                continue

            s = analyze("EUR_USD")

            if s:
                LAST_SIGNAL_TIME["EUR_USD"] = now

                msg = f"""
🚀 INSTANT SIGNAL

📊 EUR/USD
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}
📊 {s['prob']}%
"""

                for chat_id in CHAT_IDS:
                    await app.bot.send_message(chat_id, msg)

        await asyncio.sleep(30)

# ================= MAIN =================

def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(buttons))

    async def post_init(app):
        await asyncio.sleep(1)
        asyncio.get_event_loop().create_task(instant_loop(app))

    app.post_init = post_init

    print("🔥 FINAL AI CORE RUNNING")

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
