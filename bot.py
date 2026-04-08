# -*- coding: utf-8 -*-

import os
import asyncio
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

PAIR = "EUR_USD"
CHAT_IDS = set()
AUTO = True

LAST_DIRECTION = None
LAST_SIGNAL_TIME = {}

STATS = {"win": 0, "loss": 0}
HISTORY = []

COOLDOWN = 600


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


# ================= INDICATORS =================

def ema(c, period):
    k = 2 / (period + 1)
    ema_val = c[0]["c"]
    for x in c:
        ema_val = x["c"] * k + ema_val * (1 - k)
    return ema_val


def atr(c):
    trs = []
    for i in range(1, len(c)):
        tr = max(
            c[i]["h"] - c[i]["l"],
            abs(c[i]["h"] - c[i-1]["c"]),
            abs(c[i]["l"] - c[i-1]["c"])
        )
        trs.append(tr)
    return sum(trs[-14:]) / 14 if len(trs) >= 14 else 0


def candle_strength(c):
    last = c[-1]
    body = abs(last["c"] - last["o"])
    full = last["h"] - last["l"]
    return body / full if full > 0 else 0


def candle_position(c):
    last = c[-1]
    full = last["h"] - last["l"]
    return (last["c"] - last["l"]) / full if full > 0 else 0.5


def support_resistance(c):
    highs = [x["h"] for x in c[-20:]]
    lows = [x["l"] for x in c[-20:]]
    return max(highs), min(lows)


def structure(c):
    highs = [x["h"] for x in c[-5:]]
    lows = [x["l"] for x in c[-5:]]

    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return "UP"
    if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return "DOWN"
    return "RANGE"


# ================= CORE =================

def analyze(pair):
    global LAST_DIRECTION

    c1 = get_candles(pair, "M1")
    c5 = get_candles(pair, "M5")
    c15 = get_candles(pair, "M15")

    if not c1 or not c5 or not c15:
        return None

    # === TREND
    ema20 = ema(c15[-50:], 20)
    ema50 = ema(c15[-50:], 50)

    if ema20 > ema50:
        direction = "BUY"
    elif ema20 < ema50:
        direction = "SELL"
    else:
        return None

    # === STRUCTURE
    if structure(c15) != ("UP" if direction == "BUY" else "DOWN"):
        return None

    # === VOLATILITY
    if atr(c1) < 0.00015:
        return None

    # === OVERHEAT FILTER
    last5 = c1[-5:]
    if all(x["c"] > x["o"] for x in last5) or all(x["c"] < x["o"] for x in last5):
        return None

    # === LEVELS
    resistance, support = support_resistance(c15)
    price = c1[-1]["c"]

    if abs(price - resistance) < 0.00025 or abs(price - support) < 0.00025:
        return None

    # === ANTI DUPLICATE
    if LAST_DIRECTION == direction:
        return None

    # === PULLBACK ENTRY
    prev = c1[-2]
    last = c1[-1]

    if direction == "BUY":
        if not (prev["c"] < prev["o"] and last["c"] > last["o"]):
            return None
    if direction == "SELL":
        if not (prev["c"] > prev["o"] and last["c"] < last["o"]):
            return None

    # === CANDLE QUALITY
    strength = candle_strength(c1)
    pos = candle_position(c1)

    if direction == "BUY":
        if strength < 0.6 or pos < 0.7:
            return None
    else:
        if strength < 0.6 or pos > 0.3:
            return None

    # === M5 CONFIRM
    m5 = c5[-1]
    if ("BUY" if m5["c"] > m5["o"] else "SELL") != direction:
        return None

    LAST_DIRECTION = direction

    return {
        "dir": direction,
        "level": "PRO",
        "prob": 85,
        "bet": "10%" if len(HISTORY) < 3 or HISTORY[-1] == "win" else "5%"
    }


# ================= UI =================

def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")],
        [
            InlineKeyboardButton("✅", callback_data="win"),
            InlineKeyboardButton("❌", callback_data="loss")
        ],
        [InlineKeyboardButton("📊 Статистика", callback_data="stats")]
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("🔥 AI CORE v5 SYSTEM", reply_markup=keyboard())


async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s = analyze(PAIR)

        if not s:
            await q.edit_message_text("❌ Нема сигналу", reply_markup=keyboard())
            return

        msg = f"""
🚀 PRO SIGNAL

📊 EUR/USD
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

⚡ {s['level']}
📊 {s['prob']}%
💰 Ставка: {s['bet']}
"""
        await q.edit_message_text(msg, reply_markup=keyboard())

    elif q.data == "auto":
        AUTO = not AUTO
        await q.edit_message_text(f"🤖 AUTO: {AUTO}", reply_markup=keyboard())

    elif q.data == "win":
        STATS["win"] += 1
        HISTORY.append("win")
        await q.answer("+")

    elif q.data == "loss":
        STATS["loss"] += 1
        HISTORY.append("loss")
        await q.answer("-")

    elif q.data == "stats":
        total = STATS["win"] + STATS["loss"]
        wr = (STATS["win"]/total*100) if total else 0

        msg = f"""
📊 СТАТИСТИКА

Угод: {total}
✅ {STATS['win']}
❌ {STATS['loss']}
📈 Winrate: {round(wr,1)}%
"""
        await q.edit_message_text(msg, reply_markup=keyboard())


# ================= AUTO =================

async def loop(app):
    while True:
        if AUTO:
            s = analyze(PAIR)
            if s:
                msg = f"""
🚀 PRO SIGNAL

📊 EUR/USD
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

⚡ {s['level']}
📊 {s['prob']}%
💰 {s['bet']}
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
        asyncio.get_event_loop().create_task(loop(app))

    app.post_init = post_init

    print("🔥 V5 SYSTEM RUNNING")

    app.run_polling()


if __name__ == "__main__":
    main()
