# -*- coding: utf-8 -*-

import os
import asyncio
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
HISTORY = []

# ================= DATA =================

def get_candles(tf, count=120):
    try:
        params = {"granularity": tf, "count": count, "price": "M"}
        r = instruments.InstrumentsCandles(instrument=PAIR, params=params)
        client.request(r)

        return [{
            "o": float(c["mid"]["o"]),
            "c": float(c["mid"]["c"]),
            "h": float(c["mid"]["h"]),
            "l": float(c["mid"]["l"])
        } for c in r.response["candles"] if c["complete"]]
    except:
        return []

# ================= INDICATORS =================

def ema(c, p):
    k = 2 / (p + 1)
    e = c[0]["c"]
    for x in c:
        e = x["c"] * k + e * (1 - k)
    return e

def atr(c):
    trs = [abs(c[i]["h"] - c[i]["l"]) for i in range(-14, -1)]
    return sum(trs) / len(trs) if trs else 0

def strength(c):
    l = c[-1]
    return abs(l["c"] - l["o"]) / (l["h"] - l["l"]) if (l["h"] - l["l"]) else 0

def position(c):
    l = c[-1]
    return (l["c"] - l["l"]) / (l["h"] - l["l"]) if (l["h"] - l["l"]) else 0.5

def structure(c):
    if c[-1]["h"] > c[-2]["h"] and c[-1]["l"] > c[-2]["l"]:
        return "UP"
    if c[-1]["h"] < c[-2]["h"] and c[-1]["l"] < c[-2]["l"]:
        return "DOWN"
    return "RANGE"

def levels(c):
    highs = [x["h"] for x in c[-20:]]
    lows = [x["l"] for x in c[-20:]]
    return max(highs), min(lows)

# ================= AI =================

def ai_score(direction, c1, c5, c15):
    score = 0

    e20 = ema(c15[-50:], 20)
    e50 = ema(c15[-50:], 50)

    if (e20 > e50 and direction == "BUY") or (e20 < e50 and direction == "SELL"):
        score += 25

    if structure(c15) == ("UP" if direction == "BUY" else "DOWN"):
        score += 20

    if strength(c1) > 0.6:
        score += 15

    pos = position(c1)
    if (direction == "BUY" and pos > 0.7) or (direction == "SELL" and pos < 0.3):
        score += 15

    m5 = c5[-1]
    if ("BUY" if m5["c"] > m5["o"] else "SELL") == direction:
        score += 15

    last5 = c1[-5:]
    if not (all(x["c"] > x["o"] for x in last5) or all(x["c"] < x["o"] for x in last5)):
        score += 10

    if HISTORY[-3:].count("loss") >= 2:
        score -= 15

    return max(0, min(score, 100))

# ================= CORE =================

def analyze():
    global LAST_DIRECTION

    c1 = get_candles("M1")
    c5 = get_candles("M5")
    c15 = get_candles("M15")

    if not c1 or not c5 or not c15:
        return None

    direction = "BUY" if c15[-1]["c"] > c15[-1]["o"] else "SELL"

    if LAST_DIRECTION == direction:
        return None

    # флет
    if atr(c1) < 0.00012:
        return None

    # мінімальний рух
    if abs(c1[-1]["c"] - c1[-1]["o"]) < 0.00005:
        return None

    # сильна свічка
    if strength(c1) < 0.6:
        return None

    # анти кінець руху
    last3 = c1[-3:]
    if all(x["c"] > x["o"] for x in last3) or all(x["c"] < x["o"] for x in last3):
        return None

    # 🔥 ПРОБОЙ (НОВЕ)
    if direction == "BUY" and c1[-1]["c"] <= c1[-2]["h"]:
        return None
    if direction == "SELL" and c1[-1]["c"] >= c1[-2]["l"]:
        return None

    # 🔥 ПОЗИЦІЯ (НОВЕ)
    pos = position(c1)
    if direction == "BUY" and pos < 0.6:
        return None
    if direction == "SELL" and pos > 0.4:
        return None

    # рівні
    r, s = levels(c15)
    price = c1[-1]["c"]

    if abs(price - r) < 0.00025 or abs(price - s) < 0.00025:
        return None

    prev = c1[-2]
    last = c1[-1]

    if direction == "BUY" and not (prev["c"] < prev["o"] and last["c"] > last["o"]):
        return None
    if direction == "SELL" and not (prev["c"] > prev["o"] and last["c"] < last["o"]):
        return None

    score = ai_score(direction, c1, c5, c15)

    if score < 85:
        return None

    LAST_DIRECTION = direction

    return {"dir": direction, "score": score, "level": "GOOD"}

# ================= UI =================

def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")],
        [
            InlineKeyboardButton("✅", callback_data="win"),
            InlineKeyboardButton("❌", callback_data="loss")
        ]
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("🔥 V7.3 PRO BOT", reply_markup=keyboard())

async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s = analyze()

        if not s:
            try:
                await q.edit_message_text("❌ SKIP", reply_markup=keyboard())
            except:
                pass
            return

        msg = f"""
🚀 PRO SIGNAL

📊 EUR/USD
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

🧠 {s['score']}%
⚡ GOOD
"""

        try:
            await q.edit_message_text(msg, reply_markup=keyboard())
        except:
            pass

    elif q.data == "auto":
        AUTO = not AUTO
        try:
            await q.edit_message_text(f"AUTO: {AUTO}", reply_markup=keyboard())
        except:
            pass

    elif q.data == "win":
        HISTORY.append("win")
        await q.answer("+")

    elif q.data == "loss":
        HISTORY.append("loss")
        await q.answer("-")

# ================= LOOP =================

async def loop(app):
    while True:
        if AUTO:
            s = analyze()
            if s:
                msg = f"""
🚀 PRO SIGNAL

📊 EUR/USD
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

🧠 {s['score']}%
⚡ GOOD
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

    print("🔥 V7.3 RUNNING")

    app.run_polling()

if __name__ == "__main__":
    main()
