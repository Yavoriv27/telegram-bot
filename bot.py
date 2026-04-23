# -*- coding: utf-8 -*-

import os
import asyncio
from dotenv import load_dotenv
from datetime import datetime

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
    k = 2/(p+1)
    e = c[0]["c"]
    for x in c:
        e = x["c"]*k + e*(1-k)
    return e

def atr(c):
    trs = [abs(c[i]["h"]-c[i]["l"]) for i in range(-14,-1)]
    return sum(trs)/len(trs) if trs else 0

def strength(c):
    l=c[-1]
    return abs(l["c"]-l["o"])/(l["h"]-l["l"]) if (l["h"]-l["l"]) else 0

def structure(c):
    if c[-1]["h"]>c[-2]["h"] and c[-1]["l"]>c[-2]["l"]:
        return "UP"
    if c[-1]["h"]<c[-2]["h"] and c[-1]["l"]<c[-2]["l"]:
        return "DOWN"
    return "RANGE"

def levels(c):
    highs=[x["h"] for x in c[-20:]]
    lows=[x["l"] for x in c[-20:]]
    return max(highs),min(lows)

# ================= MARKET STATE =================

def market_state(c15):
    atr_val = atr(c15)
    e20 = ema(c15[-50:],20)
    e50 = ema(c15[-50:],50)

    trend_power = abs(e20 - e50)

    if atr_val < 0.0003:
        return "WEAK"
    elif atr_val > 0.0007 and trend_power > 0.0003:
        return "STRONG"
    else:
        return "NORMAL"

# ================= ENTRY =================

def entry_ok(direction, c1):
    prev = c1[-2]
    last = c1[-1]

    if direction == "BUY":
        return prev["c"] < prev["o"] and last["c"] > last["o"]
    else:
        return prev["c"] > prev["o"] and last["c"] < last["o"]

# ================= CORE =================

def analyze():
    global LAST_DIRECTION

    c1 = get_candles("M1")
    c5 = get_candles("M5")
    c15 = get_candles("M15")

    if not c1 or not c5 or not c15:
        return None

    state = market_state(c15)

    # ❌ не торгуємо в слабкому
    if state == "WEAK":
        return None

    e20 = ema(c15[-50:],20)
    e50 = ema(c15[-50:],50)
    trend = structure(c15)

    if not ((e20>e50 and trend=="UP") or (e20<e50 and trend=="DOWN")):
        return None

    direction = "BUY" if e20>e50 else "SELL"

    if LAST_DIRECTION == direction:
        return None

    # 🔥 адаптивні параметри
    if state == "STRONG":
        strength_limit = 0.45
        zone_range = 0.0008
    else:
        strength_limit = 0.55
        zone_range = 0.0006

    # ❌ перегрів
    last3 = c1[-3:]
    if all(x["c"]>x["o"] for x in last3) or all(x["c"]<x["o"] for x in last3):
        return None

    # ❌ велика свічка
    if (c1[-1]["h"]-c1[-1]["l"]) > 0.0006:
        return None

    # ENTRY
    if not entry_ok(direction, c1):
        return None

    # сила
    if strength(c1) < strength_limit:
        return None

    # зона
    r,s = levels(c15)
    price = c1[-1]["c"]

    if not (price > r-zone_range or price < s+zone_range):
        return None

    # простір
    if direction=="BUY" and (r-price)<0.0003:
        return None
    if direction=="SELL" and (price-s)<0.0003:
        return None

    LAST_DIRECTION = direction

    return {"dir":direction, "state":state}

# ================= UI =================

def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз",callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто",callback_data="auto")],
        [InlineKeyboardButton("✅",callback_data="win"),
         InlineKeyboardButton("❌",callback_data="loss")]
    ])

async def start(update:Update,context:ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("🔥 V12 ADAPTIVE",reply_markup=keyboard())

async def buttons(update:Update,context:ContextTypes.DEFAULT_TYPE):
    global AUTO

    q=update.callback_query
    await q.answer()

    if q.data=="signal":
        s=analyze()

        if not s:
            await q.edit_message_text("❌ Нема сетапу",reply_markup=keyboard())
            return

        msg=f"""
🔥 ADAPTIVE SIGNAL

📊 EUR/USD
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

🧠 Ринок: {s['state']}
📌 Вхід після M1
"""
        await q.edit_message_text(msg,reply_markup=keyboard())

    elif q.data=="auto":
        AUTO=not AUTO
        await q.edit_message_text(f"AUTO: {AUTO}",reply_markup=keyboard())

    elif q.data=="win":
        HISTORY.append("win")
        await q.answer("+")

    elif q.data=="loss":
        HISTORY.append("loss")
        await q.answer("-")

# ================= LOOP =================

async def loop(app):
    while True:
        if AUTO:
            s=analyze()
            if s:
                msg=f"""
🔥 ADAPTIVE SIGNAL

📊 EUR/USD
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

🧠 Ринок: {s['state']}
📌 Вхід після M1
"""
                for chat_id in CHAT_IDS:
                    await app.bot.send_message(chat_id,msg)

        await asyncio.sleep(25)

# ================= MAIN =================

def main():
    app=Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",start))
    app.add_handler(CallbackQueryHandler(buttons))

    async def post_init(app):
        asyncio.get_event_loop().create_task(loop(app))

    app.post_init=post_init

    print("🔥 V12 ADAPTIVE RUNNING")

    app.run_polling()

if __name__=="__main__":
    main()
