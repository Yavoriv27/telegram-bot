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

PAIR = "EUR_USD"
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


# ================= INDICATORS =================

def ema(c, period):
    k = 2 / (period + 1)
    ema_val = c[0]["c"]
    for x in c:
        ema_val = x["c"] * k + ema_val * (1 - k)
    return ema_val


def rsi(c, period=14):
    gains, losses = [], []

    for i in range(1, len(c)):
        diff = c[i]["c"] - c[i - 1]["c"]
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(abs(diff))

    if len(gains) < period or len(losses) < period:
        return 50

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(c, period=14):
    trs = []
    for i in range(1, len(c)):
        high = c[i]["h"]
        low = c[i]["l"]
        prev_close = c[i-1]["c"]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        trs.append(tr)

    return sum(trs[-period:]) / period if len(trs) >= period else 0


def support_resistance(c):
    highs = [x["h"] for x in c[-20:]]
    lows = [x["l"] for x in c[-20:]]

    return max(highs), min(lows)


def engulfing(c):
    prev = c[-2]
    last = c[-1]

    if prev["c"] < prev["o"] and last["c"] > last["o"]:
        if last["c"] > prev["o"] and last["o"] < prev["c"]:
            return "BUY"

    if prev["c"] > prev["o"] and last["c"] < last["o"]:
        if last["c"] < prev["o"] and last["o"] > prev["c"]:
            return "SELL"

    return None


def get_volume():
    try:
        data = yf.download("6E=F", interval="1m", period="1d", progress=False)
        vols = data["Volume"].tail(5).tolist()
        return vols[-1] > sum(vols[:-1]) / len(vols[:-1])
    except:
        return False


# ================= CORE =================

def analyze(pair):
    c1 = get_candles(pair, "M1")
    c5 = get_candles(pair, "M5")
    c15 = get_candles(pair, "M15")

    if not c1 or not c5 or not c15:
        return None

    # EMA TREND
    ema20 = ema(c15[-50:], 20)
    ema50 = ema(c15[-50:], 50)

    if ema20 > ema50:
        direction = "BUY"
    elif ema20 < ema50:
        direction = "SELL"
    else:
        return None

    # ATR
    volatility = atr(c1)
    if volatility < 0.00015:
        return None

    # RSI
    r = rsi(c1)

    # LEVELS
    resistance, support = support_resistance(c15)
    price = c1[-1]["c"]

    if abs(price - resistance) < 0.0003 or abs(price - support) < 0.0003:
        return None

    # PRICE ACTION
    last3 = c1[-3:]
    up = all(x["c"] > x["o"] for x in last3)
    down = all(x["c"] < x["o"] for x in last3)

    # M5 confirm
    m5_last = c5[-1]
    m5_dir = "BUY" if m5_last["c"] > m5_last["o"] else "SELL"

    score = 0
    reasons = []

    score += 30
    reasons.append("EMA")

    if direction == "BUY" and up:
        score += 20
        reasons.append("PA")
    elif direction == "SELL" and down:
        score += 20
        reasons.append("PA")

    if m5_dir == direction:
        score += 15
        reasons.append("M5")

    eng = engulfing(c1)
    if eng == direction:
        score += 20
        reasons.append("Pattern")

    if direction == "BUY" and r < 70:
        score += 10
        reasons.append("RSI")
    elif direction == "SELL" and r > 30:
        score += 10
        reasons.append("RSI")

    last = c1[-1]
    body = abs(last["c"] - last["o"])
    full = last["h"] - last["l"]

    if full > 0 and body / full > 0.6:
        score += 10
        reasons.append("Impulse")

    if get_volume():
        score += 10
        reasons.append("Volume")

    if score >= 80:
        level = "STRONG"
    elif score >= 65:
        level = "NORMAL"
    else:
        return None

    return {
        "pair": pair,
        "dir": direction,
        "prob": min(score, 95),
        "level": level,
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
    await update.message.reply_text("🔥 FINAL AI CORE v3", reply_markup=keyboard())


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
🔥 FINAL AI CORE v3

📊 EUR/USD
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

⚡ {s['level']}
📊 {s['prob']}%
🧠 {s['reasons']}
"""
        await q.edit_message_text(msg, reply_markup=keyboard())

    elif q.data == "auto":
        AUTO = not AUTO
        await q.edit_message_text(f"🤖 AUTO: {AUTO}", reply_markup=keyboard())


# ================= AUTO =================

async def instant_loop(app):
    while True:
        if AUTO:
            now = datetime.utcnow().timestamp()

            last = LAST_SIGNAL_TIME.get(PAIR)
            if last and now - last < COOLDOWN:
                await asyncio.sleep(10)
                continue

            s = analyze(PAIR)

            if s:
                LAST_SIGNAL_TIME[PAIR] = now

                msg = f"""
🚀 SIGNAL

📊 EUR/USD
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

⚡ {s['level']}
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

    print("🔥 FINAL AI CORE v3 RUNNING")

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
