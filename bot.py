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

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]
CHAT_IDS = set()
AUTO = True

LAST_SIGNAL_TIME = {}
COOLDOWN = 120  # антиспам

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

    except Exception as e:
        print("❌ OANDA ERROR:", e)
        return []

# ================= REAL VOLUME =================

def get_real_volume():
    try:
        data = yf.download("6E=F", interval="1m", period="1d", progress=False)
        vols = data["Volume"].tail(10).tolist()
        avg = sum(vols[:-1]) / (len(vols) - 1)
        return vols[-1] > avg * 1.5
    except:
        return False

# ================= CORE =================

def trend(data):
    return "UP" if data[-1]["c"] > data[-10]["c"] else "DOWN"

def volatility(c):
    return sum(abs(x["h"] - x["l"]) for x in c[-10:]) / 10

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
        power += abs(x["c"] - x["o"]) if x["c"] > x["o"] else -abs(x["c"] - x["o"])

    if power > 0.001:
        return "BUY"
    if power < -0.001:
        return "SELL"
    return "NEUTRAL"

# ================= CANDLE AI =================

def candle_ai(c):
    last = c[-1]
    full = last["h"] - last["l"]

    if full == 0:
        return "NEUTRAL", 0

    close_pos = (last["c"] - last["l"]) / full

    if close_pos > 0.7:
        return "BUY", 2
    elif close_pos < 0.3:
        return "SELL", 2

    return "NEUTRAL", 0

# ================= ANALYZE =================

def analyze(pair):
    c1 = get_candles(pair, "M1")
    c15 = get_candles(pair, "M15")

    if volatility(c1) < 0.0004:
        return None

    direction = "BUY" if trend(c15) == "UP" else "SELL"

    score = 0
    reasons = []

    if liquidity_sweep(c1) == direction:
        score += 30
        reasons.append("Liquidity")

    if fake_breakout(c1) == direction:
        score += 20
        reasons.append("Fake")

    if get_real_volume():
        score += 25
        reasons.append("Volume")

    if predict_move(c1) == direction:
        score += 15
        reasons.append("Momentum")

    ai_dir, ai_score = candle_ai(c1)

    if ai_dir == direction:
        score += 10 + ai_score
        reasons.append("Candle")

    if score < 60:
        return None

    return {
        "pair": pair,
        "dir": direction,
        "prob": min(score, 95),
        "score": score,
        "reasons": ", ".join(reasons)
    if not c1 or not c15:
    return None
    }

# ================= UI =================

def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")]
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)

    await update.message.reply_text(
        "🔥 FINAL BOSS INSTANT READY",
        reply_markup=keyboard()
    )

async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s = get_best_signal()

        if not s:
            await q.edit_message_text("❌ Нема сигналу", reply_markup=keyboard())
            return

        msg = f"""
🔥 FINAL BOSS

📊 {s['pair']}
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

📊 {s['prob']}%
🧠 {s['reasons']}
"""

        await q.edit_message_text(msg, reply_markup=keyboard())

    elif q.data == "auto":
        AUTO = not AUTO
        await q.edit_message_text(f"🤖 AUTO: {AUTO}", reply_markup=keyboard())

# ================= SIGNAL =================

def get_best_signal():
    best = None
    for pair in PAIRS:
        s = analyze(pair)
        if s:
            if not best or s["prob"] > best["prob"]:
                best = s
    return best

# ================= INSTANT LOOP =================

async def instant_loop(app):
    while True:
        if AUTO:
            for pair in PAIRS:
                now = datetime.utcnow().timestamp()

                last = LAST_SIGNAL_TIME.get(pair)
                if last and now - last < COOLDOWN:
                    continue

                s = analyze(pair)

                if s:
                    LAST_SIGNAL_TIME[pair] = now

                    msg = f"""
🚀 INSTANT SIGNAL

📊 {s['pair']}
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
        # запускаємо loop вже після старту
        asyncio.get_event_loop().create_task(instant_loop(app))

    app.post_init = post_init

    print("🔥 FINAL BOSS INSTANT RUNNING")

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
