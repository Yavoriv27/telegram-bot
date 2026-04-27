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

LAST_SIGNAL_TIME = 0

# ================= CONFIG =================

CONFIG = {
    "score_threshold": 70,
    "strength_limit": 0.5
}

stats = {
    "wins": 0,
    "losses": 0
}

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
    trs = []
    for i in range(1, len(c)):
        h = c[i]["h"]
        l = c[i]["l"]
        pc = c[i-1]["c"]

        tr = max(h-l, abs(h-pc), abs(l-pc))
        trs.append(tr)

    return sum(trs[-14:]) / 14 if len(trs) >= 14 else 0

def rsi(c, period=14):
    gains, losses = [], []

    for i in range(1, len(c)):
        diff = c[i]["c"] - c[i-1]["c"]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def strength(c):
    l = c[-1]
    return abs(l["c"]-l["o"]) / (l["h"]-l["l"]) if (l["h"]-l["l"]) else 0

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

# ================= STATS =================

def winrate():
    total = stats["wins"] + stats["losses"]
    if total == 0:
        return 0
    return round((stats["wins"] / total) * 100, 2)

def adapt():
    total = stats["wins"] + stats["losses"]

    if total < 20:
        return

    wr = winrate()

    if wr < 50:
        CONFIG["score_threshold"] -= 2
        CONFIG["strength_limit"] -= 0.02

    elif wr > 65:
        CONFIG["score_threshold"] += 2
        CONFIG["strength_limit"] += 0.02

    CONFIG["score_threshold"] = max(60, min(85, CONFIG["score_threshold"]))
    CONFIG["strength_limit"] = max(0.4, min(0.7, CONFIG["strength_limit"]))

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
    c1 = get_candles("M1")
    c5 = get_candles("M5")
    c15 = get_candles("M15")

    if not c1 or not c5 or not c15:
        return None

    e20 = ema(c15[-50:], 20)
    e50 = ema(c15[-50:], 50)

    trend15 = structure(c15)
    trend5 = structure(c5)

    direction = "BUY" if e20 > e50 else "SELL"

    score = 0

    if (e20 > e50 and trend15 == "UP") or (e20 < e50 and trend15 == "DOWN"):
        score += 30

    if trend5 == trend15:
        score += 20

    if entry_ok(direction, c1):
        score += 20

    if strength(c1) > CONFIG["strength_limit"]:
        score += 10

    r, s = levels(c15)
    price = c1[-1]["c"]

    if price > r - 0.0007 or price < s + 0.0007:
        score += 10

    rsi_val = rsi(c5)

    if direction == "BUY" and rsi_val < 70:
        score += 10
    elif direction == "SELL" and rsi_val > 30:
        score += 10

    atr_val = atr(c15)
    if (c1[-1]["h"] - c1[-1]["l"]) < atr_val * 1.5:
        score += 10

    last3 = c1[-3:]
    if all(x["c"] > x["o"] for x in last3) or all(x["c"] < x["o"] for x in last3):
        return None

    if score < CONFIG["score_threshold"]:
        return None

    return {
        "dir": direction,
        "score": score
    }

# ================= UI =================

def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")],
        [InlineKeyboardButton("✅", callback_data="win"),
         InlineKeyboardButton("❌", callback_data="loss")]
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("🔥 V14 PRO", reply_markup=keyboard())

async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s = analyze()

        if not s:
            await q.edit_message_text("❌ Нема сетапу", reply_markup=keyboard())
            return

        msg = f"""
🔥 V14 SIGNAL

📊 EUR/USD
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

📊 Score: {s['score']}%
📊 Winrate: {winrate()}%
"""
        await q.edit_message_text(msg, reply_markup=keyboard())

    elif q.data == "auto":
        AUTO = not AUTO
        await q.edit_message_text(f"AUTO: {AUTO}", reply_markup=keyboard())

    elif q.data == "win":
        stats["wins"] += 1
        adapt()
        await q.answer("+")

    elif q.data == "loss":
        stats["losses"] += 1
        adapt()
        await q.answer("-")

# ================= LOOP =================

async def loop(app):
    while True:
        if AUTO:
            s = analyze()
            if s:
                msg = f"""
🔥 AUTO SIGNAL

📊 EUR/USD
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

📊 Score: {s['score']}%
📊 Winrate: {winrate()}%
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

    print("🔥 V14 PRO RUNNING")
    app.run_polling()

if __name__ == "__main__":
    main()
