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
AUTO = True
CHAT_IDS = set()

LAST_SIGNAL_TIME = {}
COOLDOWN = 600


# ================= DATA =================
def get_candles(pair, tf, count=120):
    params = {"granularity": tf, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)

    closes = []
    candles = []

    for c in r.response["candles"]:
        if c["complete"]:
            o = float(c["mid"]["o"])
            cl = float(c["mid"]["c"])
            h = float(c["mid"]["h"])
            l = float(c["mid"]["l"])

            closes.append(cl)
            candles.append({"o": o, "c": cl, "h": h, "l": l})

    return closes, candles


# ================= INDICATORS =================
def ema(data, p):
    return sum(data[-p:]) / p

def trend(data):
    fast = ema(data, 10)
    slow = ema(data, 30)
    return "UP" if fast > slow else "DOWN"


def rsi(d, p=14):
    g, l = [], []
    for i in range(1, len(d)):
        diff = d[i] - d[i-1]
        if diff > 0: g.append(diff)
        else: l.append(abs(diff))
    ag = sum(g[-p:]) / p if g else 0.0001
    al = sum(l[-p:]) / p if l else 0.0001
    return 100 - (100 / (1 + ag/al))


def macd(d):
    return ema(d, 12) - ema(d, 26)


# ================= PULLBACK =================
def pullback_entry(candles, direction):
    if len(candles) < 4:
        return False

    last = candles[-1]
    prev = candles[-2]
    prev2 = candles[-3]

    # BUY: було падіння → тепер ріст
    if direction == "BUY":
        if prev2["c"] < prev2["o"] and prev["c"] < prev["o"]:
            if last["c"] > last["o"]:
                return True

    # SELL: було зростання → тепер падіння
    if direction == "SELL":
        if prev2["c"] > prev2["o"] and prev["c"] > prev["o"]:
            if last["c"] < last["o"]:
                return True

    return False


# ================= LATE =================
def is_late(c):
    last = c[-1]
    body = abs(last["c"] - last["o"])
    full = last["h"] - last["l"]

    if full == 0:
        return False

    return body / full > 0.7


# ================= ANALYSIS =================
def analyze(pair):

    m1, c1 = get_candles(pair, "M1")
    m5, c5 = get_candles(pair, "M5")
    m15, c15 = get_candles(pair, "M15")

    t1 = trend(m1)
    t5 = trend(m5)
    t15 = trend(m15)

    # синхрон тренду
    if t15 != t5:
        return None, "Нема тренду"

    direction = "BUY" if t15 == "UP" else "SELL"

    # ❗ ВХІД ТІЛЬКИ ПІСЛЯ ВІДКАТУ
    if not pullback_entry(c1, direction):
        return None, "Нема відкату"

    # ❗ АНТИ ЗАПІЗНЕННЯ
    if is_late(c1):
        return None, "Запізно"

    r = rsi(m1)
    m = macd(m1)

    score = 0
    reasons = []

    if direction == "BUY":
        if r < 50:
            score += 2
        if m > 0:
            score += 2

    if direction == "SELL":
        if r > 50:
            score += 2
        if m < 0:
            score += 2

    # BOOST
    if direction == "BUY" and r < 40 and m > 0:
        score += 2
        reasons.append("RSI+MACD")

    if direction == "SELL" and r > 60 and m < 0:
        score += 2
        reasons.append("RSI+MACD")

    if score < 4:
        return None, "Слабкий"

    prob = min(65 + score * 5, 95)

    return {
        "pair": pair,
        "dir": direction,
        "prob": prob,
        "score": score,
        "trend": t15,
        "reasons": ", ".join(reasons)
    }, None


# ================= SIGNAL =================
def get_signal():
    best = None
    now = datetime.now().timestamp()

    for pair in PAIRS:
        last = LAST_SIGNAL_TIME.get(pair)
        if last and now - last < COOLDOWN:
            continue

        try:
            s, _ = analyze(pair)
        except:
            continue

        if s:
            if not best or s["prob"] > best["prob"]:
                best = s

    if not best:
        return None, "Нема сигналу"

    LAST_SIGNAL_TIME[best["pair"]] = now

    return best, None


# ================= UI =================
def kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="sig")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")]
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("😈 ULTIMATE BOT READY", reply_markup=kb())


async def btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO
    q = update.callback_query
    await q.answer()

    if q.data == "sig":
        s, r = get_signal()

        if not s:
            await q.message.reply_text(f"❌ {r}")
            return

        msg = f"""
🔥 ULTIMATE

📊 {s['pair']}
{'🟢 BUY' if s['dir']=='BUY' else '🔴 SELL'}

📊 {s['prob']}%
📈 Score: {s['score']}

📈 Тренд: {s['trend']}
💡 {s['reasons']}
"""
        await q.message.reply_text(msg)

    elif q.data == "auto":
        AUTO = not AUTO
        await q.message.edit_text(f"🤖 AUTO: {AUTO}", reply_markup=kb())


async def auto(context: ContextTypes.DEFAULT_TYPE):
    if not AUTO:
        return

    s, _ = get_signal()

    if not s or s["score"] < 5:
        return

    msg = f"""
🔥 SIGNAL
🏆 {s['pair']}
{'🟢 BUY' if s['dir']=='BUY' else '🔴 SELL'}
📊 {s['prob']}%
"""

    for chat_id in CHAT_IDS:
        try:
            await context.bot.send_message(chat_id=chat_id, text=msg)
        except:
            pass


def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(btn))

    app.job_queue.run_repeating(auto, interval=60, first=10)

    print("😈 ULTIMATE BOT STARTED")
    app.run_polling()


if __name__ == "__main__":
    main()
