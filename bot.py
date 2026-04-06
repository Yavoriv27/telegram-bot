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


# ================= TREND =================
def trend_tf(data):
    fast = ema(data, 10)
    slow = ema(data, 30)

    if fast > slow:
        return "UP"
    elif fast < slow:
        return "DOWN"
    return "FLAT"


# ================= LATE FILTER =================
def is_late_entry(candles):
    last = candles[-1]
    body = abs(last["c"] - last["o"])
    full = last["h"] - last["l"]

    if full == 0:
        return False

    if body / full > 0.7:
        return True

    return False


# ================= ANALYSIS =================
def analyze_pair(pair):

    # 🔥 MULTI TF
    m1, c1 = get_candles(pair, "M1")
    m5, c5 = get_candles(pair, "M5")
    m15, c15 = get_candles(pair, "M15")

    trend1 = trend_tf(m1)
    trend5 = trend_tf(m5)
    trend15 = trend_tf(m15)

    # ❗ ФІЛЬТР СИНХРОНУ
    if trend15 != trend5:
        return None, "Нема синхрону M15/M5"

    direction = "BUY" if trend15 == "UP" else "SELL"

    # ❗ АНТИ ЗАПІЗНЕННЯ
    if is_late_entry(c1):
        return None, "Запізно"

    r = round(rsi(m1), 2)
    m = round(macd(m1), 5)

    score = 0

    # RSI
    if direction == "BUY" and r < 50:
        score += 2
    if direction == "SELL" and r > 50:
        score += 2

    # MACD
    if direction == "BUY" and m > 0:
        score += 2
    if direction == "SELL" and m < 0:
        score += 2

    # СВІЧКА
    last = c1[-1]
    if direction == "BUY" and last["c"] > last["o"]:
        score += 1
    if direction == "SELL" and last["c"] < last["o"]:
        score += 1

    # 🔥 BOOST
    boost = 0
    reasons = []

    if direction == "BUY" and r < 40 and m > 0:
        boost += 2
        reasons.append("RSI+MACD")

    if direction == "SELL" and r > 60 and m < 0:
        boost += 2
        reasons.append("RSI+MACD")

    score += boost

    if score < 5:
        return None, "Слабкий сигнал"

    prob = min(60 + score * 5, 95)

    return {
        "pair": pair,
        "dir": direction,
        "prob": prob,
        "score": score,
        "trend": trend15,
        "reasons": ", ".join(reasons)
    }, None


# ================= SIGNAL =================
def generate_signal():
    best = None
    now = datetime.now().timestamp()

    for pair in PAIRS:
        last_time = LAST_SIGNAL_TIME.get(pair)
        if last_time and now - last_time < COOLDOWN:
            continue

        try:
            s, _ = analyze_pair(pair)
        except:
            continue

        if s:
            if not best or s["prob"] > best["prob"]:
                best = s

    if not best:
        return None, "Нема умов"

    LAST_SIGNAL_TIME[best["pair"]] = now

    return best, None


# ================= UI =================
def kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")]
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("🚀 MAX BOT READY", reply_markup=kb())


async def btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s, reason = generate_signal()

        if not s:
            await q.message.reply_text(f"❌ {reason}")
            return

        msg = f"""
🔥 СИЛЬНИЙ

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


async def auto_job(context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    if not AUTO:
        return

    s, _ = generate_signal()

    if not s or s["score"] < 6:
        return

    msg = f"""
🔥 СИГНАЛ
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

    app.job_queue.run_repeating(auto_job, interval=60, first=10)

    print("🚀 MAX PRECISION BOT STARTED")
    app.run_polling()


if __name__ == "__main__":
    main()
