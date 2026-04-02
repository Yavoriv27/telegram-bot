import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import numpy as np

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

wins = 0
losses = 0
total = 0

MIN_PROB = 78
MIN_SCORE = 8


def get_candles(pair, tf, count=150):
    params = {"granularity": tf, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)
    closes = []
    for c in r.response["candles"]:
        if c["complete"]:
            closes.append(float(c["mid"]["c"]))
    return closes


def ema(d, p): return sum(d[-p:]) / p


def rsi(d, p=14):
    g, l = [], []
    for i in range(1, len(d)):
        diff = d[i] - d[i-1]
        if diff > 0: g.append(diff)
        else: l.append(abs(diff))
    ag = sum(g[-p:]) / p if g else 0.0001
    al = sum(l[-p:]) / p if l else 0.0001
    return 100 - (100 / (1 + ag/al))


def macd(d): return ema(d, 12) - ema(d, 26)


def analyze_pair(pair):
    m1 = get_candles(pair, "M1")
    m5 = get_candles(pair, "M5")
    m15 = get_candles(pair, "M15")

    trend = "BUY" if ema(m5,20) > ema(m5,50) and ema(m15,20) > ema(m15,50) else "SELL"

    r = rsi(m1)
    m = macd(m1)

    score = 0

    if trend == "BUY":
        if r < 35: score += 4
        if m > 0: score += 4
    else:
        if r > 65: score += 4
        if m < 0: score += 4

    if score < MIN_SCORE:
        return None

    prob = min(60 + score*5, 95)

    if prob < MIN_PROB:
        return None

    return {"pair": pair, "dir": trend, "prob": prob, "score": score}


def generate_signal():
    best = None

    for pair in PAIRS:
        try:
            s = analyze_pair(pair)
        except:
            continue

        if s:
            if not best or s['prob'] > best['prob']:
                best = s

    if not best:
        return None

    sec = datetime.now().second
    entry = 60 - sec

    if entry > 25:
        return None

    return best | {"entry": entry}


def kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")]
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("🚀 BOT READY", reply_markup=kb())


async def btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s = generate_signal()

        if not s:
            return

        msg = f"📊 {s['pair']}\n{'🟢 BUY' if s['dir']=='BUY' else '🔴 SELL'}\n📊 {s['prob']}%\n⏱ {s['entry']} сек"

        await q.message.reply_text(msg)

    elif q.data == "auto":
        AUTO = not AUTO
        await q.message.edit_text(f"🤖 AUTO: {AUTO}", reply_markup=kb())


async def auto_job(context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    if not AUTO:
        return

    s = generate_signal()

    if not s:
        return

    msg = f"🏆 {s['pair']} {'BUY' if s['dir']=='BUY' else 'SELL'} {s['prob']}%"

    for chat_id in CHAT_IDS:
        try:
            await context.bot.send_message(chat_id=chat_id, text=msg)
        except:
            pass


def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(btn))

    app.job_queue.run_repeating(auto_job, interval=300, first=10)

    print("🚀 FINAL BOT STARTED")
    app.run_polling()


if __name__ == "__main__":
    main()
