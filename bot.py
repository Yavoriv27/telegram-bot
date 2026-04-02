import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import json

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

import oandapyV20
from oandapyV20.endpoints import instruments

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
OANDA_KEY = os.getenv("OANDA_API_KEY")

client = oandapyV20.API(access_token=OANDA_KEY)

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]
AUTO = True

balance = 100
wins = 0
losses = 0
total = 0
loss_streak = 0
win_streak = 0

history = []

MIN_PROB = 78
MIN_SCORE = 8


def get_candles(pair, tf, count=150):
    params = {"granularity": tf, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)
    closes, full = [], []
    for c in r.response["candles"]:
        if c["complete"]:
            o = float(c["mid"]["o"])
            cl = float(c["mid"]["c"])
            h = float(c["mid"]["h"])
            l = float(c["mid"]["l"])
            closes.append(cl)
            full.append({"o": o, "c": cl, "h": h, "l": l})
    return closes, full


def ema(d, p): return np.mean(d[-p:])


def rsi(d, p=14):
    g, l = [], []
    for i in range(1, len(d)):
        diff = d[i] - d[i-1]
        if diff > 0: g.append(diff)
        else: l.append(abs(diff))
    ag = np.mean(g[-p:]) if g else 0.0001
    al = np.mean(l[-p:]) if l else 0.0001
    return 100 - (100 / (1 + ag/al))


def macd(d): return ema(d, 12) - ema(d, 26)


def analyze_pair(pair):
    m1, _ = get_candles(pair, "M1")
    m5, _ = get_candles(pair, "M5")
    m15, _ = get_candles(pair, "M15")

    trend = "BUY" if ema(m5,20) > ema(m5,50) and ema(m15,20) > ema(m15,50) else "SELL"

    r = rsi(m1)
    m = macd(m1)

    score = 0

    if trend == "BUY":
        if r < 35: score += 3
        if m > 0: score += 3
    else:
        if r > 65: score += 3
        if m < 0: score += 3

    if score < MIN_SCORE:
        return None

    prob = min(60 + score*5, 95)

    if prob < MIN_PROB:
        return None

    return {"pair": pair, "dir": trend, "prob": prob, "score": score}


def detect_bad_market():
    if len(history) < 5:
        return False
    last = history[-5:]
    losses_count = sum(1 for x in last if x == 0)
    return losses_count >= 4


def money_management(prob):
    global loss_streak, win_streak, balance

    if loss_streak >= 3:
        return 3
    if win_streak >= 3:
        return 15
    if prob > 85:
        return 10
    return 7


def generate_signal():
    if detect_bad_market():
        return None, "⛔ Ринок поганий (аналіз історії)"

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
        return None, "❌ Нема сигналу"

    sec = datetime.now().second
    entry = 60 - sec

    if entry > 25:
        return None, "⏳ Чекаємо ідеальний момент"

    stake = money_management(best['prob'])

    return best | {"entry": entry, "stake": stake}, None


def kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")],
        [InlineKeyboardButton("📊 Статистика", callback_data="stat")],
        [InlineKeyboardButton("✅ Плюс", callback_data="win"), InlineKeyboardButton("❌ Мінус", callback_data="loss")]
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 FINAL AI BOT", reply_markup=kb())


async def btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO, wins, losses, total, loss_streak, win_streak

    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s, reason = generate_signal()

        if not s:
            await q.message.reply_text(reason, reply_markup=kb())
            return

        msg = f"""
📊 {s['pair']}
{'🟢 BUY' if s['dir']=='BUY' else '🔴 SELL'}

📊 Ймовірність: {s['prob']}%
📈 Сила: {s['score']}

💵 Ставка: {s['stake']}%
⏱ Вхід через: {s['entry']} сек

🏆 FINAL SIGNAL
"""

        await q.message.reply_text(msg, reply_markup=kb())

    elif q.data == "auto":
        AUTO = not AUTO
        await q.message.reply_text(f"AUTO: {AUTO}", reply_markup=kb())

    elif q.data == "stat":
        winrate = (wins / total * 100) if total > 0 else 0
        await q.message.reply_text(f"📊 {total} | ✅ {wins} | ❌ {losses}\nWinrate: {round(winrate,1)}%", reply_markup=kb())

    elif q.data == "win":
        wins += 1
        total += 1
        win_streak += 1
        loss_streak = 0
        history.append(1)
        await q.message.reply_text("+ записано", reply_markup=kb())

    elif q.data == "loss":
        losses += 1
        total += 1
        loss_streak += 1
        win_streak = 0
        history.append(0)
        await q.message.reply_text("- записано", reply_markup=kb())


async def auto_job(context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    if not AUTO:
        return

    s, reason = generate_signal()

    if not s:
        await context.bot.send_message(chat_id=CHAT_ID, text=reason)
        return

    await context.bot.send_message(chat_id=CHAT_ID, text=f"🏆 {s['pair']} {s['dir']} {s['prob']}% FINAL")


def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(btn))

    app.job_queue.run_repeating(auto_job, interval=300, first=10)

    print("🚀 FINAL AI STARTED")
    app.run_polling()


if __name__ == "__main__":
    main()
