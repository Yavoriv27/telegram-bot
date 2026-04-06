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


# ================= LEVELS =================
def support_resistance(data):
    return min(data[-50:]), max(data[-50:])


# ================= PATTERN =================
def candle_pattern(c):
    if len(c) < 3:
        return None

    c1, c2 = c[-2], c[-1]

    if c2["c"] > c2["o"] and c1["c"] < c1["o"] and c2["c"] > c1["o"]:
        return "BUY"

    if c2["c"] < c2["o"] and c1["c"] > c1["o"] and c2["c"] < c1["o"]:
        return "SELL"

    return None


# ================= LATE FILTER =================
def is_late_entry(candles):
    if len(candles) < 10:
        return False

    last = candles[-1]
    body = abs(last["c"] - last["o"])
    full = last["h"] - last["l"]

    if full == 0:
        return False

    if body / full > 0.7:
        return True

    avg = sum(abs(c["c"] - c["o"]) for c in candles[-10:]) / 10

    if body > avg * 2:
        return True

    return False


# ================= CONFIRMATION =================
def confirm_entry(candles, direction):
    if len(candles) < 3:
        return False

    last = candles[-1]
    prev = candles[-2]

    # BUY підтвердження
    if direction == "BUY":
        return last["c"] > last["o"] and last["c"] > prev["c"]

    # SELL підтвердження
    if direction == "SELL":
        return last["c"] < last["o"] and last["c"] < prev["c"]

    return False


# ================= ANALYSIS =================
def analyze_pair(pair):
    m1, candles = get_candles(pair, "M1")

    if is_late_entry(candles):
        return None, "Запізно заходити"

    r = round(rsi(m1), 2)
    m = round(macd(m1), 5)

    support, resistance = support_resistance(m1)
    price = m1[-1]

    pat = candle_pattern(candles)

    near_support = abs(price - support) < (resistance - support) * 0.2
    near_resistance = abs(price - resistance) < (resistance - support) * 0.2

    score = 0
    direction = None

    # BUY
    if r < 35:
        score += 2
        direction = "BUY"
    if m > 0:
        score += 2
    if near_support:
        score += 3
    if pat == "BUY":
        score += 3

    # SELL
    sell_score = 0
    if r > 65:
        sell_score += 2
    if m < 0:
        sell_score += 2
    if near_resistance:
        sell_score += 3
    if pat == "SELL":
        sell_score += 3

    if sell_score > score:
        score = sell_score
        direction = "SELL"

    if score < 7:
        return None, "Нема розвороту"

    # 🔥 НОВЕ: ЧЕКАЄМО ПІДТВЕРДЖЕННЯ
    if not confirm_entry(candles, direction):
        return None, "Чекаємо підтвердження"

    prob = min(65 + score * 4, 95)
    strength = "🔥 СИЛЬНИЙ" if score >= 9 else "⚠️ СЕРЕДНІЙ"

    return {
        "pair": pair,
        "dir": direction,
        "prob": prob,
        "score": score,
        "rsi": r,
        "macd": m,
        "pattern": pat if pat else "нема",
        "strength": strength
    }, None


# ================= SIGNAL =================
def generate_signal():
    best = None
    reason = ""
    now = datetime.now().timestamp()

    for pair in PAIRS:
        last_time = LAST_SIGNAL_TIME.get(pair)
        if last_time and now - last_time < COOLDOWN:
            continue

        try:
            s, r = analyze_pair(pair)
        except:
            continue

        if s:
            if not best or s["prob"] > best["prob"]:
                best = s
        else:
            reason = r

    if not best:
        return None, reason

    LAST_SIGNAL_TIME[best["pair"]] = now

    return best | {"entry": 1}, None


# ================= UI =================
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
        s, reason = generate_signal()

        if not s:
            await q.message.reply_text(f"❌ {reason}")
            return

        msg = f"""
{s['strength']}

📊 {s['pair']}
{'🟢 BUY' if s['dir']=='BUY' else '🔴 SELL'}

📊 {s['prob']}%
📈 Сила: {s['score']}

📉 RSI: {s['rsi']}
📊 MACD: {s['macd']}
🕯 Патерн: {s['pattern']}

⏱ Вхід: ПІСЛЯ ПІДТВЕРДЖЕННЯ
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

    if not s or s["strength"] != "🔥 СИЛЬНИЙ":
        return

    msg = f"""
🔥 СИЛЬНИЙ
🏆 {s['pair']}
{'🟢 BUY' if s['dir']=='BUY' else '🔴 SELL'}
📊 {s['prob']}%
⏱ ПІСЛЯ ПІДТВЕРДЖЕННЯ
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

    print("🚀 IDEAL ENTRY BOT STARTED")
    app.run_polling()


if __name__ == "__main__":
    main()
