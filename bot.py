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

MIN_PROB = 80
MIN_SCORE = 9


# ================== DATA ==================
def get_candles(pair, tf, count=150):
    params = {"granularity": tf, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)

    closes = []
    full = []

    for c in r.response["candles"]:
        if c["complete"]:
            o = float(c["mid"]["o"])
            cl = float(c["mid"]["c"])
            h = float(c["mid"]["h"])
            l = float(c["mid"]["l"])

            closes.append(cl)
            full.append({"o": o, "c": cl, "h": h, "l": l})

    return closes, full


# ================== INDICATORS ==================
def ema(d, p):
    return sum(d[-p:]) / p


def rsi(d, p=14):
    g, l = [], []
    for i in range(1, len(d)):
        diff = d[i] - d[i-1]
        if diff > 0:
            g.append(diff)
        else:
            l.append(abs(diff))

    ag = sum(g[-p:]) / p if g else 0.0001
    al = sum(l[-p:]) / p if l else 0.0001

    return 100 - (100 / (1 + ag/al))


def macd(d):
    return ema(d, 12) - ema(d, 26)


# ================== LEVELS ==================
def support_resistance(data):
    return min(data[-50:]), max(data[-50:])


# ================== PATTERNS ==================
def candle_pattern(c):
    if len(c) < 3:
        return None

    c1, c2 = c[-2], c[-1]

    bull1 = c1["c"] > c1["o"]
    bear1 = c1["c"] < c1["o"]
    bull2 = c2["c"] > c2["o"]
    bear2 = c2["c"] < c2["o"]

    # engulfing
    if bull2 and bear1 and c2["c"] > c1["o"]:
        return "BUY"
    if bear2 and bull1 and c2["c"] < c1["o"]:
        return "SELL"

    # pin bar
    body = abs(c2["c"] - c2["o"])
    wick = c2["h"] - c2["l"]

    if wick > body * 2:
        return "BUY" if bull2 else "SELL"

    return None


# ================== PRICE ACTION ==================
def price_action(c):
    last3 = c[-3:]

    up = all(x["c"] > x["o"] for x in last3)
    down = all(x["c"] < x["o"] for x in last3)

    if up:
        return "BUY"
    if down:
        return "SELL"

    return None


# ================== NEWS FILTER ==================
def news_filter():
    hour = datetime.utcnow().hour

    # сильні новини (приблизно)
    if hour in [12, 13, 14]:
        return False

    return True


# ================== ANALYSIS ==================
def analyze_pair(pair):
    m1, candles = get_candles(pair, "M1")
    m5, _ = get_candles(pair, "M5")
    m15, _ = get_candles(pair, "M15")

    # тренд
    trend = "BUY" if ema(m5,20) > ema(m5,50) and ema(m15,20) > ema(m15,50) else "SELL"

    # сила тренду
    trend_strength = abs(ema(m5,20) - ema(m5,50)) * 10000

    r = round(rsi(m1), 2)
    m = round(macd(m1), 5)

    s, r_lvl = support_resistance(m1)
    price = m1[-1]

    near_support = abs(price - s) < (r_lvl - s) * 0.25
    near_resistance = abs(price - r_lvl) < (r_lvl - s) * 0.25

    pat = candle_pattern(candles)
    pa = price_action(candles)

    score = 0

    # RSI + MACD
    if trend == "BUY":
        if r < 35: score += 2
        if m > 0: score += 2
        if near_support: score += 2
    else:
        if r > 65: score += 2
        if m < 0: score += 2
        if near_resistance: score += 2

    # патерн
    if pat == trend:
        score += 3
    else:
        return None, "Нема патерну"

    # price action
    if pa == trend:
        score += 2
    else:
        return None, "Price Action не підтверджує"

    if score < MIN_SCORE:
        return None, "Слабкий сигнал"

    prob = min(60 + score*4, 95)

    if prob < MIN_PROB:
        return None, "Низька ймовірність"

    return {
        "pair": pair,
        "dir": trend,
        "prob": prob,
        "score": score,
        "rsi": r,
        "macd": m,
        "trend_strength": round(trend_strength,2),
        "pattern": pat
    }, None


# ================== SIGNAL ==================
def generate_signal():
    if not news_filter():
        return None, "Новини (краще не торгувати)"

    best = None
    reason = ""

    for pair in PAIRS:
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
        return None, reason if reason else "Нема умов"

    sec = datetime.now().second
    entry = 60 - sec

    if entry > 25:
        return None, "Не ідеальний момент входу"

    return best | {"entry": entry}, None


# ================== UI ==================
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
            return

        msg = f"""
🏆 {s['pair']}
{'🟢 BUY' if s['dir']=='BUY' else '🔴 SELL'}

📊 Ймовірність: {s['prob']}%
"""

📊 Ймовірність: {s['prob']}%
📈 Сила: {s['score']}

📉 RSI: {s['rsi']}
📊 MACD: {s['macd']}
💪 Тренд: {s['trend_strength']}
🕯 Патерн: {s['pattern']}

⏱ Вхід через: {s['entry']} сек
"""

        await q.message.reply_text(msg)

    elif q.data == "auto":
        AUTO = not AUTO
        await q.message.edit_text(f"🤖 AUTO: {AUTO}", reply_markup=kb())


async def auto_job(context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    if not AUTO:
        return

    s, reason = generate_signal()

    if not s:
        msg = f"❌ Нема сигналу\nПричина: {reason}"
    else:
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

    app.job_queue.run_repeating(auto_job, interval=60, first=10)

    print("🚀 FINAL PRO BOT STARTED")
    app.run_polling()


if __name__ == "__main__":
    main()
