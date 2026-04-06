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

CHAT_IDS = set()
AUTO = True

user_data = {}

# ====== AI WEIGHTS (адаптація) ======
weights = {
    "Liquidity": 1.0,
    "Fake": 1.0,
    "Volume": 1.0,
    "Momentum": 1.0,
    "Entry": 1.0
}

# ================= DATA =================

def get_candles(pair, tf, count=120):
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
                "l": float(c["mid"]["l"]),
                "v": c.get("volume", 1)
            })
    return candles

# ================= CORE =================

def trend(data):
    return "UP" if data[-1]["c"] > data[-10]["c"] else "DOWN"

def volatility(c):
    return sum(abs(x["h"] - x["l"]) for x in c[-10:]) / 10

def is_news_time():
    now = datetime.utcnow()
    minutes = now.hour * 60 + now.minute
    news = [570, 930, 1020]
    return any(abs(minutes - t) < 30 for t in news)

def too_clean_trend(c):
    last5 = c[-5:]
    return all(x["c"] > x["o"] for x in last5) or all(x["c"] < x["o"] for x in last5)

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

def volume_spike(c):
    vols = [x["v"] for x in c[-10:]]
    avg = sum(vols[:-1]) / (len(vols) - 1)
    return vols[-1] > avg * 1.5

def predict_move(c):
    power = 0
    for x in c[-5:]:
        if x["c"] > x["o"]:
            power += abs(x["c"] - x["o"])
        else:
            power -= abs(x["c"] - x["o"])

    if power > 0.001:
        return "BUY", power
    if power < -0.001:
        return "SELL", power
    return "NEUTRAL", power

def best_entry(c):
    last = c[-1]
    body = abs(last["c"] - last["o"])
    full = last["h"] - last["l"]

    if full == 0:
        return "wait"

    ratio = body / full

    if ratio > 0.7:
        return "wait"

    if 0.3 < ratio < 0.7:
        return "enter"

    return "wait"

# ================= AI ADAPT =================

def adjust_weights(win, reasons):
    for r in reasons.split(", "):
        if r in weights:
            if win:
                weights[r] = min(weights[r] + 0.05, 2.0)
            else:
                weights[r] = max(weights[r] - 0.05, 0.5)

# ================= FINAL BOSS =================

def analyze(pair):
    c1 = get_candles(pair, "M1")
    c15 = get_candles(pair, "M15")

    if is_news_time():
        return None

    if volatility(c1) < 0.0004:
        return None

    if too_clean_trend(c1):
        return None

    direction = "BUY" if trend(c15) == "UP" else "SELL"

    score = 0
    reasons = []

    sweep = liquidity_sweep(c1)
    fake = fake_breakout(c1)
    vol = volume_spike(c1)

    if sweep == direction:
        score += 30 * weights["Liquidity"]
        reasons.append("Liquidity")

    if fake == direction:
        score += 20 * weights["Fake"]
        reasons.append("Fake")

    if vol:
        score += 15 * weights["Volume"]
        reasons.append("Volume")

    pred, _ = predict_move(c1)
    if pred == direction:
        score += 15 * weights["Momentum"]
        reasons.append("Momentum")

    entry = best_entry(c1)
    if entry == "enter":
        score += 10 * weights["Entry"]
        reasons.append("Entry")
    else:
        score -= 10

    if score < 65:
        return None

    return {
        "pair": pair,
        "dir": direction,
        "score": round(score),
        "prob": min(round(score), 95),
        "entry": entry,
        "reasons": ", ".join(reasons)
    }

def get_signal():
    best = None

    for pair in PAIRS:
        try:
            s = analyze(pair)
        except:
            continue

        if s:
            if not best or s["prob"] > best["prob"]:
                best = s

    return best

# ================= UI =================

def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")],
        [InlineKeyboardButton("🤖 Авто", callback_data="auto")],
        [InlineKeyboardButton("💰 Баланс", callback_data="balance")]
    ])

def get_user(chat_id):
    if chat_id not in user_data:
        user_data[chat_id] = {"balance": 3000, "win": 0, "loss": 0, "last": None}
    return user_data[chat_id]

# ================= HANDLERS =================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    CHAT_IDS.add(chat_id)
    get_user(chat_id)

    await update.message.reply_text(
        "🔥 FINAL BOSS BOT\nREADY",
        reply_markup=keyboard()
    )

async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO

    q = update.callback_query
    chat_id = q.message.chat.id
    user = get_user(chat_id)

    await q.answer()

    if q.data == "signal":
        s = get_signal()

        if not s:
            await q.edit_message_text("❌ Нема сигналу", reply_markup=keyboard())
            return

        user["last"] = s

        bet = round(user["balance"] * 0.1, 2)

        msg = f"""
🔥 FINAL BOSS

📊 {s['pair']}
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}

💵 {bet}
📊 {s['prob']}%

🧠 {s['reasons']}
"""

        await q.edit_message_text(
            msg,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Плюс", callback_data="win"),
                 InlineKeyboardButton("❌ Мінус", callback_data="loss")],
                [InlineKeyboardButton("📈 Прогноз", callback_data="signal")]
            ])
        )

    elif q.data == "win":
        user["balance"] += user["balance"] * 0.1
        user["win"] += 1

        if user["last"]:
            adjust_weights(True, user["last"]["reasons"])

        await q.edit_message_text("✅ WIN", reply_markup=keyboard())

    elif q.data == "loss":
        user["balance"] -= user["balance"] * 0.1
        user["loss"] += 1

        if user["last"]:
            adjust_weights(False, user["last"]["reasons"])

        await q.edit_message_text("❌ LOSS", reply_markup=keyboard())

    elif q.data == "balance":
        total = user["win"] + user["loss"]
        wr = (user["win"] / total * 100) if total > 0 else 0

        await q.edit_message_text(
            f"💰 {round(user['balance'],2)}\n📊 WR: {round(wr,2)}%",
            reply_markup=keyboard()
        )

    elif q.data == "auto":
        AUTO = not AUTO
        await q.edit_message_text(f"🤖 AUTO: {AUTO}", reply_markup=keyboard())

# ================= AUTO =================

async def auto_signal(context: ContextTypes.DEFAULT_TYPE):
    if not AUTO:
        return

    s = get_signal()
    if not s:
        return

    msg = f"""
🚀 AUTO

📊 {s['pair']}
{'🔼 BUY' if s['dir']=='BUY' else '🔻 SELL'}
📊 {s['prob']}%
"""

    for chat_id in CHAT_IDS:
        try:
            await context.bot.send_message(chat_id, msg)
        except:
            pass

# ================= MAIN =================

def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(buttons))

    app.job_queue.run_repeating(auto_signal, interval=300, first=10)

    print("🔥 FINAL BOSS AI RUNNING")
    app.run_polling()

if __name__ == "__main__":
    main()
