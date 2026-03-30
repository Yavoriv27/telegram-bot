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

# ================= DATA =================
def get_data(pair):
    params = {"granularity": "M1", "count": 50, "price": "M"}

    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)

    data = []

    for c in r.response["candles"]:
        if c["complete"]:
            data.append(float(c["mid"]["c"]))

    return data

# ================= ANALYSIS =================
def analyze(prices):
    if len(prices) < 10:
        return None

    buy = 0
    sell = 0

    # momentum
    if prices[-1] > prices[-2]:
        buy += 1
    else:
        sell += 1

    # trend
    if prices[-1] > sum(prices[-5:]) / 5:
        buy += 1
    else:
        sell += 1

    # price action
    if prices[-1] > prices[-2] > prices[-3]:
        buy += 2
    elif prices[-1] < prices[-2] < prices[-3]:
        sell += 2

    total = buy + sell
    if total == 0:
        return None

    buy_score = buy / total * 100
    sell_score = sell / total * 100

    if buy_score > sell_score:
        return "BUY", buy_score
    else:
        return "SELL", sell_score

# ================= SIGNAL =================
def generate_signal():
    best = None

    for pair in PAIRS:
        prices = get_data(pair)
        signal = analyze(prices)

        if not signal:
            continue

        direction, score = signal

        if not best or score > best["score"]:
            best = {
                "pair": pair,
                "direction": direction,
                "score": score
            }

    return best

# ================= UI =================
def main_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Прогноз", callback_data="signal")]
    ])

def result_kb():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ WIN", callback_data="win"),
            InlineKeyboardButton("❌ LOSS", callback_data="loss")
        ]
    ])

# ================= HANDLERS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 OANDA BOT", reply_markup=main_kb())

async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if q.data == "signal":
        s = generate_signal()

        if not s:
            await q.message.reply_text("❌ Немає сигналу", reply_markup=main_kb())
            return

        msg = f"""
📊 {s['pair']}
{ '🟢 BUY' if s['direction']=='BUY' else '🔴 SELL' }

📈 {s['score']:.0f}%
⏱ 2 хв
🕒 {datetime.utcnow().strftime('%H:%M:%S')}
"""
        await q.message.reply_text(msg, reply_markup=result_kb())

    elif q.data in ["win", "loss"]:
        await q.message.reply_text("Збережено", reply_markup=main_kb())

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(buttons))

    print("🚀 OANDA BOT STARTED")
    app.run_polling()

if __name__ == "__main__":
    main()
