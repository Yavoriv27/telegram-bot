import os
import json
import time
import queue
import math
import threading
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List

import requests
from dotenv import load_dotenv
import pytz

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()

# ---------------- CONFIG ----------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("telegram_oanda_bot")

KYIV_TZ = pytz.timezone("Europe/Kyiv")

# ---------------- HELPERS ----------------

def now_utc():
    return datetime.now(timezone.utc)

def fmt_kyiv(dt):
    return dt.astimezone(KYIV_TZ).strftime("%Y-%m-%d %H:%M:%S")

def mean(xs):
    return sum(xs) / max(1, len(xs))

# ---------------- CANDLES ----------------

@dataclass
class Candle:
    tf_sec: int
    start_ts: float
    open: float
    high: float
    low: float
    close: float

    def update(self, price: float):
        self.close = price
        self.high = max(self.high, price)
        self.low = min(self.low, price)

class InternalCandleBuilder:
    def __init__(self, tf_sec: int):
        self.tf_sec = tf_sec
        self.current = None
        self.last_closed = None

    def on_tick(self, ts, price):
        bucket = ts - (ts % self.tf_sec)

        if not self.current:
            self.current = Candle(self.tf_sec, bucket, price, price, price, price)
            return

        if bucket == self.current.start_ts:
            self.current.update(price)
            return

        self.last_closed = self.current
        self.current = Candle(self.tf_sec, bucket, price, price, price, price)

class CandleHistory:
    def __init__(self, maxlen=400):
        self.items_list = []
        self.maxlen = maxlen

    def append(self, c):
        self.items_list.append(c)
        if len(self.items_list) > self.maxlen:
            self.items_list = self.items_list[-self.maxlen:]

    def items(self):
        return list(self.items_list)

# ---------------- INDICATORS ----------------

def rsi(values, period=14):
    if len(values) < period + 1:
        return None
    gains, losses = [], []
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def adx(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None

    trs, p_dm, m_dm = [], [], []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        p_dm.append(up if up > down and up > 0 else 0)
        m_dm.append(down if down > up and down > 0 else 0)

    if len(trs) < period:
        return None

    tr14 = sum(trs[:period])
    p14 = sum(p_dm[:period])
    m14 = sum(m_dm[:period])

    dxs = []
    for i in range(period, len(trs)):
        tr14 = tr14 - tr14 / period + trs[i]
        p14 = p14 - p14 / period + p_dm[i]
        m14 = m14 - m14 / period + m_dm[i]
        if tr14 == 0:
            continue
        di_p = 100 * p14 / tr14
        di_m = 100 * m14 / tr14
        dxs.append(100 * abs(di_p - di_m) / (di_p + di_m))

    return mean(dxs[-period:])

# ---------------- SIGNAL ENGINE ----------------

class SignalEngine:
    def __init__(self):
        self.symbol = "EUR_USD"

        self.tf_fast = 60
        self.tf_slow = 600  # 10 хв

        self.builder_fast = InternalCandleBuilder(self.tf_fast)
        self.builder_slow = InternalCandleBuilder(self.tf_slow)

        self.hist_fast = CandleHistory()
        self.hist_slow = CandleHistory()

        self.last_tick = None
        self._last_slow_ts = None

        self.cooldown_sec = 600
        self._last_sent_ts = 0
        self._last_dir = None
        self._last_signal_candle = None

        self.q = queue.Queue()

    def start_stream(self):
        api_key = os.getenv("OANDA_API_KEY")
        acc = os.getenv("OANDA_ACCOUNT_ID")

        url = f"https://stream-fxpractice.oanda.com/v3/accounts/{acc}/pricing/stream"
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"instruments": self.symbol}

        def stream():
            while True:
                with requests.get(url, headers=headers, params=params, stream=True) as r:
                    for line in r.iter_lines():
                        if not line:
                            continue
                        msg = json.loads(line)
                        if msg.get("type") == "PRICE":
                            bid = float(msg["bids"][0]["price"])
                            ask = float(msg["asks"][0]["price"])
                            self.q.put((time.time(), (bid + ask) / 2))

        threading.Thread(target=stream, daemon=True).start()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while True:
            ts, price = self.q.get()
            self.last_tick = price

            self.builder_fast.on_tick(ts, price)
            self.builder_slow.on_tick(ts, price)

            if self.builder_slow.last_closed:
                c = self.builder_slow.last_closed
                if c.start_ts != self._last_slow_ts:
                    self._last_slow_ts = c.start_ts
                    self.hist_slow.append(c)

    def compute_signal(self):
        slow = self.hist_slow.items()
        if len(slow) < 30:
            return {"ok": False, "reason": "NOT_ENOUGH_DATA"}

        last_candle = slow[-1]
        if self._last_signal_candle == last_candle.start_ts:
            return {"ok": False, "reason": "WAIT_NEXT_CANDLE"}

        self._last_signal_candle = last_candle.start_ts

        closes = [c.close for c in slow]
        highs = [c.high for c in slow]
        lows = [c.low for c in slow]

        r = rsi(closes)
        a = adx(highs, lows, closes)

        if r is None or a is None:
            return {"ok": False, "reason": "NO_DATA"}

        if a < 18:
            return {"ok": False, "reason": "MARKET_FLAT"}

        if a < 22 or a > 35:
            return {"ok": False, "reason": "ADX_FILTER"}

        now = time.time()
        if now - self._last_sent_ts < self.cooldown_sec:
            return {"ok": False, "reason": "COOLDOWN"}

        direction = None
        if 60 <= r <= 63:
            direction = "BUY"
        elif 37 <= r <= 40:
            direction = "SELL"

        if not direction or direction == self._last_dir:
            return {"ok": False, "reason": "NO_SIGNAL"}

        self._last_dir = direction
        self._last_sent_ts = now

        return {
            "ok": True,
            "direction": direction,
            "expiry_sec": 600,
            "rsi": round(r, 1),
            "adx": round(a, 1)
        }

ENGINE = SignalEngine()

# ---------------- TELEGRAM ----------------

def fmt_signal(sig):
    t = fmt_kyiv(now_utc())
    if sig.get("ok"):
        return (
            f"{'🟢 BUY' if sig['direction']=='BUY' else '🔴 SELL'}\n"
            f"⏱ 10 хв\n"
            f"🕒 {t}\n"
            f"RSI: {sig['rsi']} | ADX: {sig['adx']}"
        )
    return f"❌ Немає сигналу\n🕒 {t}\n{sig.get('reason')}"

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sig = ENGINE.compute_signal()
    await update.message.reply_text(fmt_signal(sig))

def main():
    ENGINE.start_stream()
    app = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.run_polling()

if __name__ == "__main__":
    main()
