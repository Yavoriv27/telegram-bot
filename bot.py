# ===== SINGLE INSTANCE LOCK =====
import os, sys, atexit

LOCK_FILE = "/tmp/telegram_bot.lock"
if os.path.exists(LOCK_FILE):
    print("Bot already running, exit")
    sys.exit(0)

with open(LOCK_FILE, "w") as f:
    f.write(str(os.getpid()))

atexit.register(lambda: os.remove(LOCK_FILE) if os.path.exists(LOCK_FILE) else None)
# ===== END LOCK =====

# ================= IMPORTS =================
import json
import time
import queue
import math
import threading
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import requests
from dotenv import load_dotenv
import pytz

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("telegram_oanda_bot")

KYIV_TZ = pytz.timezone("Europe/Kyiv")

# ================= HELPERS =================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def fmt_kyiv(dt_utc: datetime) -> str:
    return dt_utc.astimezone(KYIV_TZ).strftime("%Y-%m-%d %H:%M:%S")

def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v)

# ================= CANDLES =================
@dataclass
class Candle:
    tf_sec: int
    start_ts: float
    open: float
    high: float
    low: float
    close: float
    ticks: int = 0

    def update(self, price: float):
        self.close = price
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.ticks += 1

    @property
    def dir(self) -> str:
        if self.close > self.open:
            return "UP"
        if self.close < self.open:
            return "DOWN"
        return "FLAT"

class InternalCandleBuilder:
    def __init__(self, tf_sec: int):
        self.tf_sec = tf_sec
        self.current: Optional[Candle] = None
        self.last_closed: Optional[Candle] = None

    def _bucket_start(self, ts: float) -> float:
        return ts - (ts % self.tf_sec)

    def on_tick(self, ts: float, price: float):
        b = self._bucket_start(ts)
        if self.current is None:
            self.current = Candle(self.tf_sec, b, price, price, price, price, ticks=1)
            return
        if b == self.current.start_ts:
            self.current.update(price)
            return
        self.last_closed = self.current
        self.current = Candle(self.tf_sec, b, price, price, price, price, ticks=1)

class CandleHistory:
    def __init__(self, maxlen: int = 300):
        self.maxlen = maxlen
        self._items: List[Candle] = []

    def append(self, c: Candle):
        self._items.append(c)
        if len(self._items) > self.maxlen:
            self._items = self._items[-self.maxlen :]

    def items(self) -> List[Candle]:
        return list(self._items)

# ================= INDICATORS =================
def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = mean(values[:period])
    for v in values[period:]:
        e = v * k + e * (1 - k)
    return e

def rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
        return None
    gains, losses = [], []
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(values: List[float], fast=12, slow=26, signal=9):
    if len(values) < slow + signal:
        return None
    macd_line = []
    for i in range(len(values)):
        ef = ema(values[: i + 1], fast)
        es = ema(values[: i + 1], slow)
        if ef is not None and es is not None:
            macd_line.append(ef - es)
    if len(macd_line) < signal:
        return None
    sig = ema(macd_line, signal)
    if sig is None:
        return None
    line = macd_line[-1]
    return {"macd": line, "signal": sig, "hist": line - sig}

def bollinger(values: List[float], period=20, mult=2.0):
    if len(values) < period:
        return None
    window = values[-period:]
    m = mean(window)
    s = stdev(window)
    return {"mid": m, "upper": m + mult * s, "lower": m - mult * s}

def adx(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs, p_dm, m_dm = [], [], []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
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
        di_p = 100 * (p14 / tr14)
        di_m = 100 * (m14 / tr14)
        denom = di_p + di_m
        if denom == 0:
            continue
        dxs.append(100 * abs(di_p - di_m) / denom)
    if len(dxs) < period:
        return None
    a = sum(dxs[:period]) / period
    for v in dxs[period:]:
        a = (a * (period - 1) + v) / period
    return a

# ================= ENGINE =================
class SignalEngine:
    def __init__(self):
        self.symbol = os.getenv("SYMBOL", "EUR_USD")
        self.auto_enabled = os.getenv("AUTO_ENABLED", "true").lower() == "true"
        self.auto_every_sec = int(os.getenv("AUTO_EVERY_SEC", "300"))
        self.min_conf = int(os.getenv("MIN_CONF", "75"))

        self._q = queue.Queue()
        self._lock = threading.Lock()

        self.builder_1m = InternalCandleBuilder(60)
        self.builder_5m = InternalCandleBuilder(300)
        self.hist_1m = CandleHistory()
        self.hist_5m = CandleHistory()

        self._last_closed_1m_ts = None
        self._last_closed_5m_ts = None
        self.last_tick = None

    def compute_signal(self):
        return {"ok": False, "reason": "NOT_ENOUGH_DATA"}

ENGINE = SignalEngine()

# ================= TELEGRAM COMMANDS =================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Бот запущено.\n\n"
        "/status\n/signal\n/auto_on\n/auto_off\n"
        "/subscribe\n/unsubscribe\n/subs"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🟢 Статус: OK")

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Немає даних", parse_mode=ParseMode.HTML)

async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.auto_enabled = True
    await update.message.reply_text("✅ Авто ON")

async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.auto_enabled = False
    await update.message.reply_text("⛔ Авто OFF")

async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Підписка (заглушка)")

async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Відписка (заглушка)")

async def cmd_subs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👥 Підписників: 1")

# ================= MAIN =================
def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("auto_on", cmd_auto_on))
    app.add_handler(CommandHandler("auto_off", cmd_auto_off))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("subs", cmd_subs))

    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
