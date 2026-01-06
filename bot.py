# ========================= IMPORTS =========================
import os
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
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)
import os, sys, atexit

LOCK_FILE = "/tmp/telegram_bot.lock"

if os.path.exists(LOCK_FILE):
    print("Bot already running, exit")
    sys.exit(0)

with open(LOCK_FILE, "w") as f:
    f.write(str(os.getpid()))

atexit.register(lambda: os.remove(LOCK_FILE) if os.path.exists(LOCK_FILE) else None)

# ========================= INIT =========================
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("telegram_oanda_bot")

KYIV_TZ = pytz.timezone("Europe/Kyiv")

# ========================= TIME =========================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def fmt_kyiv(dt_utc: datetime) -> str:
    return dt_utc.astimezone(KYIV_TZ).strftime("%Y-%m-%d %H:%M:%S")

# ========================= MATH =========================
def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v)

# ========================= CANDLES =========================
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
        self.high = max(self.high, price)
        self.low = min(self.low, price)
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
    def __init__(self, maxlen: int = 400):
        self.maxlen = maxlen
        self._items: List[Candle] = []

    def append(self, c: Candle):
        self._items.append(c)
        if len(self._items) > self.maxlen:
            self._items = self._items[-self.maxlen:]

    def items(self) -> List[Candle]:
        return list(self._items)

# ========================= INDICATORS =========================
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
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(values: List[float], fast=12, slow=26, signal=9) -> Optional[Dict[str, float]]:
    if len(values) < slow + signal:
        return None
    macd_line = []
    for i in range(len(values)):
        ef = ema(values[: i + 1], fast)
        es = ema(values[: i + 1], slow)
        if ef is None or es is None:
            continue
        macd_line.append(ef - es)
    sig = ema(macd_line, signal)
    if sig is None:
        return None
    line = macd_line[-1]
    return {"macd": line, "signal": sig, "hist": line - sig}

def bollinger(values: List[float], period=20, mult=2.0) -> Optional[Dict[str, float]]:
    if len(values) < period:
        return None
    w = values[-period:]
    m = mean(w)
    s = stdev(w)
    return {"mid": m, "upper": m + mult * s, "lower": m - mult * s}

def adx(highs, lows, closes, period=14) -> Optional[float]:
    if len(closes) < period + 1:
        return None

    trs, p_dm, m_dm = [], [], []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        p_dm.append(up if up > dn and up > 0 else 0)
        m_dm.append(dn if dn > up and dn > 0 else 0)

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
        if di_p + di_m == 0:
            continue
        dxs.append(100 * abs(di_p - di_m) / (di_p + di_m))

    if len(dxs) < period:
        return None

    a = sum(dxs[:period]) / period
    for v in dxs[period:]:
        a = (a * (period - 1) + v) / period
    return a

# ========================= SUBSCRIBERS =========================
class Subscribers:
    def __init__(self, path: str):
        self.path = path
        self.ids: List[int] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.ids = json.load(f).get("chat_ids", [])

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"chat_ids": self.ids}, f, indent=2)

    def add(self, cid: int) -> bool:
        if cid in self.ids:
            return False
        self.ids.append(cid)
        self._save()
        return True

    def remove(self, cid: int) -> bool:
        if cid not in self.ids:
            return False
        self.ids.remove(cid)
        self._save()
        return True

    def list(self) -> List[int]:
        return list(self.ids)

SUBS = Subscribers(os.getenv("SUBSCRIBERS_FILE", "/app/subscribers.json"))

# ========================= SIGNAL TEXT =========================
def build_signal_text(
    arrow, action, t, conf, risk,
    ema20_v, ema50_v,
    rsi_v, macd_v, adx_v, bb,
    used
) -> str:
    details = []

    if ema20_v is not None and ema50_v is not None:
        details.append(f"EMA20/50: {ema20_v:.5f} / {ema50_v:.5f}")

    if rsi_v is not None:
        if rsi_v > 72:
            details.append(f"❌ <b>RSI(14): {rsi_v:.1f} — НЕ ВХОДИТИ</b>")
        else:
            details.append(f"<b>RSI(14): {rsi_v:.1f}</b>")

    if macd_v:
        details.append(f"MACD hist: {macd_v['hist']:.6f}")

    if adx_v is not None:
        if adx_v > 35:
            details.append(f"❌ <b>ADX(14): {adx_v:.1f} — НЕ ВХОДИТИ</b>")
        else:
            details.append(f"<b>ADX(14): {adx_v:.1f}</b>")

    if bb:
        details.append(f"BB mid: {bb['mid']:.5f}")

    used_txt = "\n".join(f"• {u}" for u in used[:10]) if used else "—"

    tip = (
        "🔔 Рекомендовано входити негайно"
        if conf >= 85 and risk <= 30 and not (
            (rsi_v is not None and rsi_v > 72) or
            (adx_v is not None and adx_v > 35)
        )
        else "❌ ПРОПУСТИТИ УГОДУ"
    )

    return (
        f"{arrow} <b>{action} EUR/USD</b>\n"
        f"🕒 <b>Kyiv:</b> {t}\n"
        f"📊 <b>Підтвердження:</b> {conf}%\n"
        f"⚠️ <b>Ризик:</b> {risk}%\n"
        f"{tip}\n\n"
        f"<b>Індикатори (5m):</b>\n"
        + "\n".join(f"• {d}" for d in details)
        + f"\n\n<b>Підтвердження (логіка):</b>\n{used_txt}"
    )

# ========================= TELEGRAM COMMANDS =========================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Бот запущено.\n\n"
        "Команди:\n"
        "/status — стан ринку\n"
        "/signal — сигнал зараз\n"
        "/auto_on — авто ON\n"
        "/auto_off — авто OFF\n"
        "/subscribe — отримувати автосигнали\n"
        "/unsubscribe — відписатись\n"
        "/subs — кількість підписників",
        parse_mode=ParseMode.HTML
    )

async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if SUBS.add(update.effective_chat.id):
        await update.message.reply_text("✅ Підписано")
    else:
        await update.message.reply_text("ℹ️ Ви вже підписані")

async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if SUBS.remove(update.effective_chat.id):
        await update.message.reply_text("❌ Відписано")
    else:
        await update.message.reply_text("ℹ️ Ви не були підписані")

async def cmd_subs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"👥 Підписників: {len(SUBS.list())}")

# ========================= MAIN =========================
def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("subs", cmd_subs))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("auto_on", cmd_auto_on))
    app.add_handler(CommandHandler("auto_off", cmd_auto_off))

    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )

if __name__ == "__main__":
    main()
