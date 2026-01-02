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
import pytz

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("telegram_oanda_bot")

KYIV_TZ = pytz.timezone("Europe/Kyiv")


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


# ---------------- NEWS LOCK ----------------

class NewsLock:
    def __init__(self):
        self.enabled = os.getenv("NEWS_ENABLED", "true").lower() == "true"
        self.url = (os.getenv("NEWS_URL") or "").strip()
        self.refresh_sec = int(os.getenv("NEWS_REFRESH_SEC", "600"))
        self.pre_min = int(os.getenv("NEWS_PRE_MIN", "15"))
        self.post_min = int(os.getenv("NEWS_POST_MIN", "15"))
        self.lock_on_error_min = int(os.getenv("NEWS_LOCK_ON_ERROR_MIN", "10"))
        self.lock_if_429 = os.getenv("NEWS_LOCK_IF_429", "true").lower() == "true"

        self._cache_events: List[Dict[str, Any]] = []
        self._cache_ts: float = 0.0
        self._lock_until: Optional[datetime] = None
        self._cooldown_until: float = 0.0

    def _set_lock(self, minutes: int, reason: str):
        until = now_utc() + timedelta(minutes=minutes)
        self._lock_until = until
        log.warning("News lock enabled for %s min (%s) until %s", minutes, reason, fmt_kyiv(until))

    def _fetch_events(self) -> List[Dict[str, Any]]:
        if not self.url:
            return []
        if time.time() < self._cooldown_until:
            return self._cache_events

        try:
            r = requests.get(self.url, timeout=10)
            if r.status_code == 429:
                if self.lock_if_429:
                    self._set_lock(self.lock_on_error_min, "HTTP 429")
                self._cooldown_until = time.time() + 60
                return self._cache_events

            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "events" in data:
                data = data["events"]
            if not isinstance(data, list):
                return self._cache_events
            return data

        except Exception:
            self._set_lock(self.lock_on_error_min, "fetch error")
            self._cooldown_until = time.time() + 60
            return self._cache_events

    def refresh_if_needed(self):
        if not self.enabled:
            return
        if time.time() - self._cache_ts < self.refresh_sec:
            return
        self._cache_events = self._fetch_events()
        self._cache_ts = time.time()

    def in_news_window(self, dt: datetime) -> bool:
        if not self.enabled:
            return False

        if self._lock_until and dt < self._lock_until:
            return True

        self.refresh_if_needed()
        for ev in self._cache_events:
            t_raw = ev.get("time") or ev.get("timestamp") or ev.get("ts")
            if not t_raw:
                continue
            try:
                ev_dt = datetime.fromtimestamp(float(t_raw), tz=timezone.utc)
            except Exception:
                continue

            if ev_dt - timedelta(minutes=self.pre_min) <= dt <= ev_dt + timedelta(minutes=self.post_min):
                return True

        return False


# ---------------- SIGNAL ENGINE ----------------
# (ДАЛІ ЙДЕ ТВОЯ ЛОГІКА БЕЗ ЗМІН — Я ЇЇ НЕ ЧІПАВ)

# ... (весь SignalEngine, Subscribers, fmt_signal, команди — БЕЗ ЗМІН)

def main():
    token = (os.getenv("BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("BOT_TOKEN missing")

    ENGINE.start_stream()

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("auto_on", cmd_auto_on))
    app.add_handler(CommandHandler("auto_off", cmd_auto_off))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("subs", cmd_subs))

    app.job_queue.run_repeating(auto_job, interval=ENGINE.auto_every_sec, first=10)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
