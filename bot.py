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
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()

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
                log.warning("News fetch HTTP 429")
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

        except Exception as e:
            log.warning("News fetch error: %s", e)
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
        if not self._cache_events:
            return False

        for ev in self._cache_events:
            t_raw = ev.get("time") or ev.get("timestamp") or ev.get("ts")
            if not t_raw:
                continue
            try:
                if isinstance(t_raw, (int, float)):
                    ev_dt = datetime.fromtimestamp(float(t_raw), tz=timezone.utc)
                else:
                    ev_dt = datetime.fromisoformat(str(t_raw).replace("Z", "+00:00"))
                    if ev_dt.tzinfo is None:
                        ev_dt = ev_dt.replace(tzinfo=timezone.utc)
                    ev_dt = ev_dt.astimezone(timezone.utc)
            except Exception:
                continue

            start = ev_dt - timedelta(minutes=self.pre_min)
            end = ev_dt + timedelta(minutes=self.post_min)
            if start <= dt <= end:
                return True

        return False


class OandaPriceStream(threading.Thread):
    def __init__(self, api_key: str, account_id: str, instrument: str, out_q: queue.Queue, practice: bool = True):
        super().__init__(daemon=True)
        self.api_key = api_key
        self.account_id = account_id
        self.instrument = instrument
        self.out_q = out_q
        self.practice = practice
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        base = "https://stream-fxpractice.oanda.com" if self.practice else "https://stream-fxtrade.oanda.com"
        url = f"{base}/v3/accounts/{self.account_id}/pricing/stream"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"instruments": self.instrument}

        backoff = 2
        while not self._stop.is_set():
            try:
                log.info("performing request %s", url)
                with requests.get(url, headers=headers, params=params, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    backoff = 2
                    for line in r.iter_lines():
                        if self._stop.is_set():
                            break
                        if not line:
                            continue
                        try:
                            msg = json.loads(line.decode("utf-8"))
                        except Exception:
                            continue
                        if "heartbeat" in msg:
                            continue
                        if msg.get("type") == "PRICE" and msg.get("instrument") == self.instrument:
                            bids = msg.get("bids") or []
                            asks = msg.get("asks") or []
                            if not bids or not asks:
                                continue
                            bid = float(bids[0]["price"])
                            ask = float(asks[0]["price"])
                            mid = (bid + ask) / 2.0
                            ts = time.time()
                            self.out_q.put({"ts": ts, "bid": bid, "ask": ask, "mid": mid})
            except Exception as e:
                log.warning("Stream error: %s (reconnect in %ss)", e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)


# -------- Indicators (pure python) --------

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
    gains = []
    losses = []
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


def macd(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict[str, float]]:
    if len(values) < slow + signal:
        return None
    macd_line_series = []
    for i in range(len(values)):
        sub = values[: i + 1]
        ef = ema(sub, fast)
        es = ema(sub, slow)
        if ef is None or es is None:
            continue
        macd_line_series.append(ef - es)

    if len(macd_line_series) < signal:
        return None

    sig = ema(macd_line_series, signal)
    if sig is None:
        return None
    line = macd_line_series[-1]
    hist = line - sig
    return {"macd": line, "signal": sig, "hist": hist}


def bollinger(values: List[float], period: int = 20, mult: float = 2.0) -> Optional[Dict[str, float]]:
    if len(values) < period:
        return None
    window = values[-period:]
    m = mean(window)
    s = stdev(window)
    return {"mid": m, "upper": m + mult * s, "lower": m - mult * s}


def adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None

    trs = []
    p_dm = []
    m_dm = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        p_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        m_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)

    if len(trs) < period:
        return None

    # Wilder smoothing
    tr14 = sum(trs[:period])
    p14 = sum(p_dm[:period])
    m14 = sum(m_dm[:period])

    di_plus_list = []
    di_minus_list = []
    dx_list = []

    for i in range(period, len(trs)):
        tr14 = tr14 - (tr14 / period) + trs[i]
        p14 = p14 - (p14 / period) + p_dm[i]
        m14 = m14 - (m14 / period) + m_dm[i]

        if tr14 == 0:
            continue
        di_plus = 100 * (p14 / tr14)
        di_minus = 100 * (m14 / tr14)
        di_plus_list.append(di_plus)
        di_minus_list.append(di_minus)

        denom = di_plus + di_minus
        if denom == 0:
            continue
        dx = 100 * (abs(di_plus - di_minus) / denom)
        dx_list.append(dx)

    if len(dx_list) < period:
        return None

    # ADX = Wilder EMA of DX
    a = sum(dx_list[:period]) / period
    for v in dx_list[period:]:
        a = (a * (period - 1) + v) / period
    return a


# -------- Subscriptions (multiple chats) --------

class Subscribers:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._ids: List[int] = []
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ids = data.get("chat_ids", [])
                self._ids = sorted(list({int(x) for x in ids}))
        except Exception:
            self._ids = []

    def _save(self):
        tmp = {"chat_ids": self._ids}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(tmp, f, ensure_ascii=False, indent=2)

    def add(self, chat_id: int) -> bool:
        with self._lock:
            if chat_id in self._ids:
                return False
            self._ids.append(chat_id)
            self._ids = sorted(list({int(x) for x in self._ids}))
            self._save()
            return True

    def remove(self, chat_id: int) -> bool:
        with self._lock:
            if chat_id not in self._ids:
                return False
            self._ids = [x for x in self._ids if x != chat_id]
            self._save()
            return True

    def list(self) -> List[int]:
        with self._lock:
            return list(self._ids)


class SignalEngine:
    def __init__(self):
        self.symbol = os.getenv("SYMBOL", "EUR_USD")
        self.auto_enabled = os.getenv("AUTO_ENABLED", "true").lower() == "true"
        self.auto_every_sec = int(os.getenv("AUTO_EVERY_SEC", "300"))
        self.min_conf = int(os.getenv("MIN_CONF", "75"))

        self.tf1 = int(os.getenv("CANDLE_TF_1M_SEC", "60"))
        self.tf5 = int(os.getenv("CANDLE_TF_5M_SEC", "300"))

        self._q = queue.Queue(maxsize=20000)
        self._lock = threading.Lock()

        self.builder_1m = InternalCandleBuilder(self.tf1)
        self.builder_5m = InternalCandleBuilder(self.tf5)

        self.hist_1m = CandleHistory(maxlen=400)
        self.hist_5m = CandleHistory(maxlen=400)

        self._last_closed_1m_ts: Optional[float] = None
        self._last_closed_5m_ts: Optional[float] = None

        self.last_tick: Optional[Dict[str, float]] = None
        self.news = NewsLock()

        self._stream: Optional[OandaPriceStream] = None

    def start_stream(self):
        api_key = (os.getenv("OANDA_API_KEY") or "").strip()
        account_id = (os.getenv("OANDA_ACCOUNT_ID") or "").strip()
        env = (os.getenv("OANDA_ENV") or "practice").strip().lower()

        if not api_key or not account_id:
    raise RuntimeError("OANDA_API_KEY / OANDA_ACCOUNT_ID missing in Railway variables")

        practice = (env == "practice")
        self._stream = OandaPriceStream(api_key, account_id, self.symbol, self._q, practice=practice)
        self._stream.start()

        t = threading.Thread(target=self._pump_ticks, daemon=True)
        t.start()

    def _pump_ticks(self):
        while True:
            item = self._q.get()
            ts = float(item["ts"])
            mid = float(item["mid"])
            with self._lock:
                self.last_tick = item

                self.builder_1m.on_tick(ts, mid)
                self.builder_5m.on_tick(ts, mid)

                c1 = self.builder_1m.last_closed
                if c1 and (self._last_closed_1m_ts is None or c1.start_ts != self._last_closed_1m_ts):
                    self._last_closed_1m_ts = c1.start_ts
                    self.hist_1m.append(c1)

                c5 = self.builder_5m.last_closed
                if c5 and (self._last_closed_5m_ts is None or c5.start_ts != self._last_closed_5m_ts):
                    self._last_closed_5m_ts = c5.start_ts
                    self.hist_5m.append(c5)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            last = self.last_tick
            c1 = self.builder_1m.last_closed
            c5 = self.builder_5m.last_closed
            h1 = self.hist_1m.items()
            h5 = self.hist_5m.items()
        return {"last": last, "c1": c1, "c5": c5, "h1": h1, "h5": h5}

    def _indicator_pack(self, candles: List[Candle]) -> Dict[str, Any]:
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        ema20 = ema(closes, 20)
        ema50 = ema(closes, 50)
        r = rsi(closes, 14)
        m = macd(closes, 12, 26, 9)
        b = bollinger(closes, 20, 2.0)
        a = adx(highs, lows, closes, 14)

        last_close = closes[-1] if closes else None

        return {
            "ema20": ema20, "ema50": ema50,
            "rsi": r,
            "macd": m,
            "bb": b,
            "adx": a,
            "last_close": last_close
        }

    def compute_signal(self) -> Dict[str, Any]:
        dt = now_utc()
        if self.news.in_news_window(dt):
            return {"ok": False, "reason": "NEWS_LOCK"}

        snap = self.snapshot()
        last = snap["last"]
        h1: List[Candle] = snap["h1"]
        h5: List[Candle] = snap["h5"]

        if not last or len(h5) < 60 or len(h1) < 60:
            return {"ok": False, "reason": "NOT_ENOUGH_DATA"}

        spread = float(last["ask"]) - float(last["bid"])

        p1 = self._indicator_pack(h1)
        p5 = self._indicator_pack(h5)

        # Votes / scoring
        weights = {
            "ema": 25,
            "rsi": 15,
            "macd": 20,
            "adx": 15,
            "bb": 10,
            "pa": 15
        }

        buy = 0
        sell = 0
        used = []

        # EMA trend (5m priority)
        if p5["ema20"] is not None and p5["ema50"] is not None:
            if p5["ema20"] > p5["ema50"]:
                buy += weights["ema"]; used.append("EMA20>EMA50 (5m)")
            elif p5["ema20"] < p5["ema50"]:
                sell += weights["ema"]; used.append("EMA20<EMA50 (5m)")

        # RSI
        if p5["rsi"] is not None:
            if p5["rsi"] >= 55:
                buy += weights["rsi"]; used.append(f"RSI {p5['rsi']:.1f} (bull)")
            elif p5["rsi"] <= 45:
                sell += weights["rsi"]; used.append(f"RSI {p5['rsi']:.1f} (bear)")

        # MACD histogram
        if p5["macd"] is not None:
            hist = p5["macd"]["hist"]
            if hist > 0:
                buy += weights["macd"]; used.append(f"MACD hist {hist:.6f} (+)")
            elif hist < 0:
                sell += weights["macd"]; used.append(f"MACD hist {hist:.6f} (-)")

        # ADX (trend strength) — додає вагу в сторону того, що домінує
        if p5["adx"] is not None:
            if p5["adx"] >= 20:
                if buy > sell:
                    buy += weights["adx"]; used.append(f"ADX {p5['adx']:.1f} (trend)")
                elif sell > buy:
                    sell += weights["adx"]; used.append(f"ADX {p5['adx']:.1f} (trend)")
            else:
                used.append(f"ADX {p5['adx']:.1f} (weak)")

        # Bollinger
        if p5["bb"] is not None and p5["last_close"] is not None:
            bb = p5["bb"]
            c = p5["last_close"]
            if c > bb["mid"]:
                buy += weights["bb"]; used.append("BB: close>mid")
            elif c < bb["mid"]:
                sell += weights["bb"]; used.append("BB: close<mid")

        # Price action last 3 candles on 1m
        last3 = h1[-3:]
        dirs = [c.dir for c in last3]
        if dirs.count("UP") == 3:
            buy += weights["pa"]; used.append("PA: 3x UP (1m)")
        elif dirs.count("DOWN") == 3:
            sell += weights["pa"]; used.append("PA: 3x DOWN (1m)")

        total = buy + sell
        conf = int(round((max(buy, sell) / max(1, sum(weights.values()))) * 100))

        # Risk proxy: spread + weak ADX
        risk = 20
        if spread > 0.00012:
            risk += 10
        if p5["adx"] is not None and p5["adx"] < 18:
            risk += 10
        risk = max(0, min(100, risk))

        if buy == sell or conf < self.min_conf:
            return {
                "ok": False,
                "reason": "WEAK",
                "conf": conf,
                "risk": risk,
                "spread": spread,
                "used": used,
                "p5": p5
            }

        direction = "BUY" if buy > sell else "SELL"
        return {
            "ok": True,
            "direction": direction,
            "conf": conf,
            "risk": risk,
            "spread": spread,
            "used": used,
            "p5": p5
        }


ENGINE = SignalEngine()

SUBS = Subscribers(os.getenv("SUBSCRIBERS_FILE", "/home/ec2-user/subscribers.json"))


def fmt_signal(sig: Dict[str, Any]) -> str:
    dt = now_utc()
    t = fmt_kyiv(dt)

    if sig.get("reason") == "NEWS_LOCK":
        return (
            f"⛔ <b>NEWS LOCK</b>\n"
            f"🕒 <b>Kyiv:</b> {t}\n"
            f"Сигнали заблоковані через новини."
        )

    if sig.get("reason") == "NOT_ENOUGH_DATA":
        return (
            f"⏳ <b>Немає даних</b>\n"
            f"🕒 <b>Kyiv:</b> {t}\n"
            f"Потрібно ~60 закритих свічок на 1m і 5m (приблизно 10–60 хв після старту)."
        )

    if not sig.get("ok"):
        conf = sig.get("conf", 0)
        risk = sig.get("risk", 0)
        used = sig.get("used", [])
        used_txt = "\n".join(f"• {u}" for u in used[:8]) if used else "• немає підтверджень"
        return (
            f"⚠️ <b>Сигналу немає</b>\n"
            f"🕒 <b>Kyiv:</b> {t}\n"
            f"📊 <b>Підтвердження:</b> {conf}%\n"
            f"⚠️ <b>Ризик:</b> {risk}%\n"
            f"\n<b>Що бачить бот:</b>\n{used_txt}"
        )

    direction = sig["direction"]
    conf = sig["conf"]
    risk = sig["risk"]
    p5 = sig.get("p5", {})
    used = sig.get("used", [])

    arrow = "🔼" if direction == "BUY" else "🔻"
    action = "BUY" if direction == "BUY" else "SELL"

    rsi_v = p5.get("rsi")
    adx_v = p5.get("adx")
    macd_v = p5.get("macd", {})
    bb = p5.get("bb", {})
    ema20_v = p5.get("ema20")
    ema50_v = p5.get("ema50")

    details = []
    if ema20_v is not None and ema50_v is not None:
        details.append(f"EMA20/50: {ema20_v:.5f} / {ema50_v:.5f}")
    if rsi_v is not None:
        details.append(f"RSI(14): {rsi_v:.1f}")
    if macd_v:
        details.append(f"MACD hist: {macd_v['hist']:.6f}")
    if adx_v is not None:
        details.append(f"ADX(14): {adx_v:.1f}")
    if bb:
        details.append(f"BB mid: {bb['mid']:.5f}")

    used_txt = "\n".join(f"• {u}" for u in used[:10]) if used else "• —"

    tip = "🔔 Рекомендовано входити негайно" if conf >= 85 and risk <= 30 else "⏳ Краще дочекатися закриття свічки"

    return (
        f"{arrow} <b>{action} EUR/USD</b>\n"
        f"🕒 <b>Kyiv:</b> {t}\n"
        f"📊 <b>Підтвердження:</b> {conf}%\n"
        f"⚠️ <b>Ризик:</b> {risk}%\n"
        f"{tip}\n"
        f"\n<b>Індикатори (5m):</b>\n" + "\n".join(f"• {d}" for d in details) +
        f"\n\n<b>Підтвердження (логіка):</b>\n{used_txt}"
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "✅ Бот запущено.\n\n"
        "Команди:\n"
        "/status — стан ринку/свічок\n"
        "/signal — сигнал зараз\n"
        "/auto_on — авто ON\n"
        "/auto_off — авто OFF\n"
        "/subscribe — отримувати автосигнали (для брата теж)\n"
        "/unsubscribe — відписатися\n"
        "/subs — список підписників (count)\n"
    )
    await update.message.reply_text(msg)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    snap = ENGINE.snapshot()
    dt = now_utc()
    locked = ENGINE.news.in_news_window(dt)
    h1 = snap["h1"]
    h5 = snap["h5"]
    last = snap["last"]

    text = [
        f"Стан: {'NEWS_LOCK' if locked else 'OK'}",
        f"Kyiv: {fmt_kyiv(dt)}",
        f"Авто: {'ON' if ENGINE.auto_enabled else 'OFF'} (кожні {ENGINE.auto_every_sec}s)",
        f"MIN_CONF: {ENGINE.min_conf}%",
        f"Свічки: 1m={len(h1)}  5m={len(h5)}",
    ]
    if last:
        text.append(f"Tick: bid={last['bid']:.5f} ask={last['ask']:.5f}")
    await update.message.reply_text("\n".join(text))


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sig = ENGINE.compute_signal()
    await update.message.reply_text(fmt_signal(sig), parse_mode=ParseMode.HTML)


async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.auto_enabled = True
    await update.message.reply_text("✅ Автосигнали: ON")


async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ENGINE.auto_enabled = False
    await update.message.reply_text("⛔ Автосигнали: OFF")


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    added = SUBS.add(chat_id)
    if added:
        await update.message.reply_text("✅ Підписано. Тепер цей чат отримуватиме автосигнали.")
    else:
        await update.message.reply_text("ℹ️ Цей чат вже підписаний на автосигнали.")


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    removed = SUBS.remove(chat_id)
    if removed:
        await update.message.reply_text("✅ Відписано. Цей чат більше не отримуватиме автосигнали.")
    else:
        await update.message.reply_text("ℹ️ Цей чат не був підписаний.")


async def cmd_subs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ids = SUBS.list()
    await update.message.reply_text(f"👥 Підписників: {len(ids)}")


async def auto_job(context: ContextTypes.DEFAULT_TYPE):
    if not ENGINE.auto_enabled:
        return

    sig = ENGINE.compute_signal()
    if not sig.get("ok"):
        return

    msg = fmt_signal(sig)
    for cid in SUBS.list():
        try:
            await context.bot.send_message(chat_id=cid, text=msg, parse_mode=ParseMode.HTML)
        except Exception as e:
            log.warning("Send failed to %s: %s", cid, e)


def main():
    token = (
        os.getenv("TELEGRAM_BOT_TOKEN")
        or os.getenv("BOT_TOKEN")
        or ""
    ).strip()

    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / BOT_TOKEN missing in Railway variables")

    ENGINE.start_stream()

    app = Application.builder().token(token).build()

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("auto_on", cmd_auto_on))
    app.add_handler(CommandHandler("auto_off", cmd_auto_off))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("subs", cmd_subs))

    app.job_queue.run_repeating(auto_job, interval=ENGINE.auto_every_sec, first=10, name="auto_job")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
