import asyncio
import logging
import random
import json
import os
from datetime import datetime, timedelta
from typing import Optional
import math
 
# ── dependencies ──────────────────────────────────────────────────────────────
# pip install python-telegram-bot==20.7 aiohttp
 
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
import aiohttp
 
# ── configuration ─────────────────────────────────────────────────────────────
 
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
 
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY"]
 
INITIAL_BALANCE = 1000.0   # default starting balance $
STAKE_PERCENT   = 0.10     # 10% of balance per trade
EXPIRY_MINUTES  = 2
 
AUTO_SIGNAL_INTERVAL = 180  # seconds between auto signals (3 min)
 
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("TradingBot")
 
# ── per-user state ────────────────────────────────────────────────────────────
 
def default_state() -> dict:
    return {
        "balance":      INITIAL_BALANCE,
        "auto_signals": False,
        "trades":       [],          # list of completed trades
        "pending":      None,        # active signal waiting for result
    }
 
user_states: dict[int, dict] = {}
 
def get_state(uid: int) -> dict:
    if uid not in user_states:
        user_states[uid] = default_state()
    return user_states[uid]
 
# ── market data (via Yahoo Finance / fallback synthetic) ──────────────────────
 
YAHOO_SYMBOLS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
}
 
async def fetch_candles(pair: str, interval: str = "1m", count: int = 20) -> list[dict]:
    """
    Fetch OHLCV candles from Yahoo Finance.
    Falls back to synthetic data if the request fails.
    interval: '1m' or '5m'
    """
    symbol = YAHOO_SYMBOLS[pair]
    range_  = "1d" if interval == "1m" else "5d"
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval={interval}&range={range_}"
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    raise ValueError(f"HTTP {resp.status}")
                data = await resp.json()
                result = data["chart"]["result"][0]
                timestamps = result["timestamp"]
                q = result["indicators"]["quote"][0]
                opens   = q["open"]
                highs   = q["high"]
                lows    = q["low"]
                closes  = q["close"]
                volumes = q.get("volume", [0] * len(opens))
                candles = []
                for i in range(len(timestamps)):
                    if None in (opens[i], highs[i], lows[i], closes[i]):
                        continue
                    candles.append({
                        "t": timestamps[i],
                        "o": opens[i],
                        "h": highs[i],
                        "l": lows[i],
                        "c": closes[i],
                        "v": volumes[i] or 0,
                    })
                return candles[-count:] if len(candles) >= count else candles
    except Exception as e:
        log.warning(f"fetch_candles failed ({pair} {interval}): {e} — using synthetic")
        return _synthetic_candles(pair, count)
 
 
def _synthetic_candles(pair: str, count: int) -> list[dict]:
    """Generate plausible synthetic OHLCV data for testing."""
    seed_price = {"EUR/USD": 1.0850, "GBP/USD": 1.2700, "USD/JPY": 149.50}[pair]
    pip = 0.0001 if "JPY" not in pair else 0.01
    candles = []
    price = seed_price
    now = int(datetime.utcnow().timestamp())
    for i in range(count):
        change = random.gauss(0, pip * 8)
        o = price
        c = o + change
        h = max(o, c) + abs(random.gauss(0, pip * 3))
        l = min(o, c) - abs(random.gauss(0, pip * 3))
        candles.append({"t": now - (count - i) * 60, "o": o, "h": h, "l": l, "c": c, "v": random.randint(100, 2000)})
        price = c
    return candles
 
# ── technical indicators ──────────────────────────────────────────────────────
 
def ema(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return [sum(values) / len(values)] * len(values)
    k = 2 / (period + 1)
    result = [sum(values[:period]) / period]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    # pad front
    pad = [result[0]] * (len(values) - len(result))
    return pad + result
 
def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas[-period:]]
    losses = [abs(min(d, 0)) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)
 
def macd(closes: list[float]) -> tuple[float, float, float]:
    """Returns (macd_line, signal_line, histogram)"""
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_line = [ema12[i] - ema26[i] for i in range(len(closes))]
    signal    = ema(macd_line, 9)
    hist      = [macd_line[i] - signal[i] for i in range(len(closes))]
    return macd_line[-1], signal[-1], hist[-1]
 
def stochastic(candles: list[dict], k_period: int = 14) -> tuple[float, float]:
    if len(candles) < k_period:
        return 50.0, 50.0
    slice_ = candles[-k_period:]
    h = max(c["h"] for c in slice_)
    l = min(c["l"] for c in slice_)
    if h == l:
        return 50.0, 50.0
    k = (candles[-1]["c"] - l) / (h - l) * 100
    # %D = 3-period SMA of %K (approximate)
    ks = []
    for j in range(max(0, len(candles) - k_period - 2), len(candles)):
        sl = candles[max(0, j - k_period + 1): j + 1]
        hh = max(c["h"] for c in sl)
        ll = min(c["l"] for c in sl)
        ks.append((candles[j]["c"] - ll) / (hh - ll) * 100 if hh != ll else 50.0)
    d = sum(ks[-3:]) / min(3, len(ks))
    return k, d
 
def bollinger_bands(closes: list[float], period: int = 20, std_dev: float = 2.0):
    if len(closes) < period:
        mid = closes[-1]
        return mid, mid, mid
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(variance)
    return mid, mid + std_dev * std, mid - std_dev * std
 
def adx(candles: list[dict], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 25.0
    trs, pdm, ndm = [], [], []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["h"], candles[i]["l"], candles[i-1]["c"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
        up = h - candles[i-1]["h"]
        down = candles[i-1]["l"] - l
        pdm.append(up if up > down and up > 0 else 0)
        ndm.append(down if down > up and down > 0 else 0)
    def smooth(lst): return sum(lst[-period:]) / period
    tr_s  = smooth(trs)
    pdm_s = smooth(pdm)
    ndm_s = smooth(ndm)
    if tr_s == 0:
        return 25.0
    pdi = 100 * pdm_s / tr_s
    ndi = 100 * ndm_s / tr_s
    if pdi + ndi == 0:
        return 25.0
    dx = 100 * abs(pdi - ndi) / (pdi + ndi)
    return dx
 
def price_action_signal(candles: list[dict]) -> int:
    """Analyse last 3 candles. Returns +1 (bull), -1 (bear), 0 (neutral)."""
    if len(candles) < 3:
        return 0
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    bull = sum([
        c3["c"] > c3["o"],          # current bullish
        c2["c"] > c2["o"],          # prev bullish
        c3["c"] > c2["h"],          # breakout up
        c3["l"] > c2["l"],          # higher low
    ])
    bear = sum([
        c3["c"] < c3["o"],
        c2["c"] < c2["o"],
        c3["c"] < c2["l"],
        c3["h"] < c2["h"],
    ])
    if bull >= 3: return 1
    if bear >= 3: return -1
    return 0
 
# ── signal engine ─────────────────────────────────────────────────────────────
 
async def analyse_pair(pair: str) -> Optional[dict]:
    """
    Full multi-timeframe analysis. Returns signal dict or None if no signal.
    """
    m1 = await fetch_candles(pair, "1m", 30)
    m5 = await fetch_candles(pair, "5m", 30)
 
    if len(m1) < 20 or len(m5) < 20:
        return None
 
    closes_m1 = [c["c"] for c in m1]
    closes_m5 = [c["c"] for c in m5]
 
    # ── M5 trend filter ───────────────────────────────────────────────────────
    ema20_m5 = ema(closes_m5, 20)
    m5_trend = 1 if closes_m5[-1] > ema20_m5[-1] else -1
 
    # ── M1 indicators ─────────────────────────────────────────────────────────
    ema8  = ema(closes_m1, 8)
    ema21 = ema(closes_m1, 21)
    ema_signal = 1 if ema8[-1] > ema21[-1] else -1
 
    rsi_val = rsi(closes_m1, 14)
    rsi_signal = 1 if rsi_val < 45 else (-1 if rsi_val > 55 else 0)
 
    macd_line, signal_line, hist = macd(closes_m1)
    macd_signal = 1 if (macd_line > signal_line and hist > 0) else (
                 -1 if (macd_line < signal_line and hist < 0) else 0)
 
    k, d = stochastic(m1, 14)
    stoch_signal = 1 if (k < 30 and k > d) else (-1 if (k > 70 and k < d) else 0)
 
    mid, upper, lower = bollinger_bands(closes_m1, 20)
    price = closes_m1[-1]
    bb_signal = 1 if price < lower else (-1 if price > upper else 0)
 
    adx_val = adx(m1, 14)
    trend_strong = adx_val > 20
 
    pa_signal = price_action_signal(m1)
 
    # ── score aggregation ─────────────────────────────────────────────────────
    signals = [ema_signal, rsi_signal, macd_signal, stoch_signal, bb_signal, pa_signal]
    bull_score = sum(1 for s in signals if s == 1)
    bear_score = sum(1 for s in signals if s == -1)
 
    direction = None
    if bull_score >= 4 and m5_trend == 1 and trend_strong:
        direction = "UP"
        score = bull_score
    elif bear_score >= 4 and m5_trend == -1 and trend_strong:
        direction = "DOWN"
        score = bear_score
    else:
        return None  # not enough confluence
 
    # probability based on confluence + ADX
    base_prob = 55 + (score - 4) * 6 + min(adx_val - 20, 20) * 0.5
    probability = min(round(base_prob + random.uniform(-2, 2), 1), 92.0)
 
    risk = "Low" if probability >= 75 else ("Medium" if probability >= 65 else "High")
    risk_pct = round(100 - probability, 1)
 
    return {
        "pair":        pair,
        "direction":   direction,
        "probability": probability,
        "risk":        risk,
        "risk_pct":    risk_pct,
        "adx":         round(adx_val, 1),
        "rsi":         round(rsi_val, 1),
        "score":       score,
        "price":       price,
    }
 
 
async def best_signal() -> Optional[dict]:
    """Analyse all pairs and return the best signal."""
    tasks = [analyse_pair(p) for p in PAIRS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid = [r for r in results if isinstance(r, dict)]
    if not valid:
        return None
    return max(valid, key=lambda x: x["probability"])
 
# ── message formatting ────────────────────────────────────────────────────────
 
def format_signal(sig: dict, stake: float) -> str:
    arrow = "🔼 UP" if sig["direction"] == "UP" else "🔻 DOWN"
    risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[sig["risk"]]
    return (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📡 *SIGNAL DETECTED*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💱 Pair: *{sig['pair']}*\n"
        f"📈 Direction: *{arrow}*\n"
        f"💵 Stake: *${stake:.2f}* (10%)\n"
        f"⚠️ Risk: {risk_emoji} *{sig['risk']}* ({sig['risk_pct']}%)\n"
        f"📊 Probability: *{sig['probability']}%*\n"
        f"⏱ Expiry: *{EXPIRY_MINUTES} minutes*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔬 RSI: {sig['rsi']} | ADX: {sig['adx']}\n"
        f"💰 Price: {sig['price']:.5f}\n"
    )
 
 
def format_stats(state: dict) -> str:
    trades = state["trades"]
    wins   = sum(1 for t in trades if t["result"] == "win")
    losses = sum(1 for t in trades if t["result"] == "loss")
    total  = len(trades)
    winrate = (wins / total * 100) if total else 0
    pnl    = sum(t["pnl"] for t in trades)
    return (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *STATISTICS*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Balance: *${state['balance']:.2f}*\n"
        f"📈 Trades:  {total}  |  ✅ {wins}  |  ❌ {losses}\n"
        f"🎯 Win Rate: *{winrate:.1f}%*\n"
        f"💵 Total P&L: *{'+'if pnl>=0 else ''}{pnl:.2f}$*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━"
    )
 
# ── keyboards ─────────────────────────────────────────────────────────────────
 
def main_keyboard(auto_on: bool) -> InlineKeyboardMarkup:
    auto_label = "🔴 Auto: OFF" if not auto_on else "🟢 Auto: ON"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔮 Forecast", callback_data="forecast")],
        [InlineKeyboardButton(auto_label,    callback_data="toggle_auto")],
        [
            InlineKeyboardButton("📊 Stats",   callback_data="stats"),
            InlineKeyboardButton("💰 Balance", callback_data="balance"),
        ],
        [InlineKeyboardButton("❓ Help",      callback_data="help")],
    ])
 
def result_keyboard(trade_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ WIN",  callback_data=f"result:win:{trade_id}"),
        InlineKeyboardButton("❌ LOSS", callback_data=f"result:loss:{trade_id}"),
    ]])
 
# ── handlers ──────────────────────────────────────────────────────────────────
 
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid   = update.effective_user.id
    state = get_state(uid)
    name  = update.effective_user.first_name or "Trader"
    text  = (
        f"👋 Welcome back, *{name}*!\n\n"
        f"🤖 *Binary Signal Bot* — 2-min expiry\n"
        f"💰 Balance: *${state['balance']:.2f}*\n\n"
        f"Tap *Forecast* for an instant signal\nor enable *Auto* for hands-free trading."
    )
    await update.message.reply_text(
        text,
        parse_mode="Markdown",
        reply_markup=main_keyboard(state["auto_signals"]),
    )
 
 
async def send_signal(uid: int, ctx: ContextTypes.DEFAULT_TYPE, chat_id: int):
    state = get_state(uid)
    stake = round(state["balance"] * STAKE_PERCENT, 2)
 
    await ctx.bot.send_chat_action(chat_id, "typing")
    sig = await best_signal()
 
    if sig is None:
        await ctx.bot.send_message(
            chat_id,
            "⏳ *Market is quiet.* No strong signal right now — waiting for confluence...",
            parse_mode="Markdown",
            reply_markup=main_keyboard(state["auto_signals"]),
        )
        return
 
    trade_id = f"{uid}_{int(datetime.utcnow().timestamp())}"
    state["pending"] = {
        "id":        trade_id,
        "pair":      sig["pair"],
        "direction": sig["direction"],
        "stake":     stake,
        "prob":      sig["probability"],
        "time":      datetime.utcnow().isoformat(),
    }
 
    msg = format_signal(sig, stake)
    msg += f"\n⏰ Closes in {EXPIRY_MINUTES} min — record your result below:"
 
    await ctx.bot.send_message(
        chat_id,
        msg,
        parse_mode="Markdown",
        reply_markup=result_keyboard(trade_id),
    )
 
 
async def cb_forecast(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await send_signal(q.from_user.id, ctx, q.message.chat_id)
 
 
async def cb_toggle_auto(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q     = update.callback_query
    uid   = q.from_user.id
    state = get_state(uid)
    await q.answer()
 
    state["auto_signals"] = not state["auto_signals"]
 
    if state["auto_signals"]:
        ctx.job_queue.run_repeating(
            lambda c: asyncio.create_task(send_signal(uid, c, q.message.chat_id)),
            interval=AUTO_SIGNAL_INTERVAL,
            first=AUTO_SIGNAL_INTERVAL,
            name=f"auto_{uid}",
            chat_id=q.message.chat_id,
            user_id=uid,
        )
        status = "🟢 *Auto signals enabled!*\nYou'll receive signals every ~3 minutes."
    else:
        jobs = ctx.job_queue.get_jobs_by_name(f"auto_{uid}")
        for job in jobs:
            job.schedule_removal()
        status = "🔴 *Auto signals disabled.*"
 
    await q.edit_message_text(
        status,
        parse_mode="Markdown",
        reply_markup=main_keyboard(state["auto_signals"]),
    )
 
 
async def cb_result(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q    = update.callback_query
    uid  = q.from_user.id
    data = q.data  # "result:win:<trade_id>" or "result:loss:<trade_id>"
    await q.answer()
 
    _, outcome, trade_id = data.split(":", 2)
    state = get_state(uid)
    pending = state.get("pending")
 
    if pending is None or pending["id"] != trade_id:
        await q.edit_message_reply_markup(reply_markup=None)
        return
 
    stake = pending["stake"]
    if outcome == "win":
        pnl = round(stake * 0.85, 2)   # 85% payout
        state["balance"] = round(state["balance"] + pnl, 2)
        result_text = f"✅ *WIN!* +${pnl:.2f} | Balance: ${state['balance']:.2f}"
    else:
        pnl = -stake
        state["balance"] = round(state["balance"] - stake, 2)
        result_text = f"❌ *LOSS* −${stake:.2f} | Balance: ${state['balance']:.2f}"
 
    state["trades"].append({
        "id":        trade_id,
        "pair":      pending["pair"],
        "direction": pending["direction"],
        "stake":     stake,
        "result":    outcome,
        "pnl":       pnl,
        "time":      pending["time"],
    })
    state["pending"] = None
 
    await q.edit_message_text(
        q.message.text + f"\n\n{result_text}",
        parse_mode="Markdown",
        reply_markup=None,
    )
    await ctx.bot.send_message(
        q.message.chat_id,
        result_text,
        parse_mode="Markdown",
        reply_markup=main_keyboard(state["auto_signals"]),
    )
 
 
async def cb_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    state = get_state(q.from_user.id)
    await q.message.reply_text(
        format_stats(state),
        parse_mode="Markdown",
        reply_markup=main_keyboard(state["auto_signals"]),
    )
 
 
async def cb_balance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q     = update.callback_query
    uid   = q.from_user.id
    state = get_state(uid)
    await q.answer(f"💰 Balance: ${state['balance']:.2f}", show_alert=True)
 
 
async def cb_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    help_text = (
        "📖 *HOW TO USE*\n\n"
        "1️⃣ Tap *Forecast* — get a signal instantly\n"
        "2️⃣ Open your broker, enter the trade\n"
        "3️⃣ After 2 minutes, tap ✅ WIN or ❌ LOSS\n\n"
        "🔄 *Auto mode* sends signals every ~3 min\n\n"
        "📊 *Indicators used:*\n"
        "• EMA 8/21 — trend direction\n"
        "• MACD — momentum confirmation\n"
        "• RSI (14) — overbought/oversold\n"
        "• Stochastic — entry timing\n"
        "• Bollinger Bands — volatility\n"
        "• ADX — trend strength filter\n"
        "• Price Action — last 3 candles\n\n"
        "⚠️ *Risk warning:* Binary options carry significant risk. "
        "Only trade with money you can afford to lose."
    )
    state = get_state(q.from_user.id)
    await q.message.reply_text(
        help_text,
        parse_mode="Markdown",
        reply_markup=main_keyboard(state["auto_signals"]),
    )
 
 
# ── callback router ───────────────────────────────────────────────────────────
 
async def callback_router(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    data = update.callback_query.data
    if data == "forecast":
        await cb_forecast(update, ctx)
    elif data == "toggle_auto":
        await cb_toggle_auto(update, ctx)
    elif data.startswith("result:"):
        await cb_result(update, ctx)
    elif data == "stats":
        await cb_stats(update, ctx)
    elif data == "balance":
        await cb_balance(update, ctx)
    elif data == "help":
        await cb_help(update, ctx)
 
# ── commands ──────────────────────────────────────────────────────────────────
 
async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    state = get_state(update.effective_user.id)
    await update.message.reply_text(
        format_stats(state),
        parse_mode="Markdown",
        reply_markup=main_keyboard(state["auto_signals"]),
    )
 
 
async def cmd_balance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    state = get_state(update.effective_user.id)
    await update.message.reply_text(
        f"💰 Current balance: *${state['balance']:.2f}*",
        parse_mode="Markdown",
    )
 
 
async def cmd_setbalance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    state = get_state(update.effective_user.id)
    try:
        amount = float(ctx.args[0])
        if amount <= 0:
            raise ValueError
        state["balance"] = round(amount, 2)
        await update.message.reply_text(
            f"✅ Balance set to *${state['balance']:.2f}*",
            parse_mode="Markdown",
        )
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /setbalance <amount>  e.g. /setbalance 500")
 
 
async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user_states[uid] = default_state()
    await update.message.reply_text(
        "🔄 Account reset. Balance: *$1000.00*",
        parse_mode="Markdown",
        reply_markup=main_keyboard(False),
    )
 
# ── post-init: register bot commands ─────────────────────────────────────────
 
async def post_init(app: Application):
    await app.bot.set_my_commands([
        BotCommand("start",       "Start / Main menu"),
        BotCommand("stats",       "View trade statistics"),
        BotCommand("balance",     "Check balance"),
        BotCommand("setbalance",  "Set balance manually"),
        BotCommand("reset",       "Reset account"),
    ])
    log.info("Bot commands registered.")
 
# ── main ──────────────────────────────────────────────────────────────────────
 
def main():
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌  Set BOT_TOKEN environment variable or edit the script.")
        return
 
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )
 
    # command handlers
    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("stats",       cmd_stats))
    app.add_handler(CommandHandler("balance",     cmd_balance))
    app.add_handler(CommandHandler("setbalance",  cmd_setbalance))
    app.add_handler(CommandHandler("reset",       cmd_reset))
 
    # inline keyboard handler
    app.add_handler(CallbackQueryHandler(callback_router))
 
    log.info("🚀 Bot started — polling...")
    app.run_polling(drop_pending_updates=True)
 
 
if __name__ == "__main__":
    main()
