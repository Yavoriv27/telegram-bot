# =========================================================
# QUANT INSTITUTIONAL ENGINE v3
# LIQUIDITY + ORDERFLOW + XGBOOST + RETRAINING
# =========================================================

print("🔥 QUANT ENGINE v3 STARTED")

import os
import asyncio
import sqlite3
import requests
import numpy as np
import pandas as pd
import warnings
import joblib

from datetime import datetime, time

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)

from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes
)

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

OANDA_API_KEY = os.getenv("OANDA_API_KEY")

PAIR = "EUR_USD"

BASE_URL = "https://api-fxpractice.oanda.com/v3"

HEADERS = {
    "Authorization": f"Bearer {OANDA_API_KEY}"
}

AUTO_ENABLED = True

MODEL_FILE = "quant_model.pkl"

DB_FILE = "quant_engine.db"

USERS = set()

LAST_SIGNAL = None
LAST_SIGNAL_TIME = None

# =========================================================
# DATABASE
# =========================================================

conn = sqlite3.connect(
    DB_FILE,
    check_same_thread=False
)

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS learning(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    direction TEXT,
    confidence INTEGER,
    structure TEXT,
    bos TEXT,
    choch TEXT,
    sweep TEXT,
    flow TEXT,
    volatility REAL,
    session TEXT,
    result TEXT,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# =========================================================
# OANDA DATA
# =========================================================

def get_data(granularity="M5", count=1500):

    try:

        url = (
            f"{BASE_URL}/instruments/{PAIR}/candles"
            f"?granularity={granularity}"
            f"&count={count}"
            f"&price=M"
        )

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10
        )

        data = response.json()

        candles = data["candles"]

        rows = []

        for c in candles:

            if not c["complete"]:
                continue

            rows.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": c["volume"]
            })

        return pd.DataFrame(rows)

    except Exception as e:

        print("DATA ERROR:", e)

        return pd.DataFrame()

# =========================================================
# INDICATORS
# =========================================================

def indicators(df):

    if df.empty:
        return df

    df["ema20"] = (
        df["close"]
        .ewm(span=20)
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(span=50)
        .mean()
    )

    delta = df["close"].diff()

    gain = (
        delta.clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        (-delta.clip(upper=0))
        .rolling(14)
        .mean()
    )

    rs = gain / loss

    df["rsi"] = (
        100 - (100 / (1 + rs))
    )

    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"] - df["close"].shift())

    tr = pd.concat(
        [hl, hc, lc],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        tr.rolling(14).mean()
    )

    ema12 = (
        df["close"]
        .ewm(span=12)
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(span=26)
        .mean()
    )

    df["macd"] = ema12 - ema26

    df["macd_signal"] = (
        df["macd"]
        .ewm(span=9)
        .mean()
    )

    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr14 = tr.rolling(14).mean()

    plus_di = (
        100 *
        (plus_dm.rolling(14).mean() / tr14)
    )

    minus_di = (
        100 *
        (minus_dm.rolling(14).mean() / tr14)
    )

    dx = (
        abs(plus_di - minus_di)
        / abs(plus_di + minus_di)
    ) * 100

    df["adx"] = (
        dx.rolling(14).mean()
    )

    df["momentum"] = (
        df["close"]
        - df["close"].shift(5)
    )

    df["volatility"] = (
        df["close"]
        .rolling(20)
        .std()
    )

    return df.dropna()

# =========================================================
# STRUCTURE
# =========================================================

def market_structure(df):

    highs = df["high"].values
    lows = df["low"].values

    hh = highs[-1] > highs[-3]
    hl = lows[-1] > lows[-3]

    lh = highs[-1] < highs[-3]
    ll = lows[-1] < lows[-3]

    if hh and hl:
        return "BULLISH"

    if lh and ll:
        return "BEARISH"

    return "RANGE"

# =========================================================
# BOS
# =========================================================

def bos(df):

    high = (
        df["high"]
        .iloc[-12:-1]
        .max()
    )

    low = (
        df["low"]
        .iloc[-12:-1]
        .min()
    )

    close = df["close"].iloc[-1]

    if close > high:
        return "BULLISH_BOS"

    if close < low:
        return "BEARISH_BOS"

    return None

# =========================================================
# CHOCH
# =========================================================

def choch(df):

    prev_high = (
        df["high"]
        .iloc[-20:-10]
        .max()
    )

    recent_high = (
        df["high"]
        .iloc[-10:]
        .max()
    )

    prev_low = (
        df["low"]
        .iloc[-20:-10]
        .min()
    )

    recent_low = (
        df["low"]
        .iloc[-10:]
        .min()
    )

    if recent_high > prev_high:
        return "BULLISH_CHOCH"

    if recent_low < prev_low:
        return "BEARISH_CHOCH"

    return None

# =========================================================
# LIQUIDITY MAP
# =========================================================

def liquidity_map(df):

    equal_highs = []
    equal_lows = []

    highs = df["high"].values
    lows = df["low"].values

    for i in range(5, len(df)-5):

        if abs(highs[i] - highs[i-1]) < 0.00015:
            equal_highs.append(highs[i])

        if abs(lows[i] - lows[i-1]) < 0.00015:
            equal_lows.append(lows[i])

    return {
        "highs": equal_highs[-5:],
        "lows": equal_lows[-5:]
    }

# =========================================================
# SWEEP
# =========================================================

def liquidity_sweep(df):

    last = df.iloc[-1]

    prev_high = (
        df["high"]
        .iloc[-15:-1]
        .max()
    )

    prev_low = (
        df["low"]
        .iloc[-15:-1]
        .min()
    )

    if (
        last["high"] > prev_high
        and last["close"] < prev_high
    ):
        return "SELL_SWEEP"

    if (
        last["low"] < prev_low
        and last["close"] > prev_low
    ):
        return "BUY_SWEEP"

    return None

# =========================================================
# ORDERFLOW SIMULATION
# =========================================================

def orderflow(df):

    last = df.tail(8)

    bullish = (
        (last["close"] - last["open"])
        .clip(lower=0)
        .sum()
    )

    bearish = (
        (last["open"] - last["close"])
        .clip(lower=0)
        .sum()
    )

    if bullish > bearish * 1.5:
        return "BUY"

    if bearish > bullish * 1.5:
        return "SELL"

    return "NEUTRAL"

# =========================================================
# SESSION
# =========================================================

def session():

    now = datetime.utcnow().time()

    if time(7,0) <= now <= time(11,59):
        return "LONDON"

    if time(12,0) <= now <= time(17,0):
        return "NEW_YORK"

    return "OFF"

# =========================================================
# NEWS FILTER
# =========================================================

HIGH_IMPACT = [
    "CPI",
    "FOMC",
    "Powell",
    "Interest Rate",
    "NFP",
    "GDP",
    "ECB"
]

def news_filter():

    try:

        url = (
            "https://nfs.faireconomy.media/"
            "ff_calendar_thisweek.json"
        )

        r = requests.get(url, timeout=10)

        data = r.json()

        for event in data:

            title = event.get("title", "")
            country = event.get("country", "")

            if country not in ["USD", "EUR"]:
                continue

            for word in HIGH_IMPACT:

                if word.lower() in title.lower():
                    return True

        return False

    except:
        return False

# =========================================================
# ML ENGINE
# =========================================================

def train_model():

    df = indicators(
        get_data("M5", 4000)
    )

    X = []
    y = []

    for i in range(100, len(df)-4):

        X.append([
            df["close"].iloc[i]
            - df["close"].iloc[i-3],

            df["ema20"].iloc[i]
            - df["ema50"].iloc[i],

            df["rsi"].iloc[i],

            df["adx"].iloc[i],

            df["atr"].iloc[i],

            df["macd"].iloc[i],

            df["momentum"].iloc[i],

            df["volatility"].iloc[i]
        ])

        y.append(
            1
            if df["close"].iloc[i+4]
            > df["close"].iloc[i]
            else 0
        )

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=700,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    print("MODEL ACC:", round(acc, 3))

    joblib.dump(
        (model, scaler),
        MODEL_FILE
    )

# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    if not os.path.exists(MODEL_FILE):
        train_model()

    return joblib.load(MODEL_FILE)

# =========================================================
# LIVE RETRAINING
# =========================================================

def live_retraining():

    try:

        df = pd.read_sql(
            "SELECT * FROM learning",
            conn
        )

        if len(df) >= 50:

            print("🔄 LIVE RETRAINING")

            train_model()

    except Exception as e:

        print("RETRAIN ERROR:", e)

# =========================================================
# BACKTEST ENGINE
# =========================================================

def backtest():

    df = indicators(
        get_data("M5", 3500)
    )

    wins = 0
    losses = 0

    spread = 0.00008

    for i in range(100, len(df)-5):

        row = df.iloc[i]

        direction = (
            "BUY"
            if row["ema20"] > row["ema50"]
            else "SELL"
        )

        entry = row["close"] + spread

        future = df.iloc[i+4]["close"]

        if direction == "BUY":

            if future > entry:
                wins += 1
            else:
                losses += 1

        else:

            if future < entry:
                wins += 1
            else:
                losses += 1

    total = wins + losses

    if total == 0:
        return 0

    return round(
        (wins / total) * 100,
        1
    )

# =========================================================
# LEARNING
# =========================================================

def save_learning(signal, result):

    cur.execute("""
    INSERT INTO learning(
        direction,
        confidence,
        structure,
        bos,
        choch,
        sweep,
        flow,
        volatility,
        session,
        result
    )
    VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (
        signal["direction"],
        signal["confidence"],
        signal["structure"],
        signal["bos"],
        signal["choch"],
        signal["sweep"],
        signal["flow"],
        signal["volatility"],
        signal["session"],
        result
    ))

    conn.commit()

# =========================================================
# ADAPTIVE BONUS
# =========================================================

def adaptive_bonus():

    try:

        df = pd.read_sql(
            "SELECT * FROM learning "
            "ORDER BY id DESC LIMIT 150",
            conn
        )

        if len(df) < 30:
            return 0

        wins = len(
            df[df["result"] == "WIN"]
        )

        rate = wins / len(df)

        if rate > 0.65:
            return 10

        if rate < 0.45:
            return -10

        return 0

    except:
        return 0

# =========================================================
# SIGNAL ENGINE
# =========================================================

def generate_signal():

    if session() == "OFF":
        return None

    if news_filter():
        return None

    df15 = indicators(
        get_data("M15", 1500)
    )

    df5 = indicators(
        get_data("M5", 1500)
    )

    df1 = indicators(
        get_data("M1", 1500)
    )

    if df15.empty or df5.empty or df1.empty:
        return None

    trend15 = (
        "UP"
        if df15["ema20"].iloc[-1]
        > df15["ema50"].iloc[-1]
        else "DOWN"
    )

    trend5 = (
        "UP"
        if df5["ema20"].iloc[-1]
        > df5["ema50"].iloc[-1]
        else "DOWN"
    )

    if trend15 != trend5:
        return None

    direction = (
        "BUY"
        if trend15 == "UP"
        else "SELL"
    )

    structure = market_structure(df5)
    bos_signal = bos(df5)
    choch_signal = choch(df5)
    sweep = liquidity_sweep(df1)
    flow = orderflow(df1)

    liquidity = liquidity_map(df5)

    adx = df5["adx"].iloc[-1]
    rsi = df5["rsi"].iloc[-1]
    macd = df5["macd"].iloc[-1]
    macd_signal = df5["macd_signal"].iloc[-1]
    volatility = df5["volatility"].iloc[-1]

    print("========== MARKET DEBUG ==========")
    print("TREND15:", trend15)
    print("TREND5:", trend5)
    print("ADX:", round(adx, 2))
    print("RSI:", round(rsi, 2))
    print("MACD:", round(macd, 5))
    print("STRUCTURE:", structure)
    print("BOS:", bos_signal)
    print("CHOCH:", choch_signal)
    print("SWEEP:", sweep)
    print("FLOW:", flow)
    print("==================================")

    score = 0

    if adx > 19:
        score += 15

    if direction == "BUY":

        if structure == "BULLISH":
            score += 15

        if bos_signal == "BULLISH_BOS":
            score += 20

        if choch_signal == "BULLISH_CHOCH":
            score += 15

        if sweep == "BUY_SWEEP":
            score += 20

        if flow == "BUY":
            score += 10

        if macd > macd_signal:
            score += 10

        if rsi < 70:
            score += 5

    else:

        if structure == "BEARISH":
            score += 15

        if bos_signal == "BEARISH_BOS":
            score += 20

        if choch_signal == "BEARISH_CHOCH":
            score += 15

        if sweep == "SELL_SWEEP":
            score += 20

        if flow == "SELL":
            score += 10

        if macd < macd_signal:
            score += 10

        if rsi > 30:
            score += 5

    model, scaler = load_model()

    feat = np.array([
        df5["close"].iloc[-1]
        - df5["close"].iloc[-3],

        df5["ema20"].iloc[-1]
        - df5["ema50"].iloc[-1],

        rsi,
        adx,

        df5["atr"].iloc[-1],

        macd,

        df5["momentum"].iloc[-1],

        volatility
    ]).reshape(1, -1)

    feat = scaler.transform(feat)

    prob = model.predict_proba(feat)[0][1]

    if direction == "BUY":
        score += int(prob * 8)
    else:
        score += int((1 - prob) * 8)

    score += adaptive_bonus()

    confidence = min(score, 99)

    print("FINAL SCORE:", score)
    print("CONFIDENCE:", confidence)

    if confidence < 72:
        return None

    return {
        "direction": direction,
        "confidence": confidence,
        "structure": structure,
        "bos": bos_signal,
        "choch": choch_signal,
        "sweep": sweep,
        "flow": flow,
        "volatility": round(volatility, 5),
        "session": session(),
        "liquidity": liquidity
    }

# =========================================================
# KEYBOARD
# =========================================================

def keyboard():

    kb = [
        [
            InlineKeyboardButton(
                "📈 Прогноз",
                callback_data="signal"
            )
        ],
        [
            InlineKeyboardButton(
                "🤖 AUTO ON/OFF",
                callback_data="auto"
            )
        ],
        [
            InlineKeyboardButton(
                "📊 Backtest",
                callback_data="backtest"
            )
        ],
        [
            InlineKeyboardButton(
                "🔄 Retrain",
                callback_data="retrain"
            )
        ],
        [
            InlineKeyboardButton(
                "✅ WIN",
                callback_data="plus"
            ),
            InlineKeyboardButton(
                "❌ LOSS",
                callback_data="minus"
            )
        ]
    ]

    return InlineKeyboardMarkup(kb)

# =========================================================
# TELEGRAM
# =========================================================

async def start(update: Update, context):

    USERS.add(update.effective_chat.id)

    await update.message.reply_text(
        "✅ QUANT ENGINE ACTIVE",
        reply_markup=keyboard()
    )

async def button(update: Update, context):

    global AUTO_ENABLED
    global LAST_SIGNAL

    q = update.callback_query

    await q.answer()

    if q.data == "signal":

        signal = generate_signal()

        current_price = "N/A"

        try:

            price_df = get_data("M1", 1)

            if (
                not price_df.empty
                and "close" in price_df.columns
            ):

                current_price = round(
                    price_df["close"].iloc[-1],
                    5
                )

        except Exception as e:

            print("PRICE ERROR:", e)

        if not signal:

            await q.message.reply_text(
                f"❌ No signal\n\n"
                f"💰 Price: {current_price}\n\n"
                f"Reason:\n"
                f"• weak trend\n"
                f"• low confidence\n"
                f"• flat market"
            )

            return

        LAST_SIGNAL = signal

        msg = (
            f"🔥 EUR/USD QUANT SIGNAL\n\n"
            f"📈 {signal['direction']}\n"
            f"📊 Confidence: {signal['confidence']}%\n"
            f"🧠 Structure: {signal['structure']}\n"
            f"⚡ BOS: {signal['bos']}\n"
            f"🔄 CHOCH: {signal['choch']}\n"
            f"💧 Sweep: {signal['sweep']}\n"
            f"📦 Flow: {signal['flow']}\n"
            f"🌪 Volatility: {signal['volatility']}\n"
            f"🕐 Session: {signal['session']}\n\n"
            f"⏱ 4 MIN EXPIRATION"
        )

        await q.message.reply_text(
            msg,
            reply_markup=keyboard()
        )

    elif q.data == "auto":

        AUTO_ENABLED = not AUTO_ENABLED

        state = (
            "ON"
            if AUTO_ENABLED
            else "OFF"
        )

        await q.message.reply_text(
            f"🤖 AUTO {state}",
            reply_markup=keyboard()
        )

    elif q.data == "backtest":

        wr = backtest()

        await q.message.reply_text(
            f"📊 Backtest WR: {wr}%",
            reply_markup=keyboard()
        )

    elif q.data == "retrain":

        train_model()

        await q.message.reply_text(
            "🔄 MODEL RETRAINED",
            reply_markup=keyboard()
        )

    elif q.data in ["plus", "minus"]:

        result = (
            "WIN"
            if q.data == "plus"
            else "LOSS"
        )

        if LAST_SIGNAL:

            save_learning(
                LAST_SIGNAL,
                result
            )

            live_retraining()

        await q.message.reply_text(
            f"📊 RESULT: {result}",
            reply_markup=keyboard()
        )

# =========================================================
# AUTO SIGNALS
# =========================================================

async def auto_signals(app):

    global LAST_SIGNAL
    global LAST_SIGNAL_TIME

    while True:

        try:

            if AUTO_ENABLED:

                signal = generate_signal()

                if signal:

                    now = datetime.utcnow()

                    if LAST_SIGNAL_TIME:

                        sec = (
                            now - LAST_SIGNAL_TIME
                        ).seconds

                        if sec < 240:

                            await asyncio.sleep(30)

                            continue

                    LAST_SIGNAL = signal
                    LAST_SIGNAL_TIME = now

                    msg = (
                        f"🔥 AUTO SIGNAL\n\n"
                        f"📈 {signal['direction']}\n"
                        f"📊 {signal['confidence']}%\n"
                        f"🧠 {signal['structure']}\n"
                        f"⚡ {signal['bos']}\n"
                        f"🔄 {signal['choch']}\n"
                        f"💧 {signal['sweep']}\n"
                        f"📦 {signal['flow']}\n\n"
                        f"⏱ 4 MIN"
                    )

                    for user in USERS:

                        try:

                            await app.bot.send_message(
                                chat_id=user,
                                text=msg,
                                reply_markup=keyboard()
                            )

                        except:
                            pass

        except Exception as e:

            print("AUTO ERROR:", e)

        await asyncio.sleep(120)

# =========================================================
# POST INIT
# =========================================================

async def post_init(app):

    asyncio.create_task(
        auto_signals(app)
    )

# =========================================================
# MAIN
# =========================================================

def main():

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(
        CommandHandler("start", start)
    )

    app.add_handler(
        CallbackQueryHandler(button)
    )

    print("🚀 QUANT ENGINE RUNNING")

    app.run_polling(
         drop_pending_updates=True,
         timeout=30,
         read_timeout=30,
         write_timeout=30,
         connect_timeout=30,
         pool_timeout=30
   )

if __name__ == "__main__":
    main()
