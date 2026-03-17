import os, json, time, math, queue, threading
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
import pytz

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

load_dotenv()

KYIV = pytz.timezone("Europe/Kyiv")

def now():
    return datetime.now(timezone.utc).astimezone(KYIV).strftime("%H:%M:%S")

def mean(x): return sum(x)/len(x) if x else 0
def stdev(x): return math.sqrt(mean([(i-mean(x))**2 for i in x])) if x else 0
def pip(s): return 0.01 if "JPY" in s else 0.0001
def sigmoid(x): return 1/(1+math.exp(-x))

# ===== SETTINGS =====
RSI_BUY = 55
RSI_SELL = 45
ATR_FILTER = 0.0005
MIN_PROB = 0.7

# ===== STORAGE =====
FILE = "bot_state.json"

def load():
    if not os.path.exists(FILE):
        return {
            "wins":0,"losses":0,"streak":0,
            "weights":{"trend":1,"rsi":1,"momentum":1,"m1":1,"chop":1},
            "bias":0
        }
    return json.load(open(FILE))

def save(d): json.dump(d,open(FILE,"w"))

STATE = load()

# ===== CANDLE =====
@dataclass
class Candle:
    t:float;o:float;h:float;l:float;c:float
    def update(self,p):
        self.c=p;self.h=max(self.h,p);self.l=min(self.l,p)
    @property
    def dir(self):
        return "UP" if self.c>self.o else "DOWN"

class Builder:
    def __init__(self,tf):
        self.tf=tf;self.cur=None;self.last=None
    def bucket(self,ts): return ts-ts%self.tf
    def tick(self,ts,p):
        b=self.bucket(ts)
        if not self.cur:
            self.cur=Candle(b,p,p,p,p);return
        if self.cur.t==b:self.cur.update(p)
        else:
            self.last=self.cur
            self.cur=Candle(b,p,p,p,p)

# ===== INDICATORS =====
def ema(a,p):
    if len(a)<p:return None
    k=2/(p+1);e=a[0]
    for v in a[1:]:e=v*k+e*(1-k)
    return e

def rsi(a,p=14):
    if len(a)<p+1:return None
    g=l=0
    for i in range(-p,0):
        d=a[i]-a[i-1]
        if d>0:g+=d
        else:l-=d
    if l==0:return 100
    return 100-(100/(1+(g/l)))

def atr(h,l,c,p=14):
    trs=[]
    for i in range(1,len(c)):
        trs.append(max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])))
    return mean(trs[-p:]) if len(trs)>=p else None

# ===== ENGINE =====
class Engine:
    def __init__(self,s):
        self.s=s
        self.q=queue.Queue()

        self.b1=Builder(60)
        self.b5=Builder(300)
        self.b15=Builder(900)

        self.m1=[];self.m5=[];self.m15=[]

        # антиспам
        self.last_signal_candle=None
        self.last_signal_dir=None
        self.last_signal_prob=0
        self.last_signal_time=0
        self.cooldown=180

    def start(self):
        threading.Thread(target=self.stream,daemon=True).start()
        threading.Thread(target=self.loop,daemon=True).start()

    def stream(self):
        url=f"https://stream-fxpractice.oanda.com/v3/accounts/{os.getenv('OANDA_ACCOUNT_ID')}/pricing/stream"
        headers={"Authorization":f"Bearer {os.getenv('OANDA_API_KEY')}"}
        params={"instruments":self.s}

        while True:
            try:
                r=requests.get(url,headers=headers,params=params,stream=True)
                for l in r.iter_lines():
                    if not l:continue
                    d=json.loads(l.decode())
                    if d.get("type")=="PRICE":
                        p=float(d["bids"][0]["price"])
                        self.q.put((time.time(),p))
            except:
                time.sleep(3)

    def loop(self):
        while True:
            ts,p=self.q.get()

            self.b1.tick(ts,p)
            self.b5.tick(ts,p)
            self.b15.tick(ts,p)

            if self.b1.last:self.m1.append(self.b1.last)
            if self.b5.last:self.m5.append(self.b5.last)
            if self.b15.last:self.m15.append(self.b15.last)

            self.m1=self.m1[-200:]
            self.m5=self.m5[-200:]
            self.m15=self.m15[-200:]

    def get_last_candle(self):
        if not self.m5:return None
        return self.m5[-1].t

    def signal(self):
        if len(self.m15)<60:return None

        # STOP
        if STATE["streak"]<=-3:
            return None

        # cooldown
        now_ts=time.time()
        if now_ts-self.last_signal_time<self.cooldown:
            return None

        candle=self.get_last_candle()
        if not candle:return None

        closes=[c.c for c in self.m15]
        highs=[c.h for c in self.m15]
        lows=[c.l for c in self.m15]

        e20=ema(closes,20)
        e50=ema(closes,50)
        if not e20 or not e50:return None

        direction="BUY" if e20>e50 else "SELL"

        r=rsi(closes)
        a=atr(highs,lows,closes)
        if not r or not a:return None
        if a<ATR_FILTER:return None

        score=STATE["weights"]["trend"]

        if direction=="BUY" and r>RSI_BUY: score+=STATE["weights"]["rsi"]
        if direction=="SELL" and r<RSI_SELL: score+=STATE["weights"]["rsi"]

        last3=self.m5[-3:]
        ups=sum(1 for c in last3 if c.dir=="UP")
        downs=sum(1 for c in last3 if c.dir=="DOWN")

        if direction=="BUY" and ups>=2: score+=STATE["weights"]["momentum"]
        if direction=="SELL" and downs>=2: score+=STATE["weights"]["momentum"]

        if direction=="BUY" and self.m1[-1].dir=="UP": score+=STATE["weights"]["m1"]
        if direction=="SELL" and self.m1[-1].dir=="DOWN": score+=STATE["weights"]["m1"]

        ch=stdev(closes[-20:])
        if ch>pip(self.s)*2: score+=STATE["weights"]["chop"]

        prob=sigmoid(score+STATE["bias"])

        if prob<MIN_PROB:return None

        # антидубль
        if self.last_signal_candle==candle:
            return None

        if direction==self.last_signal_dir and prob<=self.last_signal_prob:
            return None

        self.last_signal_candle=candle
        self.last_signal_dir=direction
        self.last_signal_prob=prob
        self.last_signal_time=now_ts

        return {
            "dir":direction,
            "prob":round(prob*100,1),
            "symbol":self.s,
            "time":now()
        }

# ===== MULTI =====
SYMBOLS=["EUR_USD","GBP_USD","USD_JPY"]
ENGINES=[Engine(s) for s in SYMBOLS]

for e in ENGINES:e.start()

LAST=None

def best():
    global LAST
    best=None
    for e in ENGINES:
        s=e.signal()
        if not s:continue
        if not best or s["prob"]>best["prob"]:
            best=s
    LAST=best
    return best

# ===== TRAIN =====
def train(win):
    if not LAST:return
    if win:
        STATE["wins"]+=1
        STATE["streak"]+=1
        STATE["bias"]+=0.1
    else:
        STATE["losses"]+=1
        STATE["streak"]-=1
        STATE["bias"]-=0.1
    save(STATE)

# ===== TELEGRAM =====
def kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅",callback_data="win"),
         InlineKeyboardButton("❌",callback_data="loss")]
    ])

def fmt(s):
    if STATE["streak"]<=-3:
        return "⛔ STOP (серія мінусів)"

    if not s:
        return f"❌ Нема сигналу\n🕒 {now()}"

    return (
        f"{'🟢 BUY' if s['dir']=='BUY' else '🔴 SELL'} {s['symbol']}\n"
        f"📊 {s['prob']}%\n"
        f"⏱ 2 хв\n"
        f"🕒 {s['time']}"
    )

async def signal(update:Update,context:ContextTypes.DEFAULT_TYPE):
    s=best()
    await update.message.reply_text(fmt(s),reply_markup=kb())

async def callback(update:Update,context:ContextTypes.DEFAULT_TYPE):
    q=update.callback_query
    await q.answer()
    train(q.data=="win")

    total=STATE["wins"]+STATE["losses"]
    wr=round(STATE["wins"]/total*100,1) if total else 0

    await q.edit_message_text(f"WR: {wr}% | streak: {STATE['streak']}")

async def start(update:Update,context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔥 BOT READY (PRO LEVEL)")

async def auto(context):
    s=best()
    if s:
        await context.bot.send_message(context.job.chat_id,fmt(s),reply_markup=kb())

def main():
    app=Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    app.add_handler(CommandHandler("start",start))
    app.add_handler(CommandHandler("signal",signal))
    app.add_handler(CallbackQueryHandler(callback))

    async def start_auto(update:Update,context:ContextTypes.DEFAULT_TYPE):
        context.job_queue.run_repeating(auto,interval=120,first=10,chat_id=update.effective_chat.id)
        await update.message.reply_text("✅ Автосигнали увімкнено")

    app.add_handler(CommandHandler("auto",start_auto))

    app.run_polling(drop_pending_updates=True)

if __name__=="__main__":
    main()
