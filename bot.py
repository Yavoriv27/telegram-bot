import os
import json
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv
import pytz

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from PIL import Image
import cv2
import numpy as np
import pytesseract

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("po_screenshot_bot")

KYIV_TZ = pytz.timezone("Europe/Kyiv")

PO_EXPIRY_SEC = 600

RSI_SELL_BLOCK = float(os.getenv("RSI_SELL_BLOCK", "28"))
RSI_BUY_BLOCK = float(os.getenv("RSI_BUY_BLOCK", "72"))

MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "62"))
MAX_CONFIDENCE = int(os.getenv("MAX_CONFIDENCE", "75"))

STATE_FILE = os.getenv("PO_STATE_FILE", "/app/po_state.json")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fmt_kyiv(dt_utc: datetime) -> str:
    return dt_utc.astimezone(KYIV_TZ).strftime("%Y-%m-%d %H:%M:%S")


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x.replace(",", ".").strip())
    except Exception:
        return None


def load_state() -> dict:
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"pending": {}}


def save_state(state: dict):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    except Exception:
        pass
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


STATE = load_state()


@dataclass
class POAnalysis:
    direction: str
    confidence: int
    risk: int
    why: List[str]
    rsi: Optional[float] = None
    adx: Optional[float] = None


def _img_to_cv(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thr


def _ocr_text(img_bin: np.ndarray) -> str:
    cfg = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(img_bin, config=cfg)
    return text


def extract_rsi_adx_from_image(path: str) -> Tuple[Optional[float], Optional[float]]:
    img = _img_to_cv(path)
    h, w = img.shape[:2]
    roi = img[int(h * 0.70): int(h * 0.98), int(w * 0.00): int(w * 0.65)]
    bin_img = _preprocess_for_ocr(roi)
    txt = _ocr_text(bin_img).upper().replace(" ", "")

    rsi_val = None
    adx_val = None

    for key in ("RSI",):
        i = txt.find(key)
        if i != -1:
            chunk = txt[i:i + 40]
            nums = []
            cur = ""
            for ch in chunk:
                if ch.isdigit() or ch in ".,": 
                    cur += ch
                else:
                    if cur:
                        nums.append(cur)
                        cur = ""
            if cur:
                nums.append(cur)
            for n in nums:
                v = safe_float(n)
                if v is not None and 0 <= v <= 100:
                    rsi_val = v
                    break

    for key in ("ADX",):
        i = txt.find(key)
        if i != -1:
            chunk = txt[i:i + 40]
            nums = []
            cur = ""
            for ch in chunk:
                if ch.isdigit() or ch in ".,": 
                    cur += ch
                else:
                    if cur:
                        nums.append(cur)
                        cur = ""
            if cur:
                nums.append(cur)
            for n in nums:
                v = safe_float(n)
                if v is not None and 0 <= v <= 100:
                    adx_val = v
                    break

    return rsi_val, adx_val


def analyze_price_action_basic(tf15_path: str, tf1_path: str) -> POAnalysis:
    why = []
    confidence = 50

    rsi15, adx15 = extract_rsi_adx_from_image(tf15_path)
    rsi1, adx1 = extract_rsi_adx_from_image(tf1_path)

    rsi_v = rsi1 if rsi1 is not None else rsi15
    adx_v = adx1 if adx1 is not None else adx15

    if adx_v is not None:
        if adx_v >= 25:
            confidence += 8
            why.append(f"ADX {adx_v:.1f} (рух є)")
        else:
            confidence -= 8
            why.append(f"ADX {adx_v:.1f} (слабкий рух)")
    else:
        why.append("ADX не прочитався")

    if rsi_v is not None:
        why.append(f"RSI {rsi_v:.1f}")
    else:
        why.append("RSI не прочитався")

    direction = "NEUTRAL"

    if rsi_v is not None:
        if rsi_v <= 35:
            direction = "BUY"
            confidence += 6
            why.append("RSI низький → можливий відскок вгору")
        elif rsi_v >= 65:
            direction = "SELL"
            confidence += 6
            why.append("RSI високий → можливий відкат вниз")
        else:
            direction = "NEUTRAL"
            confidence -= 6
            why.append("RSI середній → немає перекосу")

    if direction == "SELL" and rsi_v is not None and rsi_v < RSI_SELL_BLOCK:
        direction = "NEUTRAL"
        confidence -= 12
        why.append(f"SELL заборонено: RSI<{RSI_SELL_BLOCK}")

    if direction == "BUY" and rsi_v is not None and rsi_v > RSI_BUY_BLOCK:
        direction = "NEUTRAL"
        confidence -= 12
        why.append(f"BUY заборонено: RSI>{RSI_BUY_BLOCK}")

    if adx_v is not None and adx_v < 22:
        direction = "NEUTRAL"
        confidence -= 10
        why.append("Флет-фільтр: ADX<22")

    confidence = max(40, min(confidence, MAX_CONFIDENCE))

    if direction == "NEUTRAL" or confidence < MIN_CONFIDENCE:
        direction = "NEUTRAL"
        confidence = min(confidence, 60)

    risk = 100 - confidence
    return POAnalysis(direction=direction, confidence=int(confidence), risk=int(risk), why=why, rsi=rsi_v, adx=adx_v)


def fmt_po_signal(a: POAnalysis) -> str:
    t = fmt_kyiv(now_utc())

    if a.direction == "NEUTRAL":
        reasons = "\n".join([f"• {x}" for x in a.why])
        return (
            "⚪ <b>NEUTRAL</b>\n"
            f"⏱ <b>Експірація:</b> 10 хв\n"
            f"🕒 <b>Kyiv:</b> {t}\n"
            f"⚠️ <b>Причина:</b> немає сильного сетапу\n\n"
            f"<b>Деталі:</b>\n{reasons}"
        )

    arrow = "🟢 BUY" if a.direction == "BUY" else "🔴 SELL"
    reasons = "\n".join([f"• {x}" for x in a.why])

    enter_hint = "🔔 Рекомендовано входити негайно" if a.confidence >= 68 else "⏳ Краще дочекатися закриття 1M свічки"

    return (
        f"{arrow}\n"
        f"⏱ <b>Експірація:</b> 10 хв\n"
        f"🕒 <b>Kyiv:</b> {t}\n"
        f"📊 <b>Ймовірність:</b> {a.confidence}%\n"
        f"⚠️ <b>Ризик:</b> {a.risk}%\n\n"
        f"{enter_hint}\n\n"
        f"<b>Пояснення:</b>\n{reasons}"
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ PocketOption-режим активний.\n\n"
        "Команди:\n"
        "/signal — запросити сигнал (бот попросить 2 скріни: 15M і 1M)\n"
        "/reset — скинути очікування скрінів\n",
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    STATE["pending"].pop(chat_id, None)
    save_state(STATE)
    await update.message.reply_text("✅ Очікування скрінів скинуто. Напиши /signal")


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    STATE["pending"][chat_id] = {"step": 1, "tf15": None, "tf1": None}
    save_state(STATE)
    await update.message.reply_text(
        "📸 Надішли 2 скріни з PocketOption:\n"
        "1) 15M (тренд)\n"
        "2) 1M (вхід)\n\n"
        "Відправ по черзі, 2 фото підряд.",
    )


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    pending = STATE["pending"].get(chat_id)

    if not pending:
        await update.message.reply_text("Напиши /signal і потім надішли 2 скріни (15M і 1M).")
        return

    photos = update.message.photo
    if not photos:
        return

    file = await context.bot.get_file(photos[-1].file_id)

    os.makedirs("/tmp/po", exist_ok=True)

    if pending["step"] == 1:
        path = f"/tmp/po/{chat_id}_15m.jpg"
        await file.download_to_drive(path)
        pending["tf15"] = path
        pending["step"] = 2
        save_state(STATE)
        await update.message.reply_text("✅ Прийняв 15M. Тепер надішли 1M скрін.")
        return

    if pending["step"] == 2:
        path = f"/tmp/po/{chat_id}_1m.jpg"
        await file.download_to_drive(path)
        pending["tf1"] = path
        save_state(STATE)

        try:
            a = analyze_price_action_basic(pending["tf15"], pending["tf1"])
            msg = fmt_po_signal(a)
            await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
        except Exception as e:
            log.exception("analyze error: %s", e)
            await update.message.reply_text("❌ Помилка аналізу. Спробуй ще раз /signal")

        STATE["pending"].pop(chat_id, None)
        save_state(STATE)
        return


def acquire_lock(lock_path: str):
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                old_pid = int((f.read() or "").strip())
            os.kill(old_pid, 0)
            log.error("Bot already running with PID=%s", old_pid)
            raise SystemExit(0)
        except ProcessLookupError:
            try:
                os.remove(lock_path)
            except Exception:
                pass
        except Exception:
            try:
                os.remove(lock_path)
            except Exception:
                pass

    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def main():
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

    lock_path = "/tmp/telegram_bot.lock"
    acquire_lock(lock_path)

    try:
        app = Application.builder().token(token).build()

        app.add_handler(CommandHandler("start", cmd_start))
        app.add_handler(CommandHandler("signal", cmd_signal))
        app.add_handler(CommandHandler("reset", cmd_reset))
        app.add_handler(MessageHandler(filters.PHOTO, on_photo))

        app.run_polling(drop_pending_updates=True)
    finally:
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
