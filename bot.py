import os
import time

TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    raise RuntimeError("BOT_TOKEN не заданий")

print("BOT TOKEN OK, starting bot...")

while True:
    time.sleep(60)
