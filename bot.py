import os
import time

TOKEN = None
while not TOKEN:
    TOKEN = os.getenv("BOT_TOKEN")
    if not TOKEN:
        print("Waiting for BOT_TOKEN...")
        time.sleep(2)

