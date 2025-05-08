import pyautogui
import os
from datetime import datetime

SNAPSHOT_DIR = os.path.join("assets", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def save_snapshot():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(SNAPSHOT_DIR, f"snapshot_{now}.png")
    screenshot = pyautogui.screenshot()
    screenshot.save(filepath)
    print(f"[INFO] Screenshot saved: {filepath}")
