import keyboard
import time
from utils.alert import raise_alert

class HotkeyMonitor:
    def __init__(self):
        self.blocked_keys = {
            'alt+tab': self.handle_cheat_attempt,
            'ctrl+t': self.handle_cheat_attempt,
            'print screen': self.handle_cheat_attempt,
            'windows+d': self.handle_cheat_attempt
        }
        self.running = True
        
    def handle_cheat_attempt(self):
        raise_alert("Cheat attempt detected! Blocked restricted hotkey")
        return False  # Blocks the original key combination

    def start_monitoring(self):
        for hotkey, callback in self.blocked_keys.items():
            keyboard.add_hotkey(hotkey, callback, suppress=True)
        
        while self.running:
            time.sleep(0.1)

    def stop(self):
        self.running = False
        keyboard.unhook_all()