import threading
import time
import os

try:
    import winsound
    WIN_SOUND_AVAILABLE = True
except ImportError:
    WIN_SOUND_AVAILABLE = False

class BuzzerManager:
    """
    Manages PC Speaker (Buzzer) notifications.
    Uses winsound.Beep if available on Windows.
    """
    def __init__(self):
        self._enabled = True
        self._beeping = False
        self._thread = None
        self._lock = threading.Lock()

    def set_enabled(self, enabled):
        """Toggle buzzer globally"""
        with self._lock:
            self._enabled = enabled
            if not enabled:
                self._beeping = False

    def is_enabled(self):
        return self._enabled

    def trigger_alert(self, duration_sec=5, frequency=1000, is_emergency=False):
        """Sound the buzzer for a duration"""
        if not self._enabled or not WIN_SOUND_AVAILABLE:
            return

        with self._lock:
            if self._beeping:
                return # Already sounding
            self._beeping = True

        self._thread = threading.Thread(
            target=self._beep_loop, 
            args=(duration_sec, frequency, is_emergency),
            daemon=True
        )
        self._thread.start()

    def _beep_loop(self, duration, freq, is_emergency):
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                with self._lock:
                    if not self._enabled or not self._beeping:
                        break
                
                if is_emergency:
                    # Urgent pattern: ON for 100ms, OFF for 50ms (faster pips)
                    winsound.Beep(freq, 100)
                    time.sleep(0.05)
                else:
                    # Standard buzzer pattern: ON for 200ms, OFF for 100ms
                    winsound.Beep(freq, 200)
                    time.sleep(0.1)
                
        except Exception as e:
            print(f"⚠️ Buzzer error: {e}")
        finally:
            with self._lock:
                self._beeping = False

# Global singleton
buzzer = BuzzerManager()
