import time
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict
import os
import threading
import requests
from path_utils import get_resource_path, get_app_data_dir
from bson import ObjectId
from buzzer_utils import buzzer

# Optional deps
try:
    from twilio.rest import Client
except Exception:
    Client = None


class AlertEngine:
    """
    Central alert dispatcher
    - Reads config from DB
    - Applies cooldown
    - Triggers WhatsApp Cloud API / SMS Twilio
    """

    def __init__(self, db):
        self.db = db
        self.last_alert = {}  # key -> timestamp
        self.last_cleanup = time.time()
        self._load_config()
        
        # Start dedicated background cleanup thread for 48-hour retention
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(target=self._run_cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def _run_cleanup_loop(self):
        """Background loop to periodically cleanup old images (every 48 hours)"""
        print("🧹 Violation image cleanup loop started (48h retention)")
        while self.cleanup_running:
            try:
                # Retention = 48 hours (2 days)
                self._cleanup_old_snapshots(retention_days=2)
            except Exception as e:
                print(f"⚠️ Cleanup error in loop: {e}")
            
            # Check every 30 minutes
            time.sleep(1800)

    # --------------------------------------------------
    # CONFIG
    # --------------------------------------------------
    def _load_config(self):
        cfg = self.db.system_config.find_one(
            {"config_type": "alerts"}
        ) or {}

        self.enabled = cfg.get("enable_alerts", True)
        self.delivery_mode = cfg.get("delivery_mode", "whatsapp_with_sms_fallback")
        self.cooldown_sec = cfg.get("cooldown_seconds", 30)

        # Support both 'whatsapp_cloud' and 'wspanel' keys for robustness
        self.whatsapp_cfg = cfg.get("whatsapp_cloud") or cfg.get("wspanel") or {}
        self.sms_cfg = cfg.get("sms_twilio", {})
        
        # ✅ NEW: Buzzer Settings
        buzzer_enabled = cfg.get("enable_buzzer", True)
        buzzer.set_enabled(buzzer_enabled)

    def reload(self):
        self._load_config()
        print("🔔 Alert config reloaded")

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------
    def handle_violation(self, violation: Dict) -> str:
        """
        Called by PPE violation logger.
        Saves a snapshot and triggers alerts.
        Returns the snapshot filename/URL.
        """
        if not self.enabled:
            return None

        key = f"{violation.get('camera_id')}:{violation.get('person_id')}"

        now = time.time()
        last = self.last_alert.get(key, 0)

        log_id = violation.get("log_id")
        print(f"🔍 handle_violation called - log_id: {log_id}, camera: {violation.get('camera_id')}, person: {violation.get('person_name')}")

        # Always save snapshot for UI if it's a violation, 
        # but only send external alerts (WhatsApp/SMS) on cooldown.
        snapshot_url = self._save_violation_snapshot(violation)
        print(f" Snapshot saved, URL: {snapshot_url}")

        if now - last < self.cooldown_sec:
            return snapshot_url # return snapshot but skip external alert

        self.last_alert[key] = now
        
        # Add snapshot to violation data for dispatcher
        violation["snapshot_url"] = snapshot_url

        message = self._build_message(violation)

        # Run alert delivery in background
        t = threading.Thread(
            target=self._dispatch_alert,
            args=(message, violation),
            daemon=True
        )
        t.start()

        print("🚨 Alert triggered:", message)
        return snapshot_url

    def _dispatch_alert(self, message: str, violation: Dict):
        snapshot_url = violation.get("snapshot_url")
        
        if self.delivery_mode == "whatsapp_only":
            self._send_whatsapp_wspanel(message, snapshot_url)

        elif self.delivery_mode == "whatsapp_with_sms_fallback":
            success = self._send_whatsapp_wspanel(message, snapshot_url)
            if not success:
                self._send_sms_twilio(message)

        elif self.delivery_mode == "sms_only":
            self._send_sms_twilio(message)

        else:
            print(f"📣 ALERT (Console Only): {message}")


    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def handle_fire_smoke_alert(self, event: Dict) -> str:
        """High-priority fire/smoke alert — shorter cooldown (5s), urgent message."""
        if not self.enabled:
            return None

        key = f"fire_smoke:{event.get('camera_id')}:{event.get('category')}"
        now = time.time()
        
        # 5-second cooldown for fire/smoke to ensure we don't spam while allowing rapid updates if needed
        last = self.last_alert.get(key, 0)
        if now - last < 5:
            return None

        self.last_alert[key] = now
        
        # Save snapshot (Note: fire detections don't have person_id, we use "fire" as person_id for name key)
        alert_data = event.copy()
        alert_data["person_name"] = event.get("category", "Fire/Smoke").upper()
        alert_data["person_id"] = f"event_{event.get('category')}"
        
        snapshot_url = self._save_violation_snapshot(alert_data)
        alert_data["snapshot_url"] = snapshot_url
        
        # Build special emergency message
        message = self._build_fire_smoke_message(event)
        
        # Dispatch immediately in background
        t = threading.Thread(
            target=self._dispatch_alert,
            args=(message, alert_data),
            daemon=True
        )
        t.start()
        
        # ✅ NEW: Prominent Console Alert + File Logging
        print("\n" + "!"*60)
        print(f"🔥 FIRE/SMOKE ALERT TRIGGERED: {message.replace(chr(10), ' | ')}")
        print("!"*60 + "\n")
        
        # Log to dedicated file for debugging
        try:
            log_path = os.path.join(get_app_data_dir(), "fire_alerts.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {message.replace(chr(10), ' | ')} | Snapshot: {snapshot_url}\n")
        except Exception as e:
            print(f"⚠️ Failed to write to fire_alerts.log: {e}")
            
        # ✅ EMERGENCY PRIORITY: Trigger Urgent Audible Buzzer for Fire/Smoke
        buzzer.trigger_alert(duration_sec=10, frequency=1500, is_emergency=True)

        return snapshot_url

    def _build_fire_smoke_message(self, v: Dict) -> str:
        cam_id = v.get('camera_id', 'Unknown Camera')
        
        # Get zone/location from database if available
        location = "Unknown"
        cam_name = cam_id
        try:
            if hasattr(self, 'db') and self.db:
                cam_info = self.db.get_camera(cam_id)
                if cam_info:
                    location = cam_info.get('location') or cam_info.get('zone') or cam_id
                    cam_name = cam_info.get('name', cam_id)
        except Exception:
            pass
            
        event_type = v.get('category', 'Detection').upper()
        conf = v.get('confidence', 0.0)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        is_simultaneous = v.get("has_simultaneous_fall", False)
        if is_simultaneous:
            event_type = f"{event_type} & PERSON LYING ON GROUND"
            
        return (
            f"🔥 EMERGENCY ALERT: {event_type} DETECTED\n"
            f"Zone/Location: {location} (Camera: {cam_name})\n"
            f"Confidence: {conf*100:.1f}%\n"
            f"Time: {timestamp}\n"
            f"🚨 IMMEDIATE EVACUATION/ACTION REQUIRED"
        )

    def handle_fall_alert(self, event: Dict) -> str:
        """Emergency handler for fall and lying person detection"""
        if not self.enabled:
            return None

        key = f"fall:{event.get('camera_id')}:{event.get('category')}"
        now = time.time()
        
        # 10 second cooldown for fall detection
        last = self.last_alert.get(key, 0)
        if now - last < 10:
            return None

        self.last_alert[key] = now
        
        # Save snapshot
        alert_data = event.copy()
        alert_data["person_name"] = "Person Lying on Ground"
        alert_data["person_id"] = "event_fall"
        
        snapshot_url = self._save_violation_snapshot(alert_data)
        alert_data["snapshot_url"] = snapshot_url
        
        # Build special emergency message
        message = self._build_fall_message(event)
        
        # Dispatch immediately in background
        t = threading.Thread(
            target=self._dispatch_alert,
            args=(message, alert_data),
            daemon=True
        )
        t.start()
        
        # Prominent Console Alert
        print("\n" + "!"*60)
        print(f"🆘 FALL / LYING PERSON ALERT TRIGGERED: {message.replace(chr(10), ' | ')}")
        print("!"*60 + "\n")
            
        # ✅ EMERGENCY PRIORITY: Trigger Urgent Audible Buzzer for Fall/Danger
        # buzzer.trigger_alert(duration_sec=5, frequency=1000, is_emergency=True)

        return snapshot_url

    def _build_fall_message(self, v: Dict) -> str:
        cam_id = v.get('camera_id', 'Unknown Camera')
        
        # Get zone/location from database if available
        location = "Unknown"
        cam_name = cam_id
        try:
            if hasattr(self, 'db') and self.db:
                cam_info = self.db.get_camera(cam_id)
                if cam_info:
                    location = cam_info.get('location') or cam_info.get('zone') or cam_id
                    cam_name = cam_info.get('name', cam_id)
        except Exception:
            pass
            
        conf = v.get('confidence', 0.0)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        is_simultaneous = v.get("has_simultaneous_fire", False)
        event_str = "PERSON FALLING/LYING ON GROUND"
        if is_simultaneous:
            event_str = "FIRE/SMOKE & PERSON LYING ON GROUND"

        return (
            f"🆘 EMERGENCY ALERT: {event_str} DETECTED\n"
            f"Zone/Location: {location} (Camera: {cam_name})\n"
            f"Confidence: {conf*100:.1f}%\n"
            f"Time: {timestamp}\n"
            f"🚨 IMMEDIATE MEDICAL/SAFETY ATTENTION REQUIRED"
        )

    def handle_night_mode_violation(self, violation: Dict) -> str:
        """Specialized handler for person detection during night mode"""
        if not self.enabled:
            return None

        key = f"night_mode:{violation.get('camera_id')}:{violation.get('person_id')}"
        now = time.time()
        
        # Shorter cooldown for security alerts? maybe 5 seconds
        last = self.last_alert.get(key, 0)
        if now - last < 5:
            return None

        self.last_alert[key] = now
        
        # Save snapshot
        snapshot_url = self._save_violation_snapshot(violation)
        violation["snapshot_url"] = snapshot_url
        
        # Build special security message
        message = self._build_night_mode_message(violation)
        
        # Dispatch
        t = threading.Thread(
            target=self._dispatch_alert,
            args=(message, violation),
            daemon=True
        )
        t.start()
        
        print("🚨 NIGHT MODE ALERT triggered:", message)
        return snapshot_url

    def _build_night_mode_message(self, v: Dict) -> str:
        cam_name = v.get('camera_id', 'Unknown Camera')
        person_name = v.get('person_name', 'Unknown Person')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return (
            f"🔴 SECURITY ALERT: NIGHT MODE BREACH\n"
            f"Location: {cam_name}\n"
            f"Detection: {person_name}\n"
            f"Time: {timestamp}\n"
            f"Status: UNAUTHORIZED ENTRY"
        )

    def _build_message(self, v: Dict) -> str:
        cam_name = v.get('camera_id', 'Unknown Camera')
        person_name = v.get('person_name', 'Unknown')
        missing = ', '.join(v.get('missing_ppe', []))
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return (
            f"⚠️ PPE VIOLATION\n"
            f"Camera: {cam_name}\n"
            f"Person: {person_name}\n"
            f"Missing: {missing}\n"
            f"Time: {timestamp}"
        )
    
    def _save_violation_snapshot(self, violation: Dict):
        """
        Save annotated snapshot for WhatsApp alert.
        Returns public URL or None.
        """

        try:
            camera_id = violation.get("camera_id")
            if not camera_id:
                return None

            # Get annotated frame from inference controller
            frame = violation.get("annotated_frame")

            if frame is None:
                return None

            folder = Path(get_app_data_dir()) / "alert_images"
            folder.mkdir(parents=True, exist_ok=True)

            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            log_id = violation.get("log_id")
            
            if log_id:
                filename = f"violation_{log_id}.jpg"
                snapshot_url_path = f"/alert_images/{filename}"
                
                try:
                    print(f"📝 Updating violation log {log_id} with snapshot URL: {snapshot_url_path}")
                    result = self.db.recognition_logs.update_one(
                        {"_id": ObjectId(log_id)},
                        {"$set": {"snapshot_url": snapshot_url_path}}
                    )
                    
                    if result.matched_count > 0:
                        print(f"✅ Successfully updated log {log_id} with snapshot URL")
                    else:
                        print(f"⚠️ Log ID {log_id} not found in database!")
                        
                except Exception as e:
                    print(f"❌ Failed to update log with snapshot URL: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ No log_id provided, using timestamp-based filename")
                filename = f"{camera_id}_{ts}.jpg"
                
            path = folder / filename

            print(f"💾 Saving snapshot to: {path}")
            cv2.imwrite(str(path), frame)

            # IMPORTANT: must be reachable by WhatsApp Cloud
            server = os.getenv("ALERT_SERVER", "http://127.0.0.1:5000")
            return f"{server}/alert_images/{filename}"


        except Exception as e:
            print("📸 Snapshot save failed:", e)
            return None

    def _cleanup_old_snapshots(self, retention_days=2):
        """Delete alert images older than 2 days"""
        try:
            folder = Path(get_app_data_dir()) / "alert_images"
            if not folder.exists(): 
                self.last_cleanup = time.time()
                return
            
            cutoff = time.time() - (retention_days * 86400)
            count = 0
            
            # Update timestamp to prevent concurrent runs
            self.last_cleanup = time.time()
            
            for file in folder.glob("*"):
                try:
                    if file.is_file() and file.stat().st_mtime < cutoff:
                        # Skip license plate snapshots from automatic deletion
                        if file.name.startswith("plate_"):
                            continue
                        file.unlink()
                        count += 1
                except: pass
            
            if count > 0:
                print(f"🧹 Cleaned up {count} old alert images")
            
        except Exception as e:
            print(f"Cleanup error: {e}")


    # --------------------------------------------------
    # WHATSAPP WS PANEL API
    # --------------------------------------------------
    def _send_whatsapp_wspanel(self, message: str, snapshot_url: str = None) -> bool:
        # Use the already loaded config which supports both 'whatsapp_cloud' and 'wspanel' keys
        instance_id = self.whatsapp_cfg.get("instance_id")
        access_token = self.whatsapp_cfg.get("access_token")
        recipient = self.whatsapp_cfg.get("recipient_number")

        if not all([instance_id, access_token, recipient]):
            print("⚠️ WhatsApp WS Panel config incomplete in AlertEngine")
            return False

        clean_recipient = recipient.replace("+", "").replace(" ", "").strip()

        # If snapshot_url is a local address (e.g. 127.0.0.1 or localhost),
        # WS Panel will fail to download it and will drop the whole message.
        # So we strip the media_url if it's local to ensure the text snippet arrives.
        if snapshot_url and ("127.0.0.1" in snapshot_url or "localhost" in snapshot_url):
            print(f"⚠️ Snapshot URL {snapshot_url} is local. Stripping media from WS Panel request to ensure delivery.")
            snapshot_url = None

        # Build parameters for media if snapshot exists
        if snapshot_url:
            params = {
                "number": clean_recipient,
                "type": "media",
                "message": message,
                "media_url": snapshot_url,
                "instance_id": instance_id,
                "access_token": access_token
            }
        else:
            params = {
                "number": clean_recipient,
                "type": "text",
                "message": message,
                "instance_id": instance_id,
                "access_token": access_token
            }

        try:
            r = requests.get("https://wspanel.in/api/send", params=params, timeout=15)
            response_json = {}
            try:
                response_json = r.json()
            except: pass
            
            print("📝 WS Panel response:", r.text)
            
            if response_json.get("status") == "error":
                if "Invalidated" in response_json.get("message", ""):
                    print("❌ WHATSAPP ERROR: Your WS Panel Instance ID has been invalidated. Please check your wspanel.in dashboard, reconnect your instance, and update the ID in Settings.")
                return False
                
            return r.status_code == 200
        except Exception as e:
            print("❌ WS Panel exception:", e)
            return False

    # --------------------------------------------------
    # SMS TWILIO
    # --------------------------------------------------
    def _send_sms_twilio(self, message: str) -> bool:
        """Send SMS via Twilio. Returns True if success."""
        if not Client:
            print("✉️ SMS skipped (twilio not installed)")
            return False

        sid = self.sms_cfg.get("account_sid")
        token = self.sms_cfg.get("auth_token")
        from_no = self.sms_cfg.get("twilio_number")
        to_no = self.sms_cfg.get("recipient_number")

        if not all([sid, token, from_no, to_no]):
            print("✉️ SMS config incomplete")
            return False

        try:
            client = Client(sid, token)
            client.messages.create(
                body=message,
                from_=from_no,
                to=to_no
            )
            print("✉️ SMS alert sent")
            return True
        except Exception as e:
            print("✉️ SMS error:", e)
            return False

