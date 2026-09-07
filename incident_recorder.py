import cv2
import time
import os
import threading
from pathlib import Path
from datetime import datetime
from collections import deque
import random

class IncidentVideoRecorder:
    def __init__(self, db, default_retention_days=7):
        self.db = db
        self.recordings = {}  # camera_id -> { 'writer': VideoWriter, 'end_time': float, 'filepath': str, 'tmp_path': str, 'last_frame': float, ... }
        self.buffers = {} # camera_id -> deque of frames
        self.default_retention_days = default_retention_days
        self.instance_id = random.randint(1000, 9999)
        
        from path_utils import get_app_data_dir
        self.output_dir = Path(get_app_data_dir()) / "incident_videos"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = Path(get_app_data_dir()) / "incident_recorder.log"
        
        self.last_cleanup = time.time()
        self.fps = 10.0 # Match effective processing frame rate (e.g., 30fps camera / 3 frames skipped = ~10fps)
        self.target_duration = 60 # Stay open for 60 seconds after last detection
        self.min_duration = 45 # Minimum guaranteed duration
        self.buffer_size = int(5 * self.fps) # 5 seconds pre-buffer (150 frames)
        
        self._log(f"🛠\ufe0f Recorder [{self.instance_id}] initialized. Output dir: {self.output_dir}")
        self._log(f"📁 Working Directory: {os.getcwd()}")
        
        # Verify write access
        try:
            test_file = self.output_dir / f"test_{self.instance_id}.tmp"
            test_file.write_text("write test")
            test_file.unlink()
            self._log(f"✅ Write access verified for {self.output_dir}")
        except Exception as e:
            self._log(f"❌ CRITICAL ERROR: No write access to {self.output_dir}: {e}")
        
        # Background closer thread (to finalize videos even if frames stop coming)
        self.running = True
        self.closer_thread = threading.Thread(target=self._closer_loop, daemon=True)
        self.closer_thread.start()
        
        self._migrate_video_names()

    def _closer_loop(self):
        """Background thread to close recordings that have timed out or reached duration"""
        while self.running:
            try:
                now = time.time()
                to_close = []
                for cam_id, rec in list(self.recordings.items()):
                    # Finalize if:
                    # 1. Target duration reached (no recent detection)
                    # 2. Hard cap (max_end_time) reached (long-running incident)
                    # 3. Stream stalled (no frames for 15 seconds)
                    if now > rec['end_time'] or now > rec.get('max_end_time', now + 9999) or (now - rec['last_frame_time'] > 15):
                        to_close.append(cam_id)
                
                for cam_id in to_close:
                    self._log(f"⏰ Finalizing recording for {cam_id} (Duration limit or stream stall)")
                    self._close_recording(cam_id)
            except Exception as e:
                pass
            time.sleep(1.0)

    def _migrate_video_names(self):
        try:
            for file in self.output_dir.glob("*.mp4"):
                if "Violation" not in file.name:
                    parts = file.stem.split("_")
                    if len(parts) >= 3:
                        cam = parts[0]
                        date_str = parts[1]
                        time_str = parts[2]
                        if len(date_str) == 8 and len(time_str) == 6:
                            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                            formatted_time = f"{time_str[:2]}-{time_str[2:4]}-{time_str[4:]}"
                            new_name = f"{formatted_date}_{formatted_time}_{cam}_Violation_UNCATEGORIZED.mp4"
                            new_path = file.parent / new_name
                            if not new_path.exists():
                                file.rename(new_path)
                                self._log(f"🚚 Migrated name: {file.name} -> {new_name}")
        except Exception as e:
            self._log(f"⚠️ Migration error: {e}")

    def _get_retention_days(self):
        try:
            if self.db:
                db_obj = self.db.db if hasattr(self.db, 'db') else self.db
                if hasattr(db_obj, 'system_config'):
                    cfg = db_obj.system_config.find_one({"config_type": "alerts"})
                    if cfg and "video_retention_days" in cfg:
                        return int(cfg.get("video_retention_days", self.default_retention_days))
        except:
            pass
        return self.default_retention_days

    def process_frame(self, camera_id, frame, incident_type=None):
        if frame is None:
            return

        if camera_id not in self.buffers:
            self.buffers[camera_id] = deque(maxlen=self.buffer_size)
        
        self.buffers[camera_id].append(frame.copy())
        
        now = time.time()
        if now - self.last_cleanup > 3600:
            self._cleanup_old_videos()
            self.last_cleanup = now
            
        # Convert bool to string if needed
        if isinstance(incident_type, bool):
            incident_type = "GENERAL_INCIDENT" if incident_type else None

        rec = self.recordings.get(camera_id)
        
        if incident_type:
            if rec is None:
                # Start new recording
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                camera_name = camera_id
                try:
                    db_obj = self.db.db if hasattr(self.db, 'db') else self.db
                    cam_data = db_obj.cameras.find_one({"camera_id": camera_id})
                    if cam_data and cam_data.get("name"):
                        camera_name = cam_data.get("name")
                except:
                    pass

                s_cam = "".join(x for x in camera_name if x.isalnum() or x in " -_").strip().replace(" ", "-")
                s_type = str(incident_type).replace(" ", "_").replace("/", "-")
                filename = f"{ts}_{s_cam}_Violation_{s_type}.mp4"
                final_path = self.output_dir / filename
                # ✅ Use .mp4 but with a prefix. Dashboard will be updated to filter REC_TEMP_
                tmp_path = self.output_dir / f"REC_TEMP_{filename}"
                
                h, w = frame.shape[:2]
                w = w if w % 2 == 0 else w - 1
                h = h if h % 2 == 0 else h - 1
                
                # Extended list of codecs
                codecs = ['mp4v', 'avc1', 'XVID', 'MJPG', 'DIVX', 'H264']
                writer = None
                self._log(f"🎬 Initializing recording for {camera_id} at {w}x{h}, {self.fps}fps")
                for c_str in codecs:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*c_str)
                        temp_writer = cv2.VideoWriter(str(tmp_path), fourcc, self.fps, (w, h))
                        if temp_writer.isOpened():
                            self._log(f"✅ [{camera_id}] VideoWriter opened with codec {c_str}")
                            writer = temp_writer
                            break
                        else:
                            temp_writer.release()
                    except Exception as e:
                        self._log(f"⚠️ [{camera_id}] Codec {c_str} attempt failed: {e}")
                
                if writer:
                    rec = {
                        'writer': writer,
                        'start_time': now,
                        'end_time': now + self.target_duration,
                        'max_end_time': now + 180, # Hard cap at 3 mins per file to ensure playability
                        'filepath': str(final_path),
                        'tmp_path': str(tmp_path),
                        'width': w,
                        'height': h,
                        'last_frame_time': now
                    }
                    self.recordings[camera_id] = rec
                    self._log(f"🔴 RECORDING STARTED: {filename}")
                    
                    # Write pre-incident buffer
                    buf = list(self.buffers[camera_id])
                    self._log(f"📦 Writing {len(buf)} pre-incident frames to {camera_id}")
                    for f in buf:
                        if f.shape[1] != w or f.shape[0] != h:
                            f = cv2.resize(f, (w, h))
                        writer.write(f)
                else:
                    self._log(f"❌ CRITICAL: VideoWriter failed to open for {tmp_path}")
            else:
                # Incident ongoing - extend recording duration
                rec['end_time'] = now + self.target_duration
                # But if we hit the hard cap, we'll let the closer loop finalize it
        
        # Write frame if recording is active
        if rec is not None:
            try:
                f_to_write = frame
                if f_to_write.shape[1] != rec['width'] or f_to_write.shape[0] != rec['height']:
                    f_to_write = cv2.resize(f_to_write, (rec['width'], rec['height']))
                
                rec['writer'].write(f_to_write)
                rec['last_frame_time'] = now
            except Exception as e:
                self._log(f"❌ Write error for {camera_id}: {e}")
                self._close_recording(camera_id)

    def _close_recording(self, camera_id):
        # Use pop to ensure only one closer acts
        rec = self.recordings.pop(camera_id, None)
        if rec:
            try:
                # Ensure minimum duration
                total_dur = time.time() - rec['start_time']
                if total_dur < self.min_duration:
                    # Actually, we could just wait if called via closer, 
                    # but if called via stop_all we close immediately.
                    pass

                time.sleep(0.5) 
                rec['writer'].release()
                
                tmp = Path(rec['tmp_path'])
                final = Path(rec['filepath'])
                
                if tmp.exists():
                    if final.exists():
                        final = final.with_name(f"{final.stem}_{random.randint(100,999)}{final.suffix}")
                    
                    tmp.rename(final)
                    file_size = final.stat().st_size
                    self._log(f"⏹️ RECORDING FINALIZED: {final.name} ({file_size / 1024:.1f} KB, duration approx {total_dur:.1f}s)")
                    
                    if file_size < 1024:
                        self._log(f"⚠️ WARNING: Output file too small ({file_size} bytes) - likely unplayable!")
                else:
                    self._log(f"❌ ERROR: Managed file {tmp} missing!")
            except Exception as e:
                self._log(f"❌ Finalization error for {camera_id}: {e}")

    def stop_all(self):
        self._log("👋 Stopping all recordings...")
        self.running = False
        cam_ids = list(self.recordings.keys())
        for cid in cam_ids:
            self._close_recording(cid)
        
    def _cleanup_old_videos(self):
        try:
            retention_days = self._get_retention_days()
            cutoff = time.time() - (retention_days * 86400)
            count = 0
            if self.output_dir.exists():
                for file in self.output_dir.glob("*.mp4"):
                    if file.is_file() and file.stat().st_mtime < cutoff:
                        try:
                            file.unlink()
                            count += 1
                        except:
                            pass
            if count > 0:
                self._log(f"🧹 Purged {count} videos older than {retention_days} days")
        except Exception as e:
            self._log(f"❌ Cleanup error: {e}")

    def _log(self, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fmsg = f"[{ts}] {msg}\n"
        print(fmsg, end="", flush=True)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(fmsg)
        except:
            pass
