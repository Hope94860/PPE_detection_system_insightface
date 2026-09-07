import time
import random

print("📂 [InferenceController] Module loaded and ready.")
import threading
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List
import numpy as np
from threading import Lock, Thread
from queue import Queue, Empty


class InferenceController(Thread):
    """
    ✅ FIXED: Run face recognition on EVERY frame
    ✅ HIGH PERFORMANCE: Decoupled Inference & Result Processing
    No throttling - maximum recognition speed
    """

    def __init__(self, camera_manager, integrated_system, max_batch_size=8):
        super().__init__(daemon=True)

        self.camera_manager = camera_manager
        self.system = integrated_system
        self.max_batch_size = max_batch_size
        self.paused = False

        self.running = True
        self.lock = Lock()
        self.live_status = {}
        self.persons_in_frame = {}  # camera_id -> int: raw body-detector count

        # FPS tracking
        self.inference_times = deque(maxlen=60)
        self.per_camera_times = defaultdict(lambda: deque(maxlen=30))
        
        self.latest_results = {}
        self.live_violations = {}
        self.annotated_frames = {}
        self.annotated_frame_counts = defaultdict(int)
        self.raw_frames = {}
        self.raw_frame_counts = defaultdict(int)
        self.last_processed_frame_indices = defaultdict(int) # Track frame counts from camera manager
        
        # ✅ Decoupled Result Processing
        self.result_queue = Queue(maxsize=30)
        self.processor_thread = Thread(target=self._result_processing_loop, daemon=True)
        self.processor_thread.start()

        from incident_recorder import IncidentVideoRecorder
        self.incident_recorder = IncidentVideoRecorder(self.system.ppe_system.db, default_retention_days=7)

        print(
            f"🧠 InferenceController READY | "
            f"batch={self.max_batch_size} | "
            f"mode=CONTINUOUS | "
            f"threads=GPU+CPU_POST_PROCESS"
        )

    def run(self):
        print("🚀 InferenceController started - CONTINUOUS RECOGNITION MODE")

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            try:
                # Get latest frames
                with self.camera_manager.lock:
                    frames_snapshot = dict(self.camera_manager.latest_frames)

                if not frames_snapshot:
                    time.sleep(0.01)
                    continue

                batch_frames = []
                batch_camera_ids = []

                for cam_id, frame in frames_snapshot.items():
                    if frame is None:
                        continue
                        
                    # 🚀 Only process if it's a new frame from CameraDetectionManager
                    current_idx = self.camera_manager.frame_counters.get(cam_id, 0)
                    if current_idx <= self.last_processed_frame_indices[cam_id]:
                        continue
                        
                    batch_frames.append(frame)
                    batch_camera_ids.append(cam_id)
                    self.last_processed_frame_indices[cam_id] = current_idx

                if not batch_frames:
                    time.sleep(0.005) # Reduced sleep for better responsiveness
                    continue

                # ✅ ALWAYS run face recognition
                run_face = True
                
                # Run inference (GPU Heavy)
                inference_start = time.time()
                results = self.system.process_frames_with_ppe_batch(
                    batch_frames,
                    batch_camera_ids,
                    run_face=run_face
                )
                inference_end = time.time()

                # Update FPS tracking (Inference Only)
                with self.lock:
                    self.inference_times.append(inference_end)
                    for cam_id in batch_camera_ids:
                        self.per_camera_times[cam_id].append(inference_end)

                # Push to Queue for Side-Effect Processing (CPU Heavy)
                if not self.result_queue.full():
                    self.result_queue.put((results, batch_frames, batch_camera_ids))
                else:
                    # If queue full, we skip post-processing to keep up with live feed
                    pass
                    
                # Very small sleep to yield GIL
                time.sleep(0.001)

            except Exception as e:
                print(f"❌ InferenceController error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.05)

    def _result_processing_loop(self):
        """Separate thread to handle drawing and status updates"""
        while self.running:
            try:
                item = self.result_queue.get(timeout=1)
            except Empty:
                continue
                
            if item is None: break
            
            results, frames, camera_ids = item
            
            # 💓 Heartbeat: Trace processor activity
            if random.random() < 0.05:
                print(f"💓 [InferenceController] Processing batch: {camera_ids}")

            incident_frames_to_record = []
            
            try:
                # Store results and update live_status
                with self.lock:
                    for idx, cam_id in enumerate(camera_ids):
                        result = results[idx]
                        frame = frames[idx]
                        
                        # High-volume trace
                        if random.random() < 0.05:
                            print(f"🕵\uFE0F [InferenceController] Processing frame for {cam_id}. Compliance results: {len(result.get('compliance_results', []))}")

                        # Store full results
                        self.latest_results[cam_id] = result

                        # Always create annotated frame for live view
                        annotated = self.system.draw_results(frame, result)

                        self.annotated_frames[cam_id] = annotated
                        self.annotated_frame_counts[cam_id] += 1
                        
                        # Store raw frame for raw-view endpoints (like OCR page)
                        self.raw_frames[cam_id] = frame.copy()
                        self.raw_frame_counts[cam_id] += 1
                        
                        incident_type = None

                        # ✅ Count persons detected by body detector (YOLO), not faces
                        # This is the ground-truth number of people physically in the frame
                        body_count = sum(
                            1 for det in result.get("ppe_detections", [])
                            if det.get("category") in ("person", "skeleton")
                        )
                        self.persons_in_frame[cam_id] = body_count

                        # Update live_status for status panel
                        if cam_id not in self.live_status:
                            self.live_status[cam_id] = {}

                        status_map = self.live_status[cam_id]
                        current_time = datetime.now()  # Local system time (IST)

                        # Add/update all detected persons
                        for r in result.get("compliance_results", []):
                            try:
                                track_id = r.get("track_id") or r.get("person_id", "unknown")
                                log_type = r.get("log_type")

                                # ✅ Safe field extraction with defaults (handles faceless night intruder entries)
                                compliance_data = r.get("compliance") or {}
                                missing_ppe = compliance_data.get("missing_ppe", [])
                                is_compliant = compliance_data.get("is_compliant", False)
                                is_unknown = r.get("is_unknown", True)

                                # -------------------------------------------------
                                # 🚨 TRIGGER ALERT ON VIOLATION
                                # -------------------------------------------------
                                snapshot_url = None
                                zone_snapshot_url = None
                                ppe_snapshot_url = None
                                
                                # Check for PPE/Night Mode violations
                                if r.get("is_violation"):
                                    violation_log_id = r.get("violation_log_id")
                                    
                                    # Get snapshot_url from the database record
                                    if violation_log_id:
                                        try:
                                            from bson import ObjectId
                                            log_record = self.system.ppe_system.db.recognition_logs.find_one(
                                                {"_id": ObjectId(violation_log_id)}
                                            )
                                            if log_record:
                                                ppe_snapshot_url = log_record.get("snapshot_url")
                                                if ppe_snapshot_url:
                                                    snapshot_url = ppe_snapshot_url
                                                    print(f"📷 Retrieved {log_type or 'PPE'} violation snapshot URL from DB: {snapshot_url}")
                                        except Exception as e:
                                            print(f"⚠️ Failed to retrieve snapshot URL: {e}")
                                    
                                    # Trigger alert for WhatsApp/SMS notifications
                                    violation = {
                                        "camera_id": cam_id,
                                        "person_id": r.get("person_id", track_id),
                                        "person_name": r.get("person_name", "Unknown"),
                                        "missing_ppe": missing_ppe,
                                        "is_unknown": is_unknown,
                                        "log_id": r.get("violation_log_id"),
                                    }

                                    try:
                                        if log_type == "night_mode_violation":
                                            incident_type = "Night_Intruder"
                                            self.system.alert_engine.handle_night_mode_violation(violation)
                                            print(f"🌙 Night Mode alert notification sent")
                                        elif log_type in ["fire_violation", "fall_violation"]:
                                            # Already handled directly in optimized_ppe_detection.py via handle_fire_smoke_alert / handle_fall_alert
                                            pass
                                        else:
                                            if incident_type is None:
                                                incident_type = "PPE_Violation"
                                            self.system.alert_engine.handle_violation(violation)
                                            print(f"✅ Alert notification sent")
                                    except Exception as e:
                                        print("⚠️ Alert trigger error:", e)
                                        import traceback
                                        traceback.print_exc()
                                
                                # Check for zone violations
                                if not r.get("is_zone_authorized", True):
                                    zone_violation_log_id = r.get("zone_violation_log_id")
                                    
                                    if zone_violation_log_id:
                                        try:
                                            from bson import ObjectId
                                            log_record = self.system.ppe_system.db.recognition_logs.find_one(
                                                {"_id": ObjectId(zone_violation_log_id)}
                                            )
                                            if log_record:
                                                zone_snapshot_url = log_record.get("snapshot_url")
                                                if zone_snapshot_url:
                                                    snapshot_url = zone_snapshot_url
                                                    print(f"📷 Retrieved zone violation snapshot URL from DB: {snapshot_url}")
                                        except Exception as e:
                                            print(f"⚠️ Failed to retrieve zone snapshot URL: {e}")

                                status_map[track_id] = {
                                    "track_id": track_id,
                                    "person_id": r.get("person_id", track_id),
                                    "person_name": r.get("person_name", "Unknown"),
                                    "role": r.get("role", "Unknown"),
                                    "is_unknown": is_unknown,
                                    "missing_ppe": missing_ppe,
                                    "is_compliant": is_compliant,
                                    "is_zone_authorized": r.get("is_zone_authorized", True),
                                    "zone_name": r.get("zone_name"),
                                    "log_type": log_type,
                                    # ✅ Include violation_log_id so the UI can use it as a unique
                                    # deduplication key — each new violation event auto-appears in the panel
                                    "violation_log_id": r.get("violation_log_id"),
                                    "zone_violation_log_id": r.get("zone_violation_log_id"),
                                    "timestamp": current_time.isoformat(),
                                    "last_seen": time.time(),
                                    "snapshot_url": snapshot_url,
                                    "ppe_snapshot_url": ppe_snapshot_url,
                                    "zone_snapshot_url": zone_snapshot_url,
                                    "fire_category": r.get("fire_category")
                                }
                            except Exception as entry_err:
                                print(f"⚠️ Skipping compliance result entry due to error: {entry_err}")
                        # ✅ NEW: Add fire/smoke detections to status map for real-time UI display
                        # Get camera zone from db
                        cam_zone = "Unknown Location"
                        try:
                            cam_info = self.system.ppe_system.db.get_camera(cam_id)
                            if cam_info:
                                cam_zone = cam_info.get("location") or cam_info.get("zone") or cam_id
                        except Exception:
                            pass
                        
                        for fd in result.get("fire_smoke_detections", []):
                            incident_type = "Fire_Smoke"
                            print(f"🚨 [InferenceController] TRACING: Fire/Smoke detected, setting incident_type={incident_type}")
                            cat = fd["category"].upper()
                            event_key = f"EMERGENCY_{cat}_{idx}" # use index to distinguish if multiple cameras
                            status_map[event_key] = {
                                "track_id": event_key,
                                "person_id": event_key,
                                "person_name": f"🚨 {cat} DETECTED!",
                                "role": "EMERGENCY",
                                "is_unknown": False,
                                "missing_ppe": [f"HIGH CONFIDENCE: {fd['confidence']*100:.1f}%"],
                                "is_compliant": False,
                                "log_type": "fire_smoke_alert",
                                "timestamp": current_time.isoformat(),
                                "last_seen": time.time(),
                                "is_emergency": True,
                                "zone_name": cam_zone
                            }

                        # ✅ FIX: Add fall detections to status_map for real-time UI display
                        # Previously, fall detections were ONLY setting incident_type for the video recorder
                        # but were never injected into status_map, so the UI never saw them.
                        for fd in result.get("fall_detections", []):
                            incident_type = "Fall_Detected"
                            print(f"🚨 [InferenceController] TRACING: Fall detected, setting incident_type={incident_type}")
                            event_key = f"EMERGENCY_FALL_{idx}"
                            status_map[event_key] = {
                                "track_id": event_key,
                                "person_id": event_key,
                                "person_name": "🆘 PERSON LYING ON GROUND DETECTED!",
                                "role": "EMERGENCY",
                                "is_unknown": False,
                                "missing_ppe": [f"CONFIDENCE: {fd.get('confidence', 0)*100:.1f}%"],
                                "is_compliant": False,
                                "log_type": "fall_violation",
                                "timestamp": current_time.isoformat(),
                                "last_seen": time.time(),
                                "is_emergency": True,
                                "zone_name": cam_zone
                            }

                        # ✅ NEW: Add plate results to status map for OCR UI
                        for pr in result.get("plate_results", []):
                            plate_num = pr.get("plate_number", "UNKNOWN")
                            event_key = f"PLATE_{plate_num}_{idx}"
                            status_map[event_key] = {
                                "track_id": event_key,
                                "person_id": event_key,
                                "person_name": f"🔢 Plate: {plate_num}",
                                "role": "VEHICLE",
                                "is_unknown": False,
                                "plate_number": plate_num,
                                "confidence": pr.get("confidence", 0),
                                "log_type": "plate_detection",
                                "timestamp": current_time.isoformat(),
                                "last_seen": time.time(),
                                "is_ocr": True,
                                "violation_log_id": pr.get("violation_log_id")
                            }

                        # -------------------------------------------------
                        # TTL cleanup:
                        # - Emergency events (fire/fall) expire after 30s so the banner
                        #   stays visible long enough for the operator to see it.
                        # - All other events expire after 10s.
                        # -------------------------------------------------
                        now = time.time()
                        stale_keys = [
                            k for k, v in status_map.items()
                            if (
                                now - v.get("last_seen", 0) > 30
                                if v.get("log_type") in ("fire_smoke_alert", "fall_violation", "fire_violation")
                                else now - v.get("last_seen", 0) > 10
                            )
                        ]
                        for k in stale_keys:
                            del status_map[k]
                            
                        # ✅ ALWAYS Queue for video recorder:
                        # Even if incident_type is None, we send the frame to maintain the pre-buffer
                        incident_frames_to_record.append((cam_id, annotated.copy(), incident_type))
                        if incident_type:
                             print(f"🔥 [InferenceController] Incident '{incident_type}' QUEUED for recording on {cam_id}")
                                                
                                                    
            except Exception as e:
                print(f"❌ Result processing error: {e}")
            finally:
                self.result_queue.task_done()
                
            # Now process video writing outside the lock
            for cam_id, annotated_frame, incident_type in incident_frames_to_record:
                try:
                    if incident_type:
                        print(f"🔥 [InferenceController] Incident detected on {cam_id}: {incident_type}")
                    self.incident_recorder.process_frame(cam_id, annotated_frame, incident_type)
                except Exception as e:
                    print(f"❌ Error recording incident frame for {cam_id}: {e}")

    def get_latest_frame(self, camera_id: str):
        """Get annotated frame for /api/stream endpoint"""
        with self.lock:
            return self.annotated_frames.get(camera_id)

    def get_latest_frame_with_count(self, camera_id: str):
        """Get annotated frame and its update count for optimization"""
        with self.lock:
            return self.annotated_frames.get(camera_id), self.annotated_frame_counts.get(camera_id, 0)

    def get_latest_raw_frame_with_count(self, camera_id: str):
        """Get raw frame and its update count for raw streaming directly from camera manager for zero latency"""
        if hasattr(self, 'camera_manager') and self.camera_manager:
            with self.camera_manager.lock:
                frame = self.camera_manager.latest_frames.get(camera_id)
                count = self.camera_manager.frame_counters.get(camera_id, 0)
                if frame is not None:
                    return frame.copy(), count
                return None, count
        
        # Fallback if camera_manager is somehow unavailable
        with self.lock:
            return self.raw_frames.get(camera_id), self.raw_frame_counts.get(camera_id, 0)

    def get_inference_fps(self, per_camera=False):
        """Calculate inference FPS"""
        with self.lock:
            if per_camera:
                fps = {}
                for cam_id, dq in self.per_camera_times.items():
                    if len(dq) >= 2:
                        time_span = dq[-1] - dq[0]
                        fps[cam_id] = round((len(dq) - 1) / time_span, 2) if time_span > 0 else 0.0
                    else:
                        fps[cam_id] = 0.0
                return fps

            if len(self.inference_times) < 2:
                return 0.0

            time_span = self.inference_times[-1] - self.inference_times[0]
            count = len(self.inference_times) - 1
            
            return round(count / time_span, 2) if time_span > 0 else 0.0

    
    def get_live_status(self, camera_id: str):
        """Return live detection status for UI"""
        with self.lock:
            status_dict = self.live_status.get(camera_id, {})
            
            # ✅ Use body-detector count (YOLO persons in frame) as the overlay number.
            # Falls back to compliance_results count if body detector hasn't run yet.
            now = time.time()
            status_list = list(status_dict.values())

            # Body-detector count: persons physically visible — independent of face recognition
            body_count = self.persons_in_frame.get(camera_id, None)
            if body_count is None:
                # Fallback: count recent compliance entries (face-based) until body count is available
                body_count = sum(
                    1 for d in status_list
                    if now - d.get("last_seen", 0) < 1.5
                    and d.get("log_type") not in ("fire_violation", "fall_violation",
                                                   "fire_smoke_alert", "plate_detection")
                )

            # ✅ Sort with priority: fire violations FIRST, then by recency
            violation_priority = {
                'fire_violation': 1000,      # 🔥 HIGHEST PRIORITY
                'fall_violation': 500,       # 🆘 HIGH PRIORITY
                'night_mode_violation': 100,
                'zone_violation': 10,
                'ppe_violation': 1,
                'fire_smoke_alert': 1000,   # Emergency alerts too
                'plate_detection': 0
            }
            
            def sort_key(item):
                log_type = item.get('log_type', '')
                priority = violation_priority.get(log_type, 0)
                last_seen = item.get('last_seen', 0)
                return (-priority, -last_seen)
            
            sorted_history = sorted(status_list, key=sort_key)
            
            return {
                "history": sorted_history[:20],
                "current_count": body_count  # ✅ Now reflects persons in frame, not faces detected
            }

    def stop(self):
        self.running = False
        try:
            if hasattr(self, 'incident_recorder') and self.incident_recorder:
                self.incident_recorder.stop_all()
        except Exception:
            pass
        print("🛑 InferenceController stopped")
    
    def pause(self):
        self.paused = True
        print("⏸ InferenceController paused")

    def resume(self):
        self.paused = False
        print("▶ InferenceController resumed")