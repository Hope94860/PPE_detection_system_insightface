import cv2
import time
import os
from threading import Thread, Lock
from typing import Dict
from collections import deque


class CameraDetectionManager:
    """
    ULTRA-ROBUST Camera Detection Manager
    ✅ Faster reconnection (reduced timeout)
    ✅ Non-blocking connection attempts
    ✅ Better error handling
    ✅ Health monitoring
    """

    def __init__(self):
        self.latest_frames: Dict[str, any] = {}
        self.capture_threads: Dict[str, Thread] = {}
        self.running_flags: Dict[str, bool] = {}
        self.decode_times: Dict[str, deque] = {}
        self.lock = Lock()
        
        # Stream health tracking
        self.last_frame_time: Dict[str, float] = {}
        self.reconnect_attempts: Dict[str, int] = {}
        self.camera_rtsp_urls: Dict[str, str] = {}  # Store RTSP URLs
        self.frame_counters: Dict[str, int] = {}  # Track frame counts for each camera
        self.target_fps: Dict[str, float] = {} # Store target FPS for throttling

        print("📷 CameraDetectionManager READY")

    def start_detection(self, camera_id: str, rtsp_url: str, target_fps: float = None):
        """Start camera capture thread"""
        if camera_id in self.capture_threads:
            print(f"⚠️ Camera {camera_id} already running")
            return

        self.running_flags[camera_id] = True
        self.decode_times[camera_id] = deque(maxlen=60)
        self.last_frame_time[camera_id] = time.time()
        self.reconnect_attempts[camera_id] = 0
        self.camera_rtsp_urls[camera_id] = rtsp_url
        self.frame_counters[camera_id] = 0
        self.target_fps[camera_id] = target_fps

        t = Thread(
            target=self._capture_loop,
            args=(camera_id, rtsp_url),
            daemon=True
        )
        self.capture_threads[camera_id] = t
        t.start()

        print(f"▶️ Camera started: {camera_id} (Target FPS: {target_fps if target_fps else 'Auto'})")

    def stop_detection(self, camera_id: str):
        """Stop camera capture thread"""
        print(f"🛑 Stopping camera {camera_id}...")
        self.running_flags[camera_id] = False
        
        # Wait briefly for thread to exit
        if camera_id in self.capture_threads:
            thread = self.capture_threads[camera_id]
            thread.join(timeout=3.0)
            
            # Force cleanup if thread didn't exit
            if thread.is_alive():
                print(f"⚠️ Thread for {camera_id} didn't exit cleanly")
            
            del self.capture_threads[camera_id]

        with self.lock:
            self.latest_frames.pop(camera_id, None)
            self.last_frame_time.pop(camera_id, None)
            self.reconnect_attempts.pop(camera_id, None)
            self.camera_rtsp_urls.pop(camera_id, None)

        print(f"✅ Camera stopped: {camera_id}")

    def _create_capture(self, rtsp_url: str, camera_id: str):
        """
        Create VideoCapture with ULTRA-OPTIMIZED settings for low latency
        Returns (cap, success)
        """
        try:
            # ✅ Handle USB Cameras (Integer indices)
            # If the input is a digit (e.g. "0", "1"), convert to int and dont use FFMPEG
            if str(rtsp_url).isdigit():
                print(f"📷 Opening USB Camera {rtsp_url}...")
                cap = cv2.VideoCapture(int(rtsp_url))
            else:
                # RTSP Stream
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                return None, False
            
            # 🚀 CRITICAL: Minimum buffer for lowest latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Single frame buffer = minimal lag
            
            # 🚀 Let camera use native FPS and resolution for best performance
            # Don't force resolution - use what camera provides
            
            # Test read to verify connection
            ret, _ = cap.read()
            if not ret:
                cap.release()
                return None, False
            
            print(f"✅ Successfully connected to {camera_id}")
            return cap, True
            
        except Exception as e:
            print(f"❌ Error creating capture for {camera_id}: {e}")
            return None, False

    def _capture_loop(self, camera_id: str, rtsp_url: str):
        """
        Main capture loop with fast reconnection
        ✅ Non-blocking connection attempts
        ✅ Reduced timeout between retries
        ✅ Better error recovery
        """
        cap = None
        MAX_RECONNECT_ATTEMPTS = 999  # Keep trying indefinitely
        RECONNECT_DELAY = 3  # Reduced from 5s to 3s
        READ_TIMEOUT_THRESHOLD = 5  # If no frame for 5s, reconnect
        consecutive_failures = 0
        
        while self.running_flags.get(camera_id, False):
            try:
                # Initialize or reconnect
                if cap is None or not cap.isOpened():
                    print(f"🔌 Connecting to {camera_id}...")
                    
                    # Release old capture if exists
                    if cap is not None:
                        try:
                            cap.release()
                        except:
                            pass
                    
                    # Try to create new capture
                    cap, success = self._create_capture(rtsp_url, camera_id)
                    
                    if success and self.target_fps.get(camera_id) is None:
                        # Try to detect FPS from source if not provided
                        source_fps = cap.get(cv2.CAP_PROP_FPS)
                        if source_fps > 0 and source_fps < 1000: # Sanity check
                            # Only auto-throttle if it's likely a file (source_fps is exactly something like 23.97, 25, 30, 60)
                            # Or if the path looks like a file
                            is_file = os.path.isfile(rtsp_url)
                            if is_file:
                                print(f"ℹ️ Detected file source for {camera_id}, using file FPS: {source_fps}")
                                self.target_fps[camera_id] = source_fps
                    
                    if not success:
                        self.reconnect_attempts[camera_id] += 1
                        consecutive_failures += 1
                        
                        # Exponential backoff for persistent failures
                        if consecutive_failures > 3:
                            delay = min(RECONNECT_DELAY * 2, 10)  # Max 10s delay
                        else:
                            delay = RECONNECT_DELAY
                        
                        print(f"⚠️ Failed to connect to {camera_id}, "
                              f"retry in {delay}s (attempt {self.reconnect_attempts[camera_id]})")
                        
                        time.sleep(delay)
                        continue
                    
                    # Reset failure counter on success
                    consecutive_failures = 0
                    self.reconnect_attempts[camera_id] = 0
                
                # Try to read frame
                start_time = time.time()
                ret, frame = cap.read()
                
                if not ret:
                    # ✅ If this is a local video file, loop it instead of reconnecting
                    is_file = os.path.isfile(rtsp_url)
                    if is_file:
                        print(f"🔁 {camera_id}: video file ended, looping back to start")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        consecutive_failures = 0
                        continue

                    consecutive_failures += 1
                    
                    # Check if we should reconnect (RTSP/USB only)
                    time_since_frame = time.time() - self.last_frame_time.get(camera_id, 0)
                    
                    if time_since_frame > READ_TIMEOUT_THRESHOLD or consecutive_failures > 10:
                        print(f"⚠️ No frames from {camera_id} for {time_since_frame:.1f}s, reconnecting...")
                        if cap is not None:
                            cap.release()
                        cap = None
                        time.sleep(1)
                    else:
                        time.sleep(0.1)
                    
                    continue
                
                # Successfully read frame
                consecutive_failures = 0
                now = time.time()
                
                # Update latest frame (zero-copy)
                with self.lock:
                    self.latest_frames[camera_id] = frame
                    self.last_frame_time[camera_id] = now
                    self.frame_counters[camera_id] = self.frame_counters.get(camera_id, 0) + 1
                
                self.decode_times[camera_id].append(now)
                
                # 🚀 FPS Throttling (especially for video files)
                target_fps = self.target_fps.get(camera_id)
                if target_fps and target_fps > 0:
                    elapsed = time.time() - start_time
                    sleep_time = (1.0 / target_fps) - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # cv2.VideoCapture.read() blocks appropriately for RTSP but not for files
                
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Exception in {camera_id}: {e}")
                consecutive_failures += 1
                
                if cap is not None:
                    try:
                        cap.release()
                    except:
                        pass
                    cap = None
                
                time.sleep(RECONNECT_DELAY)
        
        # Cleanup
        if cap is not None:
            try:
                cap.release()
            except:
                pass
        
        print(f"🛑 Capture loop ended for {camera_id}")

    def get_decode_fps(self):
        """Get decode FPS for all cameras"""
        fps = {}
        for cam_id, dq in self.decode_times.items():
            if len(dq) >= 2:
                dt = dq[-1] - dq[0]
                fps[cam_id] = round((len(dq) - 1) / dt, 2) if dt > 0 else 0.0
            else:
                fps[cam_id] = 0.0
        return fps
    
    def get_stream_health(self, camera_id: str = None) -> dict:
        """
        Get health status of camera stream(s)
        If camera_id is None, returns health for all cameras
        """
        if camera_id:
            return self._get_single_camera_health(camera_id)
        
        # Return health for all cameras
        health = {}
        for cam_id in self.running_flags.keys():
            health[cam_id] = self._get_single_camera_health(cam_id)
        return health
    
    def _get_single_camera_health(self, camera_id: str) -> dict:
        """Get health status of a single camera"""
        if camera_id not in self.running_flags:
            return {"status": "not_running", "fps": 0.0}
        
        now = time.time()
        last_frame = self.last_frame_time.get(camera_id, 0)
        time_since_frame = now - last_frame
        
        # Determine status
        if time_since_frame > 30:
            status = "timeout"
        elif time_since_frame > 10:
            status = "degraded"
        elif time_since_frame > 5:
            status = "reconnecting"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "time_since_last_frame": round(time_since_frame, 1),
            "reconnect_attempts": self.reconnect_attempts.get(camera_id, 0),
            "fps": self.get_decode_fps().get(camera_id, 0.0),
            "is_running": self.running_flags.get(camera_id, False)
        }
    
    def restart_camera(self, camera_id: str):
        """
        Force restart a camera
        Useful for manual intervention
        """
        if camera_id not in self.camera_rtsp_urls:
            print(f"❌ Cannot restart {camera_id}: not found")
            return False
        
        rtsp_url = self.camera_rtsp_urls[camera_id]
        target_fps = self.target_fps.get(camera_id)
        
        print(f"🔄 Restarting camera {camera_id}...")
        self.stop_detection(camera_id)
        time.sleep(2)
        self.start_detection(camera_id, rtsp_url, target_fps=target_fps)
        
        return True
    
    def get_all_camera_status(self) -> dict:
        """
        Get comprehensive status for all cameras
        Useful for monitoring dashboard
        """
        status = {}
        
        for cam_id in self.running_flags.keys():
            health = self._get_single_camera_health(cam_id)
            
            status[cam_id] = {
                "camera_id": cam_id,
                "rtsp_url": self.camera_rtsp_urls.get(cam_id, "unknown"),
                "is_active": self.running_flags.get(cam_id, False),
                "health": health["status"],
                "fps": health["fps"],
                "time_since_frame": health["time_since_last_frame"],
                "reconnect_attempts": health["reconnect_attempts"]
            }
        
        return status