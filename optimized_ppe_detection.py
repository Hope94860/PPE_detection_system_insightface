
import cv2, time, hashlib, math, threading
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from threading import Lock
from collections import defaultdict, deque
import torch
from ultralytics import YOLO
import os

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class OptimizedPPEDetectionSystem:
    """
    ✅ OPTIMIZED: Enhanced PPE Detection with improved accuracy
    
    Key Improvements:
    1. Multi-confidence tier system for different PPE types
    2. Enhanced spatial verification with distance-based tolerance
    3. Temporal PPE tracking to reduce false negatives
    4. Better occlusion handling
    5. Improved status reporting with detailed metrics
    """

    PPE_CLASSES = {
        'safety_helmet': ['safety_helmet', 'helmet', 'hardhat'],
        'reflective_vest': ['reflective_vest', 'safety_vest', 'vest', 'safety_jacket', 'jacket', 'reflective_jacket'],
        'gloves': ['gloves'],
        'boots': ['boots'],
        'safety_goggles': ['safety_goggles', 'goggles'],
        'face_mask': ['face_mask', 'mask'],
        'welding_mask': ['welding_mask'],
        'safety_harness': ['safety_harness', 'harness'],
        'apron': ['apron'],
        'hearing_muff': ['hearing_muff', 'ear_muffs'],
        'suit': ['suit'],
        'person': ['person']
    }
    
    # ✅ NEW: PPE priority levels (higher = more critical)
    PPE_PRIORITY = {
        'safety_helmet': 10,
        'reflective_vest': 9,
        'safety_harness': 10,
        'welding_mask': 9,
        'safety_goggles': 8,
        'face_mask': 7,
        'gloves': 6,
        'boots': 5,
        'hearing_muff': 6,
        'apron': 5,
        'suit': 6,
        'person': 1
    }

    # ✅ NEW: Color ranges for PPE Role Inference (HSV)
    # H: 0-179, S: 0-255, V: 0-255
    COLOR_RANGES = {
        'yellow': ([20, 100, 100], [35, 255, 255]),
        'white': ([0, 0, 180], [179, 60, 255]), # High brightness, low saturation
        'red1': ([0, 100, 100], [10, 255, 255]),
        'red2': ([170, 100, 100], [179, 255, 255]),
        'blue': ([100, 150, 0], [140, 255, 255]),
        'green': ([36, 100, 100], [86, 255, 255]),
        'orange': ([11, 100, 100], [19, 255, 255])
    }

    # Role mapping based on helmet color
    ROLE_COLOR_MAPPING = {
        'yellow': 'worker',
        'white': 'engineer', # or Supervisor
        'red': 'visitor', # or Firefighter
        'blue': 'electrician',
        'green': 'safety_officer',
        'orange': 'visitor'
    }

    def __init__(self, model_path: str, confidence_threshold: float, db, use_cuda: bool = True, 
                 person_model_path: str = None, fire_model_path: str = None,
                 plate_model_path: str = None, ocr_model_path: str = None, pose_model_path: str = None):
        print("🦺 Initializing Optimized PPE Detection System...")

        self.db = db
        self.base_confidence_threshold = confidence_threshold
        self.device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'

        self.role_ppe_requirements = self._load_ppe_rules_from_db()

        os.environ['YOLO_VERBOSE'] = 'False'
        self.model = YOLO(model_path, task='detect')

        # ✅ NEW: Optional specialized person detection model
        self.person_model = None
        if person_model_path and os.path.exists(person_model_path):
            print(f"🕵️ Loading specialized person detection model: {person_model_path}")
            self.person_model = YOLO(person_model_path, task='detect')

        # ✅ NEW: Optional specialized fire/smoke detection model
        self.fire_model = None
        if fire_model_path and os.path.exists(fire_model_path):
            print(f"🔥 Loading fire and smoke detection model: {fire_model_path}")
            self.fire_model = YOLO(fire_model_path, task='detect')

        # ✅ NEW: Optional Plate Detection Model
        self.plate_model = None
        if plate_model_path and os.path.exists(plate_model_path):
            print(f"📄 Loading plate detection model: {plate_model_path}")
            self.plate_model = YOLO(plate_model_path, task='detect')

        # ✅ NEW: Optional OCR Model (for character recognition)
        self.ocr_model = None
        if ocr_model_path and os.path.exists(ocr_model_path):
            print(f"📑 Loading YOLO OCR recognition model: {ocr_model_path}")
            self.ocr_model = YOLO(ocr_model_path, task='detect')

        # ✅ NEW: Optional Pose Detection Model for Fall Detection
        self.pose_model = None
        if pose_model_path:
            print(f"🧍 Loading pose detection model: {pose_model_path}")
            self.pose_model = YOLO(pose_model_path, task='pose')
            print(f"✅ Pose detection model loaded successfully on {self.device}")

        # ✅ NEW: EasyOCR Reader
        self.ocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                print("📚 Initializing EasyOCR Reader (English)...")
                self.ocr_reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
            except Exception as e:
                print(f"⚠️ Failed to initialize EasyOCR: {e}")

        self.lock = Lock()
        
        # ✅ OPTIMIZED: Multi-tier confidence thresholds based on PPE criticality
        self.ppe_confidence_tiers = {
            'critical': {  # Helmet, harness, welding mask
                'safety_helmet': 0.35,
                'safety_harness': 0.35,
                'welding_mask': 0.40,
            },
            'high': {  # Vest, goggles, face mask
                'reflective_vest': 0.30,
                'safety_goggles': 0.35,
                'face_mask': 0.35,
            },
            'medium': {  # Gloves, boots, hearing protection
                'gloves': 0.25,
                'boots': 0.25,
                'hearing_muff': 0.30,
                'person': 0.25,
            },
            'low': {  # Apron, suit
                'apron': 0.25,
                'suit': 0.25,
            }
        }
        
        # ✅ NEW: Temporal PPE tracking for stability
        self.ppe_temporal_tracking = defaultdict(lambda: defaultdict(lambda: deque(maxlen=10)))
        self.ppe_tracking_lock = Lock()
        
        # ✅ ENHANCED: Distance-adaptive spatial verification tolerances w/ STRICT ANATOMICAL CHECKS
        self.spatial_verification_params = {
            "safety_helmet": {
                "horizontal_tolerance": 1.5,    # Stricter: must be centered on head
                "vertical_range": (-3.0, -0.2), # STRICT: Must be ABOVE face center (negative detections only)
                "min_confidence_boost": 0.05,
            },
            "reflective_vest": {
                "horizontal_tolerance": 3.0,
                "vertical_range": (0.5, 8.0),   # STRICT: Must be BELOW face center (torso)
                "min_confidence_boost": 0.03,
            },
            "safety_goggles": {
                "horizontal_tolerance": 1.2,
                "vertical_range": (-0.8, 0.5),  # Strictly eye level
                "min_confidence_boost": 0.08,
            },
            "face_mask": {
                "horizontal_tolerance": 1.2,
                "vertical_range": (0.0, 1.0),   # Strictly lower face
                "min_confidence_boost": 0.08,
            },
            "welding_mask": {
                "horizontal_tolerance": 1.5,
                "vertical_range": (-1.5, 1.0),
                "min_confidence_boost": 0.07,
            },
            "gloves": {
                "horizontal_tolerance": 6.0,    # Wide tolerance for arms
                "vertical_range": (0.5, 12.0),  # Must be below face
                "min_confidence_boost": 0.02,
            },
            "boots": {
                "horizontal_tolerance": 4.0,
                "vertical_range": (4.0, 15.0),  # Far below face
                "min_confidence_boost": 0.02,
            },
            "safety_harness": {
                "horizontal_tolerance": 3.0,
                "vertical_range": (0.5, 8.0),   # Torso region like vest
                "min_confidence_boost": 0.04,
            },
        }

        print("✅ Optimized PPE Detection System initialized")
        print(f"   Multi-tier confidence system enabled")
        print(f"   Temporal tracking enabled (10-frame buffer)")

    def _load_ppe_rules_from_db(self) -> dict:
        config = self.db.system_config.find_one({"config_type": "ppe_rules"})
        if not config:
            return {
                "default": ["safety_helmet", "reflective_vest"],
                "visitor": ["safety_helmet", "reflective_vest"]
            }
        return config.get("role_rules", {})

    def reload_ppe_rules(self):
        self.role_ppe_requirements = self._load_ppe_rules_from_db()
        print("🔄 PPE rules reloaded")

    def _map_to_ppe_category(self, class_name: str) -> Optional[str]:
        for category, aliases in self.PPE_CLASSES.items():
            if class_name in aliases:
                return category
        return None
    
    def _get_ppe_confidence_threshold(self, category: str) -> float:
        """
        ✅ NEW: Get adaptive confidence threshold for PPE category
        """
        # Check each tier
        for tier_name, tier_thresholds in self.ppe_confidence_tiers.items():
            if category in tier_thresholds:
                return tier_thresholds[category]
        
        # Fallback to base threshold
        return self.base_confidence_threshold

    def _detect_dominant_color(self, frame: np.ndarray, bbox: Dict[str, int]) -> str:
        """
        ✅ NEW: Detect dominant color of PPE item
        """
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1: return "unknown"
        
        roi = frame[y1:y2, x1:x2]
        
        # Focus on center 50% to minimize background
        rh, rw = roi.shape[:2]
        qy = int(rh * 0.25)
        qx = int(rw * 0.25)
        center_roi = roi[qy:rh-qy, qx:rw-qx]
        
        if center_roi.size == 0: return "unknown"
        
        try:
            hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
            
            # Color ranges (H: 0-180, S: 0-255, V: 0-255)
            # Tuned for common PPE colors
            ranges = {
                "Red": [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
                "Orange": [((11, 70, 50), (25, 255, 255))],
                "Yellow": [((26, 70, 50), (35, 255, 255))],
                "Green": [((36, 70, 50), (85, 255, 255))],
                "Blue": [((86, 70, 50), (130, 255, 255))],
                "White": [((0, 0, 180), (180, 50, 255))], # High V, Low S
                "Black": [((0, 0, 0), (180, 255, 50))],   # Low V
            }
            
            max_pixels = 0
            dominant_color = "unknown"
            
            for color, ranges_list in ranges.items():
                pixels = 0
                for (lower, upper) in ranges_list:
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    pixels += cv2.countNonZero(mask)
                
                if pixels > max_pixels:
                    max_pixels = pixels
                    dominant_color = color
            
            # Require at least 20% dominance
            total_pixels = center_roi.shape[0] * center_roi.shape[1]
            if max_pixels < total_pixels * 0.2:
                 return "unknown"
                 
            return dominant_color
            
        except Exception:
            return "unknown"

    def detect_ppe(self, frame: np.ndarray) -> List[Dict]:
        """
        ✅ OPTIMIZED: Enhanced PPE detection with adaptive thresholds
        """
        with self.lock:
            # Use lower base confidence, we'll filter by category-specific thresholds
            results = self.model.predict(frame, conf=0.20, verbose=False)
            detections = []
            
            for result in results:
                for box in result.boxes:
                    class_name = self.model.names[int(box.cls[0])].lower()
                    category = self._map_to_ppe_category(class_name)
                    
                    if not category:
                        continue
                    
                    confidence = float(box.conf[0])
                    
                    # ✅ Apply category-specific threshold
                    min_threshold = self._get_ppe_confidence_threshold(category)
                    
                    if confidence < min_threshold:
                        continue
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        "category": category,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                        "center": {"x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2)},
                        "area": int((x2 - x1) * (y2 - y1)),
                        "priority": self.PPE_PRIORITY.get(category, 5)
                    }
                    
                    # ✅ NEW: Detect color for helmets and vests
                    if category in ["safety_helmet", "reflective_vest", "suit", "helmet", "vest"]:
                        detection["color"] = self._detect_dominant_color(frame, detection["bbox"])
                    
                    detections.append(detection)
            
            return detections
    
    def detect_fire_smoke(self, frame: np.ndarray) -> List[Dict]:
        """
        ✅ NEW: Run fire and smoke detection
        """
        if not self.fire_model:
            # DEBUG: Print error if model missing
            print("❌ CRITICAL: Fire model not loaded (self.fire_model is None)! Check models/fire_smoke_model.onnx path.")
            return []
            
        with self.lock:
            # Lower model-level threshold to 0.20 so we capture even faint candidates
            results = self.fire_model.predict(frame, conf=0.20, verbose=False)
            detections = []
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.fire_model.names[class_id].lower()
                    # We expect "fire" and "smoke" classes
                    confidence = float(box.conf[0])
                    
                    # DEBUG: Print all raw detections to investigate specific videos
                    print(f"🔍 RAW FIRE/SMOKE Candidate: {class_name} ({confidence*100:.1f}%)")
                    
                    # ✅ ENHANCED SENSITIVITY: Fire > 75%, Smoke > 60%, Spark > 85%
                    lower_class = class_name.lower()
                    if 'fire-smoke' in lower_class and confidence <= 0.80:
                        continue
                    if 'fire' in lower_class and 'fire-smoke' not in lower_class and confidence <= 0.75:
                        continue
                    if 'smoke' in lower_class and confidence <= 0.60:
                        continue
                    if 'spark' in lower_class and confidence <= 0.85:
                        continue
                        
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # ✅ FALSE POSITIVE REDUCTION: Ignore very tiny bounding boxes (e.g., reflections/noise)
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if box_width < 30 or box_height < 30:
                        # Too small to be a verified threat, likely noise
                        continue
                    
                    detections.append({
                        "category": class_name,
                        "confidence": confidence,
                        "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                        "center": {"x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2)},
                    })
            
            return detections

    def detect_fall_pose(self, frame: np.ndarray) -> List[Dict]:
        """
        ✅ NEW: Detect if a person is lying on the ground using pose keypoints
        """
        if getattr(self, 'pose_model', None) is None:
            return []
            
        with self.lock:
            results = self.pose_model.predict(frame, conf=0.5, verbose=False)
            detections = []
            
            for result in results:
                if getattr(result, 'keypoints', None) is None or getattr(result.keypoints, 'xy', None) is None or getattr(result, 'boxes', None) is None:
                    continue
                    
                if result.keypoints.xy.numel() == 0:
                    continue
                    
                for idx, kpts in enumerate(result.keypoints.xy):
                    box = result.boxes[idx]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    kpts = kpts.cpu().numpy()
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    is_lying = False
                    
                    if len(kpts) >= 13:
                        l_shoulder, r_shoulder = kpts[5], kpts[6]
                        l_hip, r_hip = kpts[11], kpts[12]
                        
                        if (l_shoulder[0] != 0 or l_shoulder[1] != 0) and (l_hip[0] != 0 or l_hip[1] != 0):
                            shoulder_cy = (l_shoulder[1] + r_shoulder[1]) / 2 if r_shoulder[1] != 0 else l_shoulder[1]
                            hip_cy = (l_hip[1] + r_hip[1]) / 2 if r_hip[1] != 0 else l_hip[1]
                            
                            shoulder_cx = (l_shoulder[0] + r_shoulder[0]) / 2 if r_shoulder[0] != 0 else l_shoulder[0]
                            hip_cx = (l_hip[0] + r_hip[0]) / 2 if r_hip[0] != 0 else l_hip[0]
                            
                            dx = abs(shoulder_cx - hip_cx)
                            dy = abs(shoulder_cy - hip_cy)
                            
                            if dx > dy and bbox_width > bbox_height:
                                is_lying = True
                    
                    if not is_lying and bbox_width > bbox_height * 1.5:
                        is_lying = True

                    if is_lying:
                        detections.append({
                            "category": "lying_person",
                            "confidence": conf,
                            "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                            "center": {"x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2)},
                        })
            return detections

    def detect_plates(self, frame: np.ndarray) -> List[Dict]:
        """
        ✅ NEW: Run License Plate Detection (ALPR)
        """
        with self.lock:
            detections = []
            boxes_to_process = []
            
            # Priority 1: Use specialized plate_model if available
            if self.plate_model:
                results = self.plate_model.predict(frame, conf=0.20, verbose=False)
                for result in results:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        boxes_to_process.append((int(x1), int(y1), int(x2), int(y2), conf))
            else:
                # Priority 2: Fallback to OpenCV HaarCascade for license plates
                if getattr(self, 'plate_cascade', None) is None:
                    cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
                    self.plate_cascade = cv2.CascadeClassifier(cascade_path)
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                plates = self.plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 20))
                for (x, y, w, h) in plates:
                    boxes_to_process.append((x, y, x + w, y + h, 0.60)) # Default HAAR conf
            
            # Process OCR for all detected plates
            for (x1, y1, x2, y2, confidence) in boxes_to_process:
                # Ensure coordinates are within frame bounds (avoid negative indices/crashes)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                plate_crop = frame[y1:y2, x1:x2]
                plate_text = "UNKNOWN"
                
                if plate_crop.size > 0:
                    # ✅ NEW: Pre-processing for better OCR (Upscale if too small)
                    ph, pw = plate_crop.shape[:2]
                    if ph < 64:
                        scale = 64 / ph
                        plate_crop = cv2.resize(plate_crop, (int(pw * scale), 64), interpolation=cv2.INTER_CUBIC)

                    # Priority 1: EasyOCR (dedicated library)
                    if getattr(self, 'ocr_reader', None):
                        try:
                            # detail=0 returns only text
                            results = self.ocr_reader.readtext(plate_crop, detail=0)
                            if results:
                                # Join and clean up (remove spaces, special chars)
                                raw_text = "".join(results).upper()
                                plate_text = "".join(c for c in raw_text if c.isalnum())
                        except Exception as e:
                            print(f"⚠️ EasyOCR Error: {e}")
                    
                    # Priority 2: YOLO OCR (fallback)
                    elif getattr(self, 'ocr_model', None):
                        try:
                            ocr_results = self.ocr_model.predict(plate_crop, conf=0.25, verbose=False)
                            chars = []
                            for or_res in ocr_results:
                                for b in or_res.boxes:
                                    chars.append({
                                        'x': b.xyxy[0][0].item(),
                                        'text': self.ocr_model.names[int(b.cls[0])]
                                    })
                            if chars:
                                chars.sort(key=lambda x: x['x'])
                                plate_text = "".join([c['text'] for c in chars]).upper()
                        except Exception as e:
                            print(f"⚠️ YOLO OCR Error: {e}")

                detections.append({
                    "category": "license_plate",
                    "confidence": confidence,
                    "plate_number": plate_text,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })
            
            return detections
    
    def detect_ppe_batch(self, frames: list, is_night_mode: bool = False) -> List[List[Dict]]:
        """
        ✅ OPTIMIZED: Batch PPE detection with adaptive thresholds
        """
        if not frames:
            return []
        
        resized_frames = [cv2.resize(f, (640, 640)) for f in frames]
        
        results = None
        person_results = None
        
        with self.lock:
            if not hasattr(self, 'batch_failed_models'):
                self.batch_failed_models = {}

            def _safe_predict(mdl, frs, c):
                mdl_id = id(mdl)
                # If we already know this model fails with batching, skip straight to single frames
                if self.batch_failed_models.get(mdl_id, False):
                    res = []
                    for f in frs:
                        res.extend(mdl.predict([f], conf=c, verbose=False))
                    return res
                    
                try:
                    return mdl.predict(frs, conf=c, verbose=False)
                except Exception as ex:
                    # Mark this model as unsupported for batching
                    self.batch_failed_models[mdl_id] = True
                    print(f"⚠️ Batch predict failed ({ex}), falling back to single frames (will remember for this model)...")
                    res = []
                    for f in frs:
                        res.extend(mdl.predict([f], conf=c, verbose=False))
                    return res

            # 1. Main PPE model: Skip in Night Mode if specialized person model exists
            if not is_night_mode or not self.person_model:
                results = _safe_predict(self.model, resized_frames, 0.20)
            
            # 2. Specialized person detection: Only run in Night Mode as per requirement
            if is_night_mode and self.person_model:
                person_results = _safe_predict(self.person_model, resized_frames, 0.25)

            # 3. ✅ NEW: Pose detection for visibility-based exemptions
            pose_results = None
            if self.pose_model:
                pose_results = _safe_predict(self.pose_model, resized_frames, 0.30)
                if pose_results:
                    pose_count = sum(len(r.boxes) for r in pose_results)
                    if pose_count > 0:
                        print(f"🧍 Pose Detection: Found {pose_count} person(s) across batch")

        
        batch_detections = []
        num_frames = len(frames)
        
        for idx in range(num_frames):
            detections = []
            orig_h, orig_w = frames[idx].shape[:2]
            sx, sy = orig_w / 640, orig_h / 640
            
            # Process main model results
            if results and idx < len(results):
                for box in results[idx].boxes:
                    class_name = self.model.names[int(box.cls[0])].lower()
                    category = self._map_to_ppe_category(class_name)
                    
                    if not category:
                        continue
                    
                    # ✅ EXCLUSIVITY:
                    # In Night Mode: Only detect people
                    # In Day Mode: Skip human detection (use Face-based counting only)
                    if is_night_mode:
                        if category != 'person': continue
                    else:
                        if category == 'person': continue
                
                    confidence = float(box.conf[0])
                    min_threshold = self._get_ppe_confidence_threshold(category)
                    
                    # ✅ Lowered person threshold specifically for better counting
                    if category == 'person' and min_threshold > 0.20:
                        min_threshold = 0.20
                        
                    if confidence < min_threshold:
                        continue
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    
                    detections.append({
                        "category": category,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "center": {"x": int((x1+x2)/2), "y": int((y1+y2)/2)},
                        "area": int((x2 - x1) * (y2 - y1)),
                        "priority": self.PPE_PRIORITY.get(category, 5)
                    })
                
            # 2. Process specialized person model results if available
            if person_results:
                for box in person_results[idx].boxes:
                    class_name = self.person_model.names[int(box.cls[0])].lower()
                    # Use broad match for person-like classes
                    person_keywords = ['person', 'human', 'body', 'people', 'man', 'woman']
                    if not any(k in class_name for k in person_keywords):
                        continue
                        
                    confidence = float(box.conf[0])
                    if confidence < 0.20: # Lowered threshold for sensitivity
                        continue
                        
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    
                    detections.append({
                        "category": "person",
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "center": {"x": int((x1+x2)/2), "y": int((y1+y2)/2)},
                        "area": int((x2 - x1) * (y2 - y1)),
                        "priority": self.PPE_PRIORITY.get("person", 1)
                    })

            # 3. ✅ NEW: Process pose detections
            if pose_results and idx < len(pose_results):
                res = pose_results[idx]
                if hasattr(res, 'keypoints') and res.keypoints is not None:
                    # Access keypoints.xy which is [num_persons, 17, 2]
                    kpts_data = res.keypoints.xy.cpu().numpy()
                    kpts_conf = res.keypoints.conf.cpu().numpy() if hasattr(res.keypoints, 'conf') else None
                    boxes = res.boxes.xyxy.cpu().numpy()
                    
                    for p_idx, kpts in enumerate(kpts_data):
                        px1, py1, px2, py2 = boxes[p_idx]
                        px1, py1, px2, py2 = int(px1*sx), int(py1*sy), int(px2*sx), int(py2*sy)
                        
                        bbox_width = px2 - px1
                        bbox_height = py2 - py1
                        is_lying = False
                        
                        # Rescale keypoints
                        scaled_kpts = []
                        for ki, (kx, ky) in enumerate(kpts):
                            k_conf = float(kpts_conf[p_idx][ki]) if kpts_conf is not None else (1.0 if (kx > 0 or ky > 0) else 0.0)
                            scaled_kpts.append({
                                "x": int(kx * sx),
                                "y": int(ky * sy),
                                "conf": k_conf
                            })
                        
                        # ✅ Check for fall (lying person) from keypoints
                        if len(kpts) >= 17:
                            # User requirement: Whole body must be completely in frame
                            # Checking head (0-4) and ankle (15-16) keypoints visibility
                            has_head = False
                            has_ankle = False
                            if kpts_conf is not None:
                                head_confs = kpts_conf[p_idx][0:5]
                                ankle_confs = kpts_conf[p_idx][15:17]
                                if any(c > 0.5 for c in head_confs): has_head = True
                                if any(c > 0.5 for c in ankle_confs): has_ankle = True
                            else:
                                head_pts = kpts[0:5]
                                ankle_pts = kpts[15:17]
                                if any(pt[0] > 0 or pt[1] > 0 for pt in head_pts): has_head = True
                                if any(pt[0] > 0 or pt[1] > 0 for pt in ankle_pts): has_ankle = True

                            # Only consider lying person if whole body is visible
                            if has_head and has_ankle:
                                l_shoulder, r_shoulder = kpts[5], kpts[6]
                                l_hip, r_hip = kpts[11], kpts[12]
                                
                                if (l_shoulder[0] != 0 or l_shoulder[1] != 0) and (l_hip[0] != 0 or l_hip[1] != 0):
                                    shoulder_cy = (l_shoulder[1] + r_shoulder[1]) / 2 if r_shoulder[1] != 0 else l_shoulder[1]
                                    hip_cy = (l_hip[1] + r_hip[1]) / 2 if r_hip[1] != 0 else l_hip[1]
                                    shoulder_cx = (l_shoulder[0] + r_shoulder[0]) / 2 if r_shoulder[0] != 0 else l_shoulder[0]
                                    hip_cx = (l_hip[0] + r_hip[0]) / 2 if r_hip[0] != 0 else l_hip[0]
                                    dx = abs(shoulder_cx - hip_cx)
                                    dy = abs(shoulder_cy - hip_cy)
                                    if dx > dy and bbox_width > bbox_height:
                                        is_lying = True
                                
                                if not is_lying and bbox_width > bbox_height * 1.5:
                                    is_lying = True
                        
                        if is_lying:
                            print(f"🆘 FALL/LYING DETECTED: Ratio {bbox_width/bbox_height:.2f} | Conf: {float(res.boxes.conf[p_idx]):.2f}")

                        detections.append({
                            "category": "skeleton",
                            "keypoints": scaled_kpts,
                            "bbox": {"x1": px1, "y1": py1, "x2": px2, "y2": py2},
                            "confidence": float(res.boxes.conf[p_idx]),
                            "verified": True
                        })
                        
                        if is_lying:
                            detections.append({
                                "category": "lying_person",
                                "confidence": float(res.boxes.conf[p_idx]),
                                "bbox": {"x1": px1, "y1": py1, "x2": px2, "y2": py2},
                                "center": {"x": int((px1+px2)/2), "y": int((py1+py2)/2)},
                                "verified": True
                            })

            # 4. ✅ DE-DUPLICATE: Remove overlapping person detections
            # If two "person" bboxes overlap significantly, keep the one with higher confidence
            person_indices = [i for i, d in enumerate(detections) if d["category"] in ["person", "skeleton"]]
            to_remove = set()
            
            if len(person_indices) > 1:
                for i in range(len(person_indices)):
                    for j in range(i + 1, len(person_indices)):
                        idx1, idx2 = person_indices[i], person_indices[j]
                        b1, b2 = detections[idx1]["bbox"], detections[idx2]["bbox"]
                        
                        # Simple IoU
                        ix1, iy1 = max(b1["x1"], b2["x1"]), max(b1["y1"], b2["y1"])
                        ix2, iy2 = min(b1["x2"], b2["x2"]), min(b1["y2"], b2["y2"])
                        
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2 - ix1) * (iy2 - iy1)
                            union = (b1["x2"]-b1["x1"])*(b1["y2"]-b1["y1"]) + (b2["x2"]-b2["x1"])*(b2["y2"]-b2["y1"]) - inter
                            iou = inter / union if union > 0 else 0
                            
                            if iou > 0.45: # High overlap
                                c1, c2 = detections[idx1]["category"], detections[idx2]["category"]
                                if c1 == "skeleton" and c2 == "person":
                                    to_remove.add(idx2)
                                elif c2 == "skeleton" and c1 == "person":
                                    to_remove.add(idx1)
                                elif detections[idx1].get("confidence", 0) >= detections[idx2].get("confidence", 0):
                                    to_remove.add(idx2)
                                else:
                                    to_remove.add(idx1)
            
            final_detections = [d for k, d in enumerate(detections) if k not in to_remove]
            
            # DEBUG
            if len(final_detections) > 0:
                print(f"✅ DEBUG: Final detections for frame {idx}: {len(final_detections)} items")
            else:
                if len(detections) > 0:
                    print(f"⚠️ DEBUG: Detections existed ({len(detections)}) but ALL were removed or lost!")

            # Assign color for items in final detections
            for det in final_detections:
                if det["category"] in ["safety_helmet", "reflective_vest", "suit", "helmet", "vest"]:
                    det_color = self._detect_dominant_color(frames[idx], det["bbox"])
                    det["color"] = det_color.lower() if det_color else "unknown"
            
            batch_detections.append(final_detections)
        
        # FINAL DEBUG
        print(f"🚀 DEBUG: detect_ppe_batch returning {len(batch_detections)} frames of results")
        return batch_detections

    def verify_ppe_on_person(self, face_bbox: Tuple[int, int, int, int], 
                            ppe_detections: List[Dict], frame_height: int,
                            person_id: str = None) -> List[Dict]:
        """
        ✅ OPTIMIZED: Enhanced spatial verification with temporal tracking
        """
        if not ppe_detections:
            return []
        
        x1, y1, x2, y2 = face_bbox
        face_w, face_h = x2 - x1, y2 - y1
        face_cx, face_cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        verified = []
        
        # ✅ NEW: Find the skeleton corresponding to this face
        skeleton = None
        for ppe in ppe_detections:
            if ppe["category"] == "skeleton":
                s_bbox = ppe["bbox"]
                # Check if face center is inside skeleton bbox
                if s_bbox["x1"] <= face_cx <= s_bbox["x2"] and s_bbox["y1"] <= face_cy <= s_bbox["y2"]:
                    skeleton = ppe
                    # Include skeleton in verified list for compliance check
                    verified.append(ppe)
                    break

        for ppe in ppe_detections:
            if ppe["category"] == "skeleton":
                continue # Already handled or skip processing as PPE

            px, py = ppe["center"]["x"], ppe["center"]["y"]
            dx, dy = abs(px - face_cx), py - face_cy
            cat = ppe["category"]
            
            # Get spatial verification parameters
            params = self.spatial_verification_params.get(cat, {
                "horizontal_tolerance": 5.0,
                "vertical_range": (-2.0, 10.0),
                "min_confidence_boost": 0.0
            })
            
            h_tol = params["horizontal_tolerance"]
            v_min, v_max = params["vertical_range"]
            conf_boost = params["min_confidence_boost"]
            
            # ✅ ENHANCED: Distance-based verification
            horizontal_valid = dx < face_w * h_tol
            vertical_valid = face_h * v_min < dy < face_h * v_max
            
            valid = horizontal_valid and vertical_valid
            
            # ✅ NEW: Confidence boost for very close proximity
            if valid:
                # Calculate proximity score (0-1, higher = closer)
                h_proximity = 1.0 - (dx / (face_w * h_tol))
                v_center = (v_min + v_max) / 2
                v_proximity = 1.0 - abs((dy / face_h) - v_center) / ((v_max - v_min) / 2)
                proximity_score = (h_proximity + v_proximity) / 2
                
                # Boost confidence for close proximity
                boosted_confidence = min(ppe["confidence"] + (proximity_score * conf_boost), 1.0)
                ppe["confidence"] = boosted_confidence
                ppe["proximity_score"] = proximity_score
            
            ppe["verified"] = valid
            ppe["spatial_distance"] = {
                "horizontal": dx,
                "vertical": dy,
                "horizontal_ratio": dx / face_w if face_w > 0 else 0,
                "vertical_ratio": dy / face_h if face_h > 0 else 0
            }
            
            verified.append(ppe)
        
        # ✅ NEW: Apply temporal tracking if person_id provided
        if person_id:
            verified = self._apply_temporal_ppe_tracking(person_id, verified)
        
        return verified
    
    def _apply_temporal_ppe_tracking(self, person_id: str, current_detections: List[Dict]) -> List[Dict]:
        """
        ✅ NEW: Apply temporal smoothing to PPE detections
        Reduces false negatives from momentary occlusions
        """
        with self.ppe_tracking_lock:
            # Update tracking history
            current_time = time.time()
            
            for detection in current_detections:
                if detection.get("verified", False):
                    category = detection["category"]
                    self.ppe_temporal_tracking[person_id][category].append({
                        "confidence": detection["confidence"],
                        "timestamp": current_time,
                        "verified": True
                    })
            
            # Check temporal history for each required PPE
            enhanced_detections = current_detections.copy()
            
            # For each PPE category in tracking history
            for category, history in self.ppe_temporal_tracking[person_id].items():
                if not history:
                    continue
                
                # Clean old entries (>2 seconds)
                recent_history = [h for h in history if current_time - h["timestamp"] < 2.0]
                self.ppe_temporal_tracking[person_id][category] = deque(recent_history, maxlen=10)
                
                # Check if this category is currently detected
                currently_detected = any(
                    d["category"] == category and d.get("verified", False)
                    for d in current_detections
                )
                
                # If not currently detected but was recently detected consistently
                if not currently_detected and len(recent_history) >= 5:
                    # Calculate average confidence from recent history
                    avg_confidence = np.mean([h["confidence"] for h in recent_history])
                    
                    # If confidence was consistently high, assume temporary occlusion
                    if avg_confidence > 0.5:
                        # Add a "ghost" detection with reduced confidence
                        ghost_detection = {
                            "category": category,
                            "class_name": category,
                            "confidence": avg_confidence * 0.7,  # Reduced confidence
                            "verified": True,
                            "temporal_inference": True,  # Mark as inferred
                            "bbox": {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
                            "center": {"x": 0, "y": 0},
                            "area": 0,
                            "priority": self.PPE_PRIORITY.get(category, 5)
                        }
                        enhanced_detections.append(ghost_detection)
            
            return enhanced_detections

    def check_person_compliance(self, person_role: str, detected_ppe: List[Dict], 
                               face_detected: bool = True, person_id: str = None,
                               frame_size: Tuple[int, int] = None, face_bbox: Tuple[int, int, int, int] = None) -> Dict:
        """
        ✅ OPTIMIZED: Enhanced compliance checking with detailed metrics
        Supports visibility-based exemptions (e.g., skip boots if feet out of frame)
        """
        role = person_role.lower()
        required_ppe = self.role_ppe_requirements.get(role, self.role_ppe_requirements.get("default", []))
        
        effective_ppe = {}  # Changed to dict to store confidence
        
        for d in detected_ppe:
            if not d.get("verified", True):
                continue
            
            detected_category = d["category"]
            
            if detected_category not in required_ppe:
                continue
            
            # Get minimum confidence threshold for this category
            min_conf = self._get_ppe_confidence_threshold(detected_category)
            
            if d["confidence"] >= min_conf:
                # Store highest confidence detection for each category
                if detected_category not in effective_ppe or d["confidence"] > effective_ppe[detected_category]["confidence"]:
                    effective_ppe[detected_category] = {
                        "confidence": d["confidence"],
                        "temporal_inference": d.get("temporal_inference", False),
                        "proximity_score": d.get("proximity_score", 0.0)
                    }
        
        missing = []
        wearing = []
        wearing_details = {}
        
        # ✅ NEW: Check for visibility-based exemptions using skeleton keypoints
        excluded_from_missing = []
        
        # Extract skeleton if available
        skeleton = next((d for d in detected_ppe if d["category"] == "skeleton"), None)
        
        if skeleton:
            kpts = skeleton.get("keypoints", [])
            # Threshold for keypoint presence
            KP_THRESH = 0.25
            
            # 1. Boots exemption (Ankles: 15, 16)
            if len(kpts) > 16:
                left_ankle, right_ankle = kpts[15], kpts[16]
                if left_ankle["conf"] < KP_THRESH and right_ankle["conf"] < KP_THRESH:
                    excluded_from_missing.append("boots")
            
            # 2. Gloves exemption (Wrists: 9, 10)
            if len(kpts) > 10:
                left_wrist, right_wrist = kpts[9], kpts[10]
                if left_wrist["conf"] < KP_THRESH and right_wrist["conf"] < KP_THRESH:
                    excluded_from_missing.append("gloves")
            
            # 3. Vest exemption (Shoulders: 5, 6; Hips: 11, 12)
            if len(kpts) > 12:
                left_shoulder, right_shoulder = kpts[5], kpts[6]
                left_hip, right_hip = kpts[11], kpts[12]
                # If upper torso is mostly missing, skip vest
                torso_visible_parts = sum(1 for kp in [left_shoulder, right_shoulder, left_hip, right_hip] if kp["conf"] > KP_THRESH)
                if torso_visible_parts < 2:
                    excluded_from_missing.append("reflective_vest")
                    excluded_from_missing.append("suit")
                    excluded_from_missing.append("apron")
            
            # 4. Helmet exemption (Head: 0-4)
            if len(kpts) > 4:
                head_visible_parts = sum(1 for i in range(5) if kpts[i]["conf"] > KP_THRESH)
                if head_visible_parts == 0:
                    excluded_from_missing.append("safety_helmet")
                    excluded_from_missing.append("helmet")
                    excluded_from_missing.append("welding_mask")

        # Fallback to bbox-based visibility check if skeleton is missing
        if not skeleton and frame_size and face_bbox:
            frame_h, frame_w = frame_size
            x1, y1, x2, y2 = face_bbox
            face_h = y2 - y1
            face_cy = (y1 + y2) / 2
            
            # Boots visibility check
            boots_expected_y = face_cy + (face_h * 6.5)
            if boots_expected_y > frame_h:
                excluded_from_missing.append("boots")

        for ppe_item in required_ppe:
            if ppe_item in effective_ppe:
                wearing.append(ppe_item)
                wearing_details[ppe_item] = effective_ppe[ppe_item]
            elif ppe_item in excluded_from_missing:
                # print(f"DEBUG: {ppe_item} excluded from missing due to visibility")
                pass
            else:
                missing.append(ppe_item)
        
        is_compliant = len(missing) == 0 if face_detected else False
        compliance_percentage = (len(wearing) / len(required_ppe) * 100 if required_ppe else 100.0) if face_detected else 0.0
        
        # ✅ NEW: Calculate compliance quality score
        if wearing:
            avg_confidence = np.mean([details["confidence"] for details in wearing_details.values()])
            avg_proximity = np.mean([details.get("proximity_score", 0.5) for details in wearing_details.values()])
            temporal_count = sum(1 for details in wearing_details.values() if details.get("temporal_inference", False))
            
            # Quality score considers confidence, proximity, and temporal inference
            quality_score = (avg_confidence * 0.5 + avg_proximity * 0.3 + 
                           (1.0 - temporal_count / max(len(wearing), 1)) * 0.2)
        else:
            quality_score = 0.0
        
        # ✅ NEW: Prioritize missing PPE by criticality
        missing_with_priority = sorted(
            [(item, self.PPE_PRIORITY.get(item, 5)) for item in missing],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "is_compliant": is_compliant,
            "required_ppe": required_ppe,
            "wearing_ppe": wearing,
            "wearing_details": wearing_details,  # ✅ NEW: Detailed info
            "missing_ppe": missing,
            "missing_priority": [item for item, _ in missing_with_priority],  # ✅ NEW: Prioritized list
            "compliance_percentage": compliance_percentage,
            "compliance_quality_score": quality_score,  # ✅ NEW: Quality metric
            "temporal_inference_used": any(d.get("temporal_inference", False) for d in detected_ppe if d.get("verified", False))
        }

    def draw_ppe_detections(self, frame: np.ndarray, ppe_detections: List[Dict], 
                           show_unverified: bool = False) -> np.ndarray:
        """
        ✅ ENHANCED: Draw PPE detections with priority-based coloring
        """
        annotated = frame.copy()
        
        # Priority-based colors (red = critical, yellow = high, green = medium/low)
        priority_colors = {
            10: (0, 0, 255),    # Critical - Red
            9: (0, 100, 255),   # High - Orange
            8: (0, 165, 255),   # Medium-High - Light Orange
            7: (0, 255, 255),   # Medium - Yellow
            6: (0, 255, 128),   # Medium-Low - Yellow-Green
            5: (0, 255, 0),     # Low - Green
        }
        
        for det in ppe_detections:
            if not show_unverified and not det.get('verified', True):
                if det.get("category") != "skeleton": # Skeletons aren't always 'verified' as PPE
                    continue
            
            if det.get("category") == "skeleton":
                kpts = det.get("keypoints", [])
                if not kpts: continue
                # Draw skeleton lines (standard COCO connections)
                skeleton_lines = [
                    (5, 6), (5, 11), (6, 12), (11, 12), # Torso
                    (5, 7), (7, 9), (6, 8), (8, 10),    # Arms
                    (11, 13), (13, 15), (12, 14), (14, 16) # Legs
                ]
                for p1, p2 in skeleton_lines:
                    if p1 < len(kpts) and p2 < len(kpts):
                        kp1, kp2 = kpts[p1], kpts[p2]
                        if kp1["conf"] > 0.4 and kp2["conf"] > 0.4:
                            cv2.line(annotated, (kp1["x"], kp1["y"]), (kp2["x"], kp2["y"]), (0, 255, 0), 2)
                # Draw keypoints
                for kp in kpts:
                    if kp["conf"] > 0.4:
                        cv2.circle(annotated, (kp["x"], kp["y"]), 4, (255, 255, 0), -1)
                continue # Keypoints drawn, move to next detection
            
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # Get color based on priority
            priority = det.get('priority', 5)
            color = priority_colors.get(priority, (255, 255, 255))
            
            # ✅ Different line style for temporal inference
            thickness = 2
            if det.get('temporal_inference', False):
                # Dashed line for temporal inference
                self._draw_dashed_rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # ✅ Enhanced label with confidence, color, and proximity
            label = f"{det['category']}"
            if det.get("color") and det["color"] != "unknown":
                label += f" ({det['color']})"
            label += f": {det['confidence']:.2f}"
            if det.get('proximity_score'):
                label += f" (P:{det['proximity_score']:.1f})"
            
            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1-text_height-8), (x1+text_width+4, y1), color, -1)
            cv2.putText(annotated, label, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness):
        """Draw dashed rectangle for temporal inference"""
        x1, y1 = pt1
        x2, y2 = pt2
        dash_length = 10
        
        # Top
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        # Bottom
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        # Left
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        # Right
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def draw_face_label(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int],
                       person_name: str, compliance_info: Dict, is_unknown: bool) -> np.ndarray:
        """
        ✅ ENHANCED: Draw face label with detailed compliance info
        """
        x1, y1, x2, y2 = face_bbox
        
        missing_ppe = compliance_info.get("missing_priority", compliance_info.get("missing_ppe", []))
        quality_score = compliance_info.get("compliance_quality_score", 0.0)
        is_zone_violation = not compliance_info.get("is_zone_authorized", True)
        
        # Status text and color logic (Zone Priority > PPE)
        if is_zone_violation:
            status_text = "WRONG ZONE"
            status_color = (0, 0, 255)  # Red for Zone Violation
        elif compliance_info.get("is_night_mode_violation"):
            status_text = "NIGHT BREACH"
            status_color = (255, 0, 255) # Magenta
        elif missing_ppe:
            status_text = f"Missing: {', '.join(missing_ppe[:2])}"  # Show top 2
            if len(missing_ppe) > 2:
                status_text += f" +{len(missing_ppe)-2}"
            status_color = (0, 165, 255)  # Orange for PPE Violation
        else:
            status_text = f"Compliant (Q:{quality_score:.1f})"
            status_color = (0, 255, 0)  # Green
        
        # ✅ ENHANCED: Caption with Name and ID for visitors
        if is_unknown:
            # Check if name contains ID like "Intruder (visitor_01)"
            if "(" in person_name and ")" in person_name:
                name_text = person_name # Already has ID
            else:
                name_text = f"Visitor ({person_name})"
        else:
            name_text = person_name
        
        # Draw with background
        cv2.rectangle(frame, (x1, y1-50), (x2, y1), (0, 0, 0), -1)
        cv2.putText(frame, name_text, (x1+5, y1-28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # ✅ NEW: Temporal inference indicator
        if compliance_info.get("temporal_inference_used", False):
            cv2.putText(frame, "T", (x2-20, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return frame


class OptimizedIntegratedSystem:
    """
    ✅ OPTIMIZED: Enhanced integrated system with accurate status reporting
    """
    
    def __init__(self, face_system, ppe_system, db, alert_engine):
        self.face_system = face_system
        self.ppe_system = ppe_system
        self.db = db
        self.alert_engine = alert_engine
        self.unknown_trackers = {}
        self.track_id_counter = defaultdict(int)
        self.track_last_seen = {}
        self.violation_cooldown = {}
        self.VIOLATION_COOLDOWN_SEC = 30
        self.TRACKER_CLEANUP_SEC = 60
        self.last_cleanup = time.time()
        self.identity_cache = {}
        
        # ✅ NEW: Status tracking for frontend
        self.status_history = defaultdict(lambda: deque(maxlen=30))  # 30-frame history
        self.status_lock = Lock()
        
        # ✅ NEW: Attendance tracking (once per day)
        self.attendance_cache = {}  # {person_id: 'YYYY-MM-DD'}
        
        # ✅ NEW: Active violation tracking (to prevent repeated logs)
        self.active_violations = {}  # {(camera_id, person_id): timestamp}
        self.active_violation_logs = {}  # {(camera_id, person_id): log_id}
        
        
        # ✅ SYNC: Ensure we have the latest rules from ppe_system
        self.role_ppe_requirements = self.ppe_system.role_ppe_requirements
        
        # ✅ EMERGENCY PRIORITY: Increased fire detection frequency (Check every 0.1s)
        self.FIRE_CHECK_INTERVAL = 0.1
        self.fire_last_check = {}
        self.fire_cache = {}

        # ✅ NEW: OCR/Plate detection optimization
        self.ocr_last_check = {}  # {camera_id: timestamp}
        self.ocr_cache = {}       # {camera_id: list_of_plate_results}
        self.OCR_CHECK_INTERVAL = 1.0 # Check plates every 1s (increased frequency)
        self.plate_log_cooldown = {} # {(camera_id, plate_num): last_log_time}
        self.plate_entry_exit = {} # {plate_num: "Entry"/"Exit"} — toggles on each DB log
        # ✅ NEW: Load OCR Camera list from DB
        try:
            cfg = self.db.system_config.find_one({"config_type": "ppe_rules"}) or {}
            val = cfg.get("ocr_camera_id")
            if isinstance(val, list):
                self.ocr_camera_ids = val
            elif isinstance(val, str):
                self.ocr_camera_ids = [val]
            else:
                self.ocr_camera_ids = []
                
            if self.ocr_camera_ids:
                print(f"📷 OCR Cameras (Loaded from DB): {self.ocr_camera_ids}")
        except Exception as e:
            print(f"⚠️ Failed to load OCR camera from DB: {e}")
            self.ocr_camera_ids = []
        
        print(" Optimized Integrated Face + PPE System initialized")
        print("    Enhanced status tracking enabled")
        print("    Automatic attendance marking enabled")
        print("    🔥 Fire detection HIGH PRIORITY enabled (Interval: 1.0s)")

    @property
    def ocr_camera_id(self):
        """Standardized: Getter for single active OCR camera ID."""
        return self.ocr_camera_ids[0] if getattr(self, "ocr_camera_ids", []) else None

    @ocr_camera_id.setter
    def ocr_camera_id(self, val):
        """Standardized: Setter for active OCR camera. Keeps ocr_camera_ids in sync."""
        if not val or val == "DISABLED":
            print("📷 OCR processing: DISABLED (all cameras)")
            self.ocr_camera_ids = []
            self.ocr_last_check = {}
        else:
            cam_target = str(val).strip()
            self.ocr_camera_ids = [cam_target]
            print(f"📷 Designated OCR Camera: {cam_target} (Ready for plate scanning)")
            
            # Persistent: Save to DB
            try:
                self.db.system_config.update_one(
                    {"config_type": "ppe_rules"},
                    {"$set": {"ocr_camera_id": self.ocr_camera_ids}},
                    upsert=True
                )
            except:
                pass

    def _cleanup_stale_trackers(self):
        now = time.time()
        if now - self.last_cleanup < self.TRACKER_CLEANUP_SEC:
            return
        
        stale_tracks = [track_id for track_id, last_seen in self.track_last_seen.items() 
                       if now - last_seen > self.TRACKER_CLEANUP_SEC]
        
        for track_id in stale_tracks:
            self.track_last_seen.pop(track_id, None)
            for camera_id in self.unknown_trackers:
                if track_id in self.unknown_trackers[camera_id]:
                    del self.unknown_trackers[camera_id][track_id]
        
        self.last_cleanup = now

    def _calculate_iou(self, box1, box2):
        """Calculate IOU between two boxes (x1,y1,x2,y2)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection + 1e-6)

    def _is_contained(self, inner_box, outer_box):
        """Check if inner_box is mostly inside outer_box"""
        ix1 = max(inner_box[0], outer_box[0])
        iy1 = max(inner_box[1], outer_box[1])
        ix2 = min(inner_box[2], outer_box[2])
        iy2 = min(inner_box[3], outer_box[3])
        
        if ix2 <= ix1 or iy2 <= iy1: return False
        
        inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
        intersection = (ix2 - ix1) * (iy2 - iy1)
        
        return (intersection / inner_area) > 0.8

    def _assign_track_id(self, camera_id: str, bbox: Tuple, is_body: bool = False, exclude_ids: set = None) -> str:
        """
        ✅ OPTIMIZED: Robust Persistence Tracking
        Matches current detections to previous tracks.
        Always prefers body-based tracking for maximum persistence.
        """
        self._cleanup_stale_trackers()
        if camera_id not in self.unknown_trackers:
            self.unknown_trackers[camera_id] = {} 

        if exclude_ids is None:
            exclude_ids = set()

        candidates = []
        current_tracks = self.unknown_trackers[camera_id]
        
        # Current detection center
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        for track_id, data in current_tracks.items():
            if track_id in exclude_ids:
                continue
                
            # 1. IoU Score
            iou = self._calculate_iou(bbox, data['bbox'])
            
            # 2. Centroid Distance Score (Normalized by image size approximation 640x640)
            t_bbox = data['bbox']
            tx = (t_bbox[0] + t_bbox[2]) / 2
            ty = (t_bbox[1] + t_bbox[3]) / 2
            
            dist_sq = (cx - tx)**2 + (cy - ty)**2
            # Use sqrt for linear falloff. 200px max distance.
            dist = math.sqrt(dist_sq)
            dist_score = max(0, 1.0 - (dist / 200.0)) 
            
            # 3. Containment Score (for face vs body matching)
            containment = 1.0 if self._is_contained(bbox, t_bbox) or self._is_contained(t_bbox, bbox) else 0.0
            
            # Hybrid Score: Prioritize IoU, fallback to Distance/Containment
            # IoU is reducing rapidly with motion, so distance helps keep track
            final_score = iou + (dist_score * 0.6) + (containment * 0.4)
            
            candidates.append((final_score, track_id))
        
        # Sort by best score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Adaptive Threshold
        # If IoU is high, great. If distance is very close, also good.
        MATCH_THRESHOLD = 0.4 
        
        if candidates and candidates[0][0] > MATCH_THRESHOLD:
            best_score, best_track_id = candidates[0]
            
            # Update existing track
            # If current is body and old was face, upgrade to body for better future tracking
            old_data = current_tracks[best_track_id]
            
            # Only update position if significantly moved or upgrading to body
            # This reduces jitter
            current_tracks[best_track_id]['bbox'] = bbox
            current_tracks[best_track_id]['last_seen'] = time.time()
            self.track_last_seen[best_track_id] = time.time()
            return best_track_id
            
        # Create new track
        self.track_id_counter[camera_id] += 1
        new_track_id = f"visitor_{self.track_id_counter[camera_id]:02d}"
        
        current_tracks[new_track_id] = {
            'bbox': bbox,
            'last_seen': time.time()
        }
        self.track_last_seen[new_track_id] = time.time()
        return new_track_id

    def _should_log_violation(
        self,
        camera_id: str,
        person_id: str,
        missing_ppe: list,
        track_id: str = None
    ) -> bool:
        """
        Violation cooldown per (camera + person + PPE set)
        """
        key_id = track_id if track_id and person_id.startswith("visitor_") else person_id

        cooldown_key = (
            camera_id,
            key_id,
            tuple(sorted(missing_ppe))
        )

        now = time.time()
        last = self.violation_cooldown.get(cooldown_key, 0)

        if now - last < self.VIOLATION_COOLDOWN_SEC:
            return False

        self.violation_cooldown[cooldown_key] = now
        return True


    def _mark_attendance_if_needed(self, person_id: str, camera_id: str):
        """
        ✅ NEW: Mark attendance only once per day per person
        """
        if not person_id or person_id == "unknown" or person_id.startswith("visitor_"):
            return

        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        
        # Check cache first (fast)
        if self.attendance_cache.get(person_id) == today_str:
            return

        # Mark in DB
        log_id = f"ATTENDANCE_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # We run this in a separate thread to not block inference?
            # For now, it's a quick DB update, so standard call is fine.
            # But improved safety: check if db call succeeds
            success = self.db.mark_attendance(person_id, camera_id, log_id)
            if success:
                self.attendance_cache[person_id] = today_str
                print(f"✅ Attendance marked for {person_id} on {today_str}")
        except Exception as e:
            print(f"❌ Failed to mark attendance: {e}")

    def process_frames_with_ppe_batch(self, frames: list, camera_ids: list, run_face: bool = True):
        """
        ✅ OPTIMIZED: Enhanced batch processing with accurate status tracking
        """
        assert len(frames) == len(camera_ids)
        timestamp = datetime.utcnow()
        face_results_batch = []
        now_ts = time.time()
        
        # ✅ NEW: Night Mode Check
        is_night_mode = self.db.is_night_mode_active()
        if is_night_mode:
            print(f"🌙 NIGHT MODE ACTIVE - Skipping PPE Detection & Enabling Security Alerts")
        
        # ✅ EMERGENCY PRIORITY: 1. Fire and Smoke Detection (Check this BEFORE anything else)
        fire_smoke_batch = []
        for idx in range(len(frames)):
            frame = frames[idx]
            camera_id = camera_ids[idx]
            
            try:
                now = time.time()
                last_check = self.fire_last_check.get(camera_id, 0)
                
                if now - last_check > self.FIRE_CHECK_INTERVAL:
                    # Run actual inference
                    detections = self.ppe_system.detect_fire_smoke(frame)
                    self.fire_last_check[camera_id] = now
                    self.fire_cache[camera_id] = detections
                    fire_smoke_batch.append(detections)
                    if detections:
                        print(f"🔥 FIRE/SMOKE DETECTED on {camera_id} during initial check!")
                else:
                    # Return cached result
                    fire_smoke_batch.append(self.fire_cache.get(camera_id, []))
            except Exception as e:
                print(f"❌ Fire/Smoke detection error for {camera_id}: {e}")
                fire_smoke_batch.append([])
        
        # ✅ EMERGENCY OVERRIDE: If all cameras have fire, we could skip everything.
        # But let's check on per-camera basis later. For now, we still need to process 
        # other cameras in the batch if they don't have fire.

        
        # Process faces
        for idx, (frame, camera_id) in enumerate(zip(frames, camera_ids)):
            try:
                faces = self.face_system.process_frame(frame, camera_id, store_logs=False)
                for fr in faces:
                    if fr["person_id"] and fr["person_id"] != "unknown":
                        person = self.db.get_person(fr["person_id"])
                        if person:
                            bbox_str = f"{camera_id}:{fr['box'][0]}_{fr['box'][1]}"
                            self.identity_cache[bbox_str] = {
                                "person_id": fr["person_id"],
                                "person_name": fr["person_name"],
                                "role": person.get("role", "default"),
                                "last_seen": now_ts
                            }
                face_results_batch.append(faces)
            except Exception as e:
                print(f"❌ Face processing error for {camera_id}: {e}")
                face_results_batch.append([])
        
        # Detect PPE
        try:
            # ✅ EXCLUSIVITY: Pass night mode flag to skip PPE model calls
            ppe_batch = self.ppe_system.detect_ppe_batch(frames, is_night_mode=is_night_mode)
        except Exception as e:
            print(f"❌ PPE batch detection error: {e}")
            ppe_batch = [[] for _ in frames]
        
        # Fire/Smoke batch was already computed at the top of this method (highest priority).
        
        
        # ✅ OPTIMIZED: Re-use fall detection from ppe_batch (calculated via pose model)
        fall_batch = []
        for idx in range(len(frames)):
            ppe_dets = ppe_batch[idx] if idx < len(ppe_batch) else []
            # Extract lying_person detections already found during pose processing in detect_ppe_batch
            falls = [d for d in ppe_dets if d.get("category") == "lying_person"]
            fall_batch.append(falls)
        
        # ✅ NEW: Plate Detection/OCR (Rate-limited)
        plate_results_batch = []
        for idx in range(len(frames)):
            frame = frames[idx]
            camera_id = camera_ids[idx]
            
            try:
                # ── Only run OCR on designated OCR cameras (Case-Insensitive) ──
                active_ocr_targets = [cid.upper() for cid in self.ocr_camera_ids]
                if not active_ocr_targets or camera_id.upper() not in active_ocr_targets:
                    plate_results_batch.append(self.ocr_cache.get(camera_id, []))
                    continue

                now = time.time()
                last_check = self.ocr_last_check.get(camera_id, 0)
                
                if now - last_check > self.OCR_CHECK_INTERVAL:
                    detections = self.ppe_system.detect_plates(frame)
                    
                    import re
                    final_detections = []
                    found_indian_plate = False
                    
                    for det in detections:
                        plate_num = det.get("plate_number")
                        
                        if plate_num and plate_num != "UNKNOWN":
                            clean_text = re.sub(r'[^A-Z0-9]', '', plate_num.upper())
                            # Relaxed Indian plate format: AA 0(0) AA(A) 0000(0)
                            # 1. State (2L) 2. Dist (1-2D) 3. Series (1-3L) 4. Num (4-5D)
                            strict_pattern = r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4,5}$'
                            is_indian_plate = (len(clean_text) >= 7 and len(clean_text) <= 12 and bool(re.match(strict_pattern, clean_text)))
                            
                            if not is_indian_plate:
                                # Log rejection only if it looks somewhat like a plate (at least 5 chars)
                                if len(clean_text) > 4:
                                    print(f"🔍 Plate Filter: '{clean_text}' rejected (format/length: {len(clean_text)})")

                            if is_indian_plate:
                                found_indian_plate = True
                                det["plate_number"] = clean_text

                                # ── DB-backed 2-minute cooldown (survives restarts) ──
                                # Check in-memory first (fast path), then verify against DB
                                cooldown_key = (camera_id, clean_text)
                                last_log_mem = self.plate_log_cooldown.get(cooldown_key, 0)
                                within_mem_cooldown = (now - last_log_mem) < 60

                                if not within_mem_cooldown:
                                    # Confirm with DB — look for any entry in last 2 minutes
                                    from datetime import timedelta
                                    cutoff = datetime.utcnow() - timedelta(seconds=60)
                                    recent = self.ppe_system.db.recognition_logs.find_one({
                                        'log_type': 'plate_detection',
                                        'plate_number': clean_text,
                                        'timestamp': {'$gte': cutoff}
                                    }, sort=[('timestamp', -1)])

                                    if recent:
                                        # Found a recent DB entry — honour the cooldown
                                        # Sync in-memory cache from DB timestamp
                                        db_ts = recent.get('timestamp', datetime.utcnow())
                                        elapsed = (datetime.utcnow() - db_ts).total_seconds()
                                        self.plate_log_cooldown[cooldown_key] = now - elapsed
                                        # Also restore entry_exit toggle from DB
                                        last_ee = recent.get('entry_exit', 'Entry')
                                        self.plate_entry_exit[clean_text] = 'Exit' if last_ee == 'Entry' else 'Entry'
                                        within_mem_cooldown = True
                                        print(f"⏳ Plate {clean_text} in DB cooldown ({60 - elapsed:.0f}s left) — skipping")

                                if not within_mem_cooldown:
                                    # Determine Entry or Exit — restore from DB if not in memory
                                    if clean_text not in self.plate_entry_exit:
                                        last_db = self.ppe_system.db.recognition_logs.find_one(
                                            {'log_type': 'plate_detection', 'plate_number': clean_text},
                                            sort=[('timestamp', -1)]
                                        )
                                        if last_db:
                                            last_ee = last_db.get('entry_exit', 'Entry')
                                            self.plate_entry_exit[clean_text] = 'Exit' if last_ee == 'Entry' else 'Entry'

                                    entry_exit = self.plate_entry_exit.get(clean_text, "Entry")
                                    log_id = self._log_plate_detection(
                                        camera_id=camera_id,
                                        plate_number=clean_text,
                                        confidence=det.get("confidence", 0),
                                        bbox=det.get("bbox", {}),
                                        frame=frame,
                                        entry_exit=entry_exit
                                    )
                                    det["violation_log_id"] = log_id
                                    det["entry_exit"] = entry_exit
                                    self.plate_log_cooldown[cooldown_key] = now
                                    self.plate_entry_exit[clean_text] = "Exit" if entry_exit == "Entry" else "Entry"
                                    print(f"🚗 Plate logged: {clean_text} [{entry_exit}] (cam: {camera_id})")
                                else:
                                    # Within cooldown — show in live view only, no new DB entry
                                    det["entry_exit"] = self.plate_entry_exit.get(clean_text, "Entry")

                                # Only valid Indian plates shown in live feed
                                final_detections.append(det)
                        # Non-Indian format plates are silently dropped

                    if found_indian_plate:
                        # Pause OCR scanning for 5 seconds after a valid plate
                        self.ocr_last_check[camera_id] = now + 5.0
                    else:
                        self.ocr_last_check[camera_id] = now
                        
                    self.ocr_cache[camera_id] = final_detections
                    plate_results_batch.append(final_detections)
                else:
                    plate_results_batch.append(self.ocr_cache.get(camera_id, []))
            except Exception as e:
                print(f"❌ Plate detection error for {camera_id}: {e}")
                plate_results_batch.append([])
        
        # Process compliance
        results_batch = []
        for idx in range(len(frames)):
            camera_id = camera_ids[idx]
            frame = frames[idx]
            face_results = face_results_batch[idx] if idx < len(face_results_batch) else []
            ppe_detections = ppe_batch[idx] if idx < len(ppe_batch) else []
            compliance_results = []
            
            # ✅ TRACKING: Keep track of assigned IDs for this frame to prevent duplicates
            used_track_ids = set()
            
            # 🚨 EMERGENCY CHECK: If fire or fall is detected, suppress PPE/Face logic
            _fire_det = fire_smoke_batch[idx] if idx < len(fire_smoke_batch) else []
            _fall_det = fall_batch[idx] if idx < len(fall_batch) else []
            if len(_fire_det) > 0 or len(_fall_det) > 0:
                face_results = []
                ppe_detections = []
            
            for fr in face_results:
                face_bbox = fr["box"]
                person_id = fr.get("person_id")
                person_name = fr.get("person_name")
                
                # Check known identity cache
                bbox_str = f"{camera_id}:{face_bbox[0]}_{face_bbox[1]}"
                cached = self.identity_cache.get(bbox_str)
                
                track_id = None
                
                if cached:
                    person_id = cached["person_id"]
                    person_name = cached["person_name"]
                    role = cached["role"]
                    is_unknown = False
                elif person_id and person_id != "unknown":
                    person = self.db.get_person(person_id)
                    role = person.get("role", "default") if person else "visitor"
                    is_unknown = False
                else:
                    # ✅ PERSISTENCE FIX: Find the body body for this face to use as anchor
                    tracking_bbox = face_bbox
                    is_body = False
                    
                    for det in ppe_detections:
                        if det.get("category") in ["person", "skeleton"]:
                            p_box = (det["bbox"]["x1"], det["bbox"]["y1"], det["bbox"]["x2"], det["bbox"]["y2"])
                            # Check if face center is inside person body
                            f_cx, f_cy = (face_bbox[0] + face_bbox[2]) / 2, (face_bbox[1] + face_bbox[3]) / 2
                            if (p_box[0] <= f_cx <= p_box[2] and p_box[1] <= f_cy <= p_box[3]):
                                tracking_bbox = p_box
                                is_body = True
                                break
                    
                    # ✅ OPTIMIZED: Use body anchor tracking for unknowns
                    track_id = self._assign_track_id(camera_id, tracking_bbox, is_body=is_body, exclude_ids=used_track_ids)
                    used_track_ids.add(track_id)
                    
                    person_id = track_id
                    person_name = f"Unknown ({track_id})"
                    role = "visitor"
                    is_unknown = True
                
                # ✅ NEW: Mark attendance for known persons
                if not is_unknown:
                    self._mark_attendance_if_needed(person_id, camera_id)
                
                # ✅ OPTIMIZED: Enhanced PPE verification with person tracking
                if is_night_mode:
                    # In Night Mode, we treat even "compliant" state as a violation if person detected
                    # We skip complex PPE check to save resources
                    verified_ppe = []
                    compliance = {
                        "is_compliant": False,  # Breach
                        "compliance_percentage": 0,
                        "missing_ppe": ["NIGHT MODE BREACH"],
                        "required_ppe": [],
                        "wearing_ppe": [],
                        "is_night_mode": True
                    }
                else:
                    verified_ppe = self.ppe_system.verify_ppe_on_person(
                        face_bbox, ppe_detections, frame.shape[0], person_id=person_id
                    )
                    
                    # ✅ OPTIMIZED: Enhanced compliance check
                    compliance = self.ppe_system.check_person_compliance(
                        role, verified_ppe, face_detected=True, person_id=person_id,
                        frame_size=frame.shape[:2], face_bbox=face_bbox
                    )
                
                result = {
                    "camera_id": camera_id,
                    "person_id": person_id,
                    "person_name": person_name,
                    "role": role,
                    "face_bbox": face_bbox,
                    "face_confidence": fr.get("confidence", 0.0),
                    "face_size": fr.get("face_size", (0, 0)),  
                    "detector_type": fr.get("detector", "unknown"),  
                    "ppe_detections": verified_ppe,
                    "compliance": compliance,
                    "is_violation": not compliance["is_compliant"],
                    "is_unknown": is_unknown,
                    "track_id": track_id if is_unknown else person_id,
                    "timestamp": timestamp,
                }

                
                # ✅ NEW: Update status history for frontend
                with self.status_lock:
                    status_key = f"{camera_id}:{person_id}"
                    self.status_history[status_key].append({
                        "timestamp": now_ts,
                        "is_compliant": compliance["is_compliant"],
                        "compliance_percentage": compliance["compliance_percentage"],
                        "compliance_quality": compliance.get("compliance_quality_score", 0.0),
                        "missing_ppe": compliance["missing_ppe"],
                        "wearing_ppe": compliance["wearing_ppe"]
                    })
                
                # Log violations with state tracking
                violation_log_id = None
                zone_violation_log_id = None
                snapshot_url = None
                
                # Determine unique key for state tracking
                track_key = result["person_id"] if not is_unknown else result.get("track_id")
                
                # ✅ Initialize variables to prevent UnboundLocalError
                is_authorized = True
                zone_name = None
                allowed_roles = []
                
                if is_night_mode:
                    # ✅ NEW: NIGHT MODE LOGIC
                    night_key = (camera_id, track_key, "night_breach")
                    if night_key not in self.active_violations:
                        violation_log_id = self._log_night_mode_violation(
                            person_id=result["person_id"],
                            person_name=result["person_name"],
                            camera_id=camera_id,
                            face_bbox=face_bbox,
                            is_unknown=is_unknown,
                            track_id=result["track_id"],
                            annotated_frame=frame
                        )
                        self.active_violation_logs[night_key] = violation_log_id
                    else:
                        violation_log_id = self.active_violation_logs.get(night_key)
                        snapshot_url = f"/alert_images/night_violation_{violation_log_id}.jpg" if violation_log_id else None
                    
                    self.active_violations[night_key] = time.time()
                else:
                    # ------------------------------------------
                    # STANDARD MODE (PPE + ZONE)
                    # ------------------------------------------
                    # ✅ NEW: Check zone access authorization
                    is_authorized, zone_name, allowed_roles = self._check_zone_access(role, camera_id)
                    
                    if not is_authorized:
                        # Zone violation detected!
                        zone_key = (camera_id, track_key, "zone_violation", zone_name)
                        
                        # Only log if not already active
                        if zone_key not in self.active_violations:
                            print(f"🔍 DEBUG: About to log ZONE violation for {result['person_name']} in {zone_name}")
                            
                            zone_violation_log_id = self._log_zone_violation(
                                person_id=result["person_id"],
                                person_name=result["person_name"],
                                role=result["role"],
                                camera_id=camera_id,
                                zone_name=zone_name,
                                allowed_roles=allowed_roles,
                                face_bbox=face_bbox,
                                is_unknown=is_unknown,
                                track_id=result["track_id"],
                                annotated_frame=frame
                            )
                            self.active_violation_logs[zone_key] = zone_violation_log_id
                        else:
                            # Previously active - retrieve existing log ID
                            zone_violation_log_id = self.active_violation_logs.get(zone_key)
                            snapshot_url = f"/alert_images/zone_violation_{zone_violation_log_id}.jpg" if zone_violation_log_id else None
                        
                        # Mark zone violation as active
                        self.active_violations[zone_key] = time.time()
                    else:
                        # Authorized - clear zone violation state
                        zone_key = (camera_id, track_key, "zone_violation")
                        if zone_key in self.active_violations:
                            del self.active_violations[zone_key]
                            if zone_key in self.active_violation_logs:
                                del self.active_violation_logs[zone_key]
                    
                    # Check PPE compliance violations
                    if result["is_violation"]:
                        # Determine unique state key (camera + person + set of missing PPE)
                        missing_tuple = tuple(sorted(compliance.get("missing_ppe", [])))
                        active_key = (camera_id, track_key, "ppe_violation", missing_tuple)

                        # Only log if NOT already active (prevent continuous logging of same state)
                        if active_key not in self.active_violations:
                            print(f"🔍 DEBUG: New PPE violation state for {result['person_name']}: {missing_tuple}")
                            
                            violation_log_id = self._log_ppe_violation(
                                person_id=result["person_id"],
                                person_name=result["person_name"],
                                role=result["role"],
                                camera_id=camera_id,
                                compliance=compliance,
                                face_bbox=face_bbox,
                                is_unknown=is_unknown,
                                track_id=result["track_id"],
                                annotated_frame=frame
                            )
                            self.active_violation_logs[active_key] = violation_log_id
                        else:
                            # Previously active - retrieve existing log ID
                            violation_log_id = self.active_violation_logs.get(active_key)
                            snapshot_url = f"/alert_images/violation_{violation_log_id}.jpg" if violation_log_id else None
                        
                        # Mark as active / Keep session alive (update timestamp)
                        self.active_violations[active_key] = time.time()
                    else:
                        # Completely COMPLIANT -> Clear ALL active violation states for this person/camera
                        # This ensures that if they violate again immediately with the same things, it's logged as a new event
                        keys_to_clear = [k for k in self.active_violations.keys() 
                                        if k[0] == camera_id and k[1] == track_key]
                        for k in keys_to_clear:
                            del self.active_violations[k]
                            if k in self.active_violation_logs:
                                del self.active_violation_logs[k]
                
                result["violation_log_id"] = violation_log_id
                result["zone_violation_log_id"] = zone_violation_log_id
                result["zone_name"] = zone_name
                result["is_zone_authorized"] = is_authorized
                result["snapshot_url"] = snapshot_url if snapshot_url else result.get("snapshot_url")
                if is_night_mode:
                    result["log_type"] = "night_mode_violation"
                compliance_results.append(result)
            
            # ✅ NEW: Multi-Inference - Also detect persons with NO faces detected (Supports both Day/Night Mode)
            for det in ppe_detections:
                if det.get("category") in ["person", "skeleton"]:
                    # Check if this person overlaps with any detected face
                    person_bbox = (det["bbox"]["x1"], det["bbox"]["y1"], det["bbox"]["x2"], det["bbox"]["y2"])
                    has_overlap = False
                    for fr in face_results:
                        f_box = fr["box"]
                        # ✅ Check if face center is inside person body (Robust for varying sizes)
                        f_cx, f_cy = (f_box[0] + f_box[2]) / 2, (f_box[1] + f_box[3]) / 2
                        if (person_bbox[0] <= f_cx <= person_bbox[2] and 
                            person_bbox[1] <= f_cy <= person_bbox[3]):
                            has_overlap = True
                            break
                    
                    if not has_overlap:
                        # Person detected without a face!
                        # ✅ REQUIREMENT: Only handle faceless detections in Night Mode
                        if not is_night_mode:
                            continue
                            
                        track_id = self._assign_track_id(camera_id, person_bbox, is_body=True, exclude_ids=used_track_ids)
                        used_track_ids.add(track_id)
                        violation_log_id = None
                        
                        # ------------------------------------------
                        # NIGHT MODE: Intruder Logic (Faceless)
                        # ------------------------------------------
                        night_key = (camera_id, track_id, "night_breach_detection")
                        
                        if night_key not in self.active_violations:
                            violation_log_id = self._log_night_mode_violation(
                                person_id=track_id,
                                person_name=f"Intruder ({track_id})",
                                camera_id=camera_id,
                                face_bbox=person_bbox, # Use person bbox as face bbox for logging
                                is_unknown=True,
                                track_id=track_id,
                                annotated_frame=frame
                            )
                            self.active_violation_logs[night_key] = violation_log_id
                        else:
                            violation_log_id = self.active_violation_logs.get(night_key)
                        
                        self.active_violations[night_key] = time.time()
                        
                        # Add to status history for frontend count
                        with self.status_lock:
                            status_key = f"{camera_id}:{track_id}"
                            self.status_history[status_key].append({
                                "timestamp": time.time(),
                                "is_compliant": False,
                                "compliance_percentage": 0,
                                "missing_ppe": ["NIGHT MODE"],
                                "wearing_ppe": []
                            })

                        # Add to compliance results for drawing & counting
                        compliance_results.append({
                            "camera_id": camera_id,
                            "person_id": track_id,
                            "person_name": f"Intruder ({track_id})",
                            "face_bbox": person_bbox,
                            "is_violation": True,
                            "is_unknown": True,
                            "compliance": {"is_compliant": False, "missing_ppe": ["NIGHT MODE"]},
                            "log_type": "night_mode_violation",
                            "violation_log_id": violation_log_id,
                            "track_id": track_id
                        })
            
            # 🚨 EMERGENCY PRIORITY: Handle Fire/Smoke — overrides all other detections
            fire_detections = fire_smoke_batch[idx] if idx < len(fire_smoke_batch) else []
            fall_detections = fall_batch[idx] if idx < len(fall_batch) else []
            
            fire_key = (camera_id, "SYSTEM", "FIRE_ALERT")
            fall_key = (camera_id, "SYSTEM", "FALL_ALERT")
            
            # Fetch the actual zone name (location) from the database so the frontend can display it
            camera_info = self.db.get_camera(camera_id) if hasattr(self, 'db') and self.db else None
            camera_zone_name = (camera_info.get("location") or camera_info.get("zone") or camera_id) if camera_info is not None else "Unknown Location"
            
            if len(fire_detections) > 0 or len(fall_detections) > 0:
                # 🚨 EMERGENCY OVERRIDE: Wipe ALL PPE/Zone/Compliant results.
                # Fire/Fall are life-safety emergencies.
                compliance_results = []
                
            if len(fire_detections) > 0:
                # Check if we should create a new DB log for this fire event
                if fire_key not in self.active_violations:
                    print(f"🔥 FIRE OUTBREAK DETECTED ON CAMERA {camera_id} 🔥")
                    # Log the most confident detection as the primary event
                    main_fd = max(fire_detections, key=lambda x: x.get('confidence', 0))
                    fire_log_id, snapshot_url = self._log_fire_violation(
                        camera_id=camera_id,
                        category=main_fd.get('category', 'fire'),
                        confidence=main_fd.get('confidence', 0),
                        bbox=main_fd.get('bbox', {}),
                        annotated_frame=frame
                    )
                    self.active_violation_logs[fire_key] = fire_log_id
                else:
                    fire_log_id = self.active_violation_logs.get(fire_key)
                    snapshot_url = f"/alert_images/fire_{fire_log_id}.jpg" if fire_log_id else None
                
                # Update cooldown timestamp
                self.active_violations[fire_key] = time.time()
                
                # Trigger alerts and build FIRE-ONLY UI entries
                for fd in fire_detections:
                    fd["camera_id"] = camera_id
                    fd["log_id"] = fire_log_id
                    
                    # Create annotated frame for the snapshot
                    annotated_for_alert = frame.copy()
                    category = fd.get('category', 'fire').upper()
                    conf = fd.get('confidence', 0.0)
                    bbox = fd.get('bbox', {})
                    x1, y1 = int(bbox.get('x1', 0)), int(bbox.get('y1', 0))
                    x2, y2 = int(bbox.get('x2', 0)), int(bbox.get('y2', 0))
                    
                    # Draw FIRE/SMOKE bounding box
                    color = (0, 0, 139)  # Dark Red for both fire and smoke
                    cv2.rectangle(annotated_for_alert, (x1, y1), (x2, y2), color, 4)
                    label = f"🔥 {category}: {conf*100:.1f}%"
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated_for_alert, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    cv2.putText(annotated_for_alert, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    fd["annotated_frame"] = annotated_for_alert  # Pass frame for alert engine snapshot
                    
                    # Flag if BOTH are occurring so alert_engine knows
                    fd["has_simultaneous_fall"] = len(fall_detections) > 0
                    
                    # Trigger alert engine immediately (WhatsApp/Terminal)
                    self.alert_engine.handle_fire_smoke_alert(fd)
                    compliance_results.append({
                        "camera_id": camera_id,
                        "person_id": "EMERGENCY",
                        "person_name": f"🔥 {category} DETECTED — EVACUATE NOW",
                        "role": "FIRE EMERGENCY",
                        "is_violation": True,
                        "is_unknown": True,
                        "is_compliant": False,
                        "missing_ppe": [],             # No PPE info — evacuation scenario
                        "is_zone_authorized": True,    # Zone auth irrelevant in emergency
                        "zone_name": camera_zone_name,
                        "log_type": "fire_violation",
                        "fire_category": fd.get('category', 'fire'),
                        "violation_log_id": fire_log_id,
                        "snapshot_url": snapshot_url,
                        "track_id": "fire_event",
                        "timestamp": time.time()
                    })
            else:
                if fire_key in self.active_violations:
                    del self.active_violations[fire_key]
                    if fire_key in self.active_violation_logs:
                        del self.active_violation_logs[fire_key]

            if len(fall_detections) > 0:
                if fall_key not in self.active_violations:
                    print(f"🆘 PERSON LYING ON GROUND DETECTED ON CAMERA {camera_id} 🆘")
                    main_fd = max(fall_detections, key=lambda x: x.get('confidence', 0))
                    fall_log_id, snapshot_url = self._log_fall_violation(
                        camera_id=camera_id,
                        confidence=main_fd.get('confidence', 0),
                        bbox=main_fd.get('bbox', {}),
                        annotated_frame=frame
                    )
                    self.active_violation_logs[fall_key] = fall_log_id
                else:
                    fall_log_id = self.active_violation_logs.get(fall_key)
                    snapshot_url = f"/alert_images/fall_{fall_log_id}.jpg" if fall_log_id else None
                
                self.active_violations[fall_key] = time.time()
                
                for fd in fall_detections:
                    fd["camera_id"] = camera_id
                    fd["log_id"] = fall_log_id
                    
                    annotated_for_alert = frame.copy()
                    conf = fd.get('confidence', 0.0)
                    bbox = fd.get('bbox', {})
                    x1, y1 = int(bbox.get('x1', 0)), int(bbox.get('y1', 0))
                    x2, y2 = int(bbox.get('x2', 0)), int(bbox.get('y2', 0))
                    
                    color = (0, 165, 255) # Orange
                    cv2.rectangle(annotated_for_alert, (x1, y1), (x2, y2), color, 4)
                    label = f"🆘 LYING PERSON: {conf*100:.1f}%"
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated_for_alert, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    cv2.putText(annotated_for_alert, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    fd["annotated_frame"] = annotated_for_alert
                    
                    # Flag if BOTH are occurring so alert_engine knows
                    fd["has_simultaneous_fire"] = len(fire_detections) > 0
                    
                    self.alert_engine.handle_fall_alert(fd)
                    
                    compliance_results.append({
                        "camera_id": camera_id,
                        "person_id": "EMERGENCY_FALL",
                        "person_name": f"🆘 PERSON LYING ON GROUND",
                        "role": "DANGER",
                        "is_violation": True,
                        "is_unknown": True,
                        "is_compliant": False,
                        "missing_ppe": [],
                        "is_zone_authorized": True,
                        "zone_name": camera_zone_name,
                        "log_type": "fall_violation",
                        "violation_log_id": fall_log_id,
                        "snapshot_url": snapshot_url,
                        "track_id": "fall_event",
                        "timestamp": time.time()
                    })
            else:
                if fall_key in self.active_violations:
                    del self.active_violations[fall_key]
                    if fall_key in self.active_violation_logs:
                        del self.active_violation_logs[fall_key]
            
            # Debug log only for non-emergency frames
            if compliance_results and not any(r.get("log_type") in ["fire_violation", "fall_violation"] for r in compliance_results):
                print(f"📈 DEBUG {camera_id}: Found {len(face_results)} faces + {len(compliance_results)-len(face_results)} faceless persons")
            
            results_batch.append({
                "face_results": face_results,
                "ppe_detections": ppe_detections,
                "compliance_results": compliance_results,
                "fire_smoke_detections": fire_detections,
                "fall_detections": fall_detections,
                "plate_results": plate_results_batch[idx] if idx < len(plate_results_batch) else [],
                "timestamp": timestamp,
                "frame_quality_metrics": self._calculate_frame_metrics(compliance_results)
            })
        
        # Cleanup stale cache
        now = time.time()
        stale_keys = [k for k in list(self.identity_cache.keys()) 
                     if now - self.identity_cache[k]["last_seen"] > 30]
        for k in stale_keys:
            del self.identity_cache[k]
            
        # Cleanup expired violation sessions (e.g. person left for > 15s)
        expired_sessions = [k for k, last_seen in self.active_violations.items()
                           if now - last_seen > 15]
        for k in expired_sessions:
            del self.active_violations[k]
            if k in self.active_violation_logs:
                del self.active_violation_logs[k]
        
        return results_batch
    
    def _calculate_frame_metrics(self, compliance_results: List[Dict]) -> Dict:
        """
        ✅ NEW: Calculate frame-level quality metrics for status panel
        """
        if not compliance_results:
            return {
                "total_persons": 0,
                "compliant_count": 0,
                "violation_count": 0,
                "unknown_count": 0,
                "avg_compliance_percentage": 0.0,
                "avg_quality_score": 0.0
            }
        
        total = len(compliance_results)
        compliant = sum(1 for r in compliance_results if r.get("compliance", {}).get("is_compliant", False))
        violations = sum(1 for r in compliance_results if r.get("is_violation", False))
        unknown = sum(1 for r in compliance_results if r.get("is_unknown", False))
        
        avg_compliance = np.mean([r.get("compliance", {}).get("compliance_percentage", 0.0) for r in compliance_results])
        avg_quality = np.mean([r.get("compliance", {}).get("compliance_quality_score", 0.0) for r in compliance_results])
        
        return {
            "total_persons": total,
            "compliant_count": compliant,
            "violation_count": violations,
            "unknown_count": unknown,
            "avg_compliance_percentage": float(avg_compliance),
            "avg_quality_score": float(avg_quality)
        }
    
    def get_status_summary(self, camera_id: str = None) -> Dict:
        """
        ✅ NEW: Get comprehensive status summary for frontend
        """
        with self.status_lock:
            if camera_id:
                # Filter by camera
                relevant_keys = [k for k in self.status_history.keys() if k.startswith(f"{camera_id}:")]
            else:
                relevant_keys = list(self.status_history.keys())
            
            if not relevant_keys:
                print(f"📊 DEBUG: Status Summary for {camera_id}: 0 persons (history empty)")
                return {
                    "active_persons": 0,
                    "compliant_persons": 0,
                    "violation_persons": 0,
                    "avg_compliance": 0.0,
                    "avg_quality": 0.0,
                    "status_details": []
                }
            
            print(f"📊 DEBUG: Status Summary for {camera_id}: {len(relevant_keys)} active person keys")
            
            status_details = []
            compliant_count = 0
            violation_count = 0
            
            for key in relevant_keys:
                history = self.status_history[key]
                if not history:
                    continue
                
                # Get most recent status
                recent = list(history)[-1]
                cam_id, person_id = key.split(":", 1)
                
                if recent["is_compliant"]:
                    compliant_count += 1
                else:
                    violation_count += 1
                
                status_details.append({
                    "camera_id": cam_id,
                    "person_id": person_id,
                    "is_compliant": recent["is_compliant"],
                    "compliance_percentage": recent["compliance_percentage"],
                    "compliance_quality": recent["compliance_quality"],
                    "missing_ppe": recent["missing_ppe"],
                    "wearing_ppe": recent["wearing_ppe"],
                    "last_update": recent["timestamp"]
                })
            
            avg_compliance = np.mean([s["compliance_percentage"] for s in status_details]) if status_details else 0.0
            avg_quality = np.mean([s["compliance_quality"] for s in status_details]) if status_details else 0.0
            
            return {
                "active_persons": len(status_details),
                "compliant_persons": compliant_count,
                "violation_persons": violation_count,
                "avg_compliance": avg_compliance,
                "avg_quality": avg_quality,
                "status_details": status_details
            }
    
    def _check_zone_access(self, person_role: str, camera_id: str) -> tuple:
        """
        ✅ NEW: Check if person's role is authorized for the zone
        Returns: (is_authorized: bool, zone_name: str, allowed_roles: list)
        """
        try:
            # Get camera to find its zone
            camera = self.db.get_camera(camera_id)
            if not camera:
                return (True, "unknown", [])  # No camera found, allow access
            
            zone_name = camera.get("zone", "default")
            
            # Get zone configuration
            zone = self.db.get_zone(zone_name)
            if not zone:
                return (True, zone_name, [])  # No zone config, allow access
            
            allowed_roles = zone.get("allowed_roles", [])
            
            # Empty list means all roles are allowed
            if not allowed_roles:
                return (True, zone_name, [])
            
            # Check if person's role is in the allowed list (case-insensitive)
            person_role_lower = person_role.lower()
            allowed_roles_lower = [r.lower() for r in allowed_roles]
            
            is_authorized = person_role_lower in allowed_roles_lower
            
            return (is_authorized, zone_name, allowed_roles)
            
        except Exception as e:
            print(f"❌ Error checking zone access: {e}")
            return (True, "unknown", [])  # On error, allow access to prevent false positives
    
    def _log_zone_violation(self, person_id: str, person_name: str, role: str, camera_id: str,
                           zone_name: str, allowed_roles: list, face_bbox: Tuple, 
                           is_unknown: bool = False, track_id: str = None, annotated_frame = None):
        """
        ✅ NEW: Log zone access violations (unauthorized role in restricted zone)
        """
        try:
            camera = self.db.get_camera(camera_id)
            
            # Generate violation log
            violation_log = {
                'log_type': 'zone_violation',  # Different type from PPE violations
                'person_id': person_id,
                'person_name': person_name,
                'role': role,
                'is_unknown_person': is_unknown,
                'track_id': track_id,
                'camera_id': camera_id,
                'camera_name': camera.get('name', camera_id) if camera else camera_id,
                'location': camera.get('location', 'Unknown') if camera else 'Unknown',
                'zone': zone_name,
                'allowed_roles': allowed_roles,
                'violation_reason': f"Unauthorized role '{role}' in zone '{zone_name}'",
                'violation_type': 'unauthorized_zone_access',
                'face_bbox': {'x1': int(face_bbox[0]), 'y1': int(face_bbox[1]), 
                             'x2': int(face_bbox[2]), 'y2': int(face_bbox[3])},
                'timestamp': datetime.utcnow(),
                'is_alert': True,
                'alert_sent': False,
                'resolved': False,
                'snapshot_url': None
            }
            
            result = self.db.recognition_logs.insert_one(violation_log)
            log_id = str(result.inserted_id)
            
            # Save snapshot if annotated frame is provided
            snapshot_url = None
            if annotated_frame is not None:
                try:
                    from pathlib import Path
                    import cv2
                    import numpy as np
                    from path_utils import get_resource_path
                    
                    folder = Path(get_resource_path("alert_images"))
                    folder.mkdir(exist_ok=True)
                    
                    filename = f"zone_violation_{log_id}.jpg"
                    path = folder / filename
                    
                    # Create annotated copy of the frame
                    annotated = annotated_frame.copy()
                    
                    # Draw face bounding box in RED for zone violations
                    x1, y1, x2, y2 = int(face_bbox[0]), int(face_bbox[1]), int(face_bbox[2]), int(face_bbox[3])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red
                    
                    # Draw violation message
                    allowed_text = ", ".join([r.replace('_', ' ').title() for r in allowed_roles])
                    label = f"{person_name} - WRONG ZONE"
                    sublabel = f"Zone: {zone_name} (Allowed: {allowed_text})"
                    
                    # Background for text
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    (sub_width, sub_height), _ = cv2.getTextSize(sublabel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    
                    cv2.rectangle(annotated, (x1, y1 - text_height - sub_height - 20), 
                                (x1 + max(text_width, sub_width) + 10, y1), (0, 0, 255), -1)
                    cv2.putText(annotated, label, (x1 + 5, y1 - sub_height - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated, sublabel, (x1 + 5, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Save the annotated frame
                    cv2.imwrite(str(path), annotated)
                    snapshot_url = f"/alert_images/{filename}"
                    
                    # Update the log with snapshot URL
                    self.db.recognition_logs.update_one(
                        {"_id": result.inserted_id},
                        {"$set": {"snapshot_url": snapshot_url}}
                    )
                    
                    print(f"📸 Zone violation snapshot saved: {snapshot_url}")
                    
                except Exception as e:
                    print(f"⚠️ Failed to save zone violation snapshot: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"🚫 ZONE VIOLATION logged: {person_name} ({role}) @ {zone_name} - Log ID: {log_id}")
            return log_id
            
        except Exception as e:
            print(f"❌ Failed to log zone violation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _log_ppe_violation(self, person_id: str, person_name: str, role: str, camera_id: str,
                          compliance: Dict, face_bbox: Tuple, is_unknown: bool = False, 
                          track_id: str = None, annotated_frame = None):
        """
        ✅ ENHANCED: Log violations with detailed compliance metrics and save snapshot
        """
        try:
            camera = self.db.get_camera(camera_id)
            
            # Generate violation log first to get the ID
            violation_log = {
                'log_type': 'ppe_violation',
                'person_id': person_id,
                'person_name': person_name,
                'role': role,
                'is_unknown_person': is_unknown,
                'track_id': track_id,
                'camera_id': camera_id,
                'camera_name': camera.get('name', camera_id) if camera else camera_id,
                'location': camera.get('location', 'Unknown') if camera else 'Unknown',
                'missing_ppe': compliance['missing_ppe'],
                'missing_ppe_priority': compliance.get('missing_priority', compliance['missing_ppe']),
                'required_ppe': compliance['required_ppe'],
                'wearing_ppe': compliance['wearing_ppe'],
                'wearing_details': compliance.get('wearing_details', {}),
                'compliance_percentage': compliance['compliance_percentage'],
                'compliance_quality_score': compliance.get('compliance_quality_score', 0.0),
                'temporal_inference_used': compliance.get('temporal_inference_used', False),
                'face_bbox': {'x1': int(face_bbox[0]), 'y1': int(face_bbox[1]), 
                             'x2': int(face_bbox[2]), 'y2': int(face_bbox[3])},
                'timestamp': datetime.utcnow(),
                'is_alert': True,
                'alert_sent': False,
                'resolved': False,
                'snapshot_url': None  # Will be updated after saving
            }
            
            result = self.db.recognition_logs.insert_one(violation_log)
            log_id = str(result.inserted_id)
            
            # Save snapshot if annotated frame is provided
            snapshot_url = None
            if annotated_frame is not None:
                try:
                    print(f"🔍 DEBUG: Attempting to save snapshot for violation {log_id}")
                    print(f"🔍 DEBUG: Frame shape: {annotated_frame.shape if hasattr(annotated_frame, 'shape') else 'No shape'}")
                    
                    from pathlib import Path
                    import cv2
                    import numpy as np
                    from path_utils import get_resource_path
                    
                    folder = Path(get_resource_path("alert_images"))
                    folder.mkdir(exist_ok=True)
                    print(f"🔍 DEBUG: Alert images folder: {folder}")
                    
                    filename = f"violation_{log_id}.jpg"
                    path = folder / filename
                    print(f"🔍 DEBUG: Full path: {path}")
                    
                    # Create annotated copy of the frame
                    annotated = annotated_frame.copy()
                    
                    # Draw face bounding box IN ORANGE for PPE violations
                    x1, y1, x2, y2 = int(face_bbox[0]), int(face_bbox[1]), int(face_bbox[2]), int(face_bbox[3])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange
                    
                    # Draw person name and missing PPE
                    label = f"{person_name}"
                    if compliance['missing_ppe']:
                        label += f" - Missing: {', '.join(compliance['missing_ppe'])}"
                    
                    # Background for text (Orange for PPE violation)
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 165, 255), -1)
                    cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Save the annotated frame
                    success = cv2.imwrite(str(path), annotated)
                    print(f"🔍 DEBUG: cv2.imwrite success: {success}")
                    
                    if success:
                        snapshot_url = f"/alert_images/{filename}"
                        
                        # Update the log with snapshot URL
                        update_result = self.db.recognition_logs.update_one(
                            {"_id": result.inserted_id},
                            {"$set": {"snapshot_url": snapshot_url}}
                        )
                        print(f"🔍 DEBUG: Database update matched: {update_result.matched_count}, modified: {update_result.modified_count}")
                        
                        print(f"📸 Snapshot saved: {snapshot_url}")
                    else:
                        print(f"❌ cv2.imwrite failed to save image")
                    
                except Exception as e:
                    print(f"⚠️ Failed to save snapshot: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️ No annotated_frame provided for violation {log_id}")
            
            print(f"🚨 Violation logged: {person_name} @ {camera_id} (Quality: {compliance.get('compliance_quality_score', 0.0):.2f}) - Log ID: {log_id}, Snapshot: {snapshot_url}")
            return log_id
            
        except Exception as e:
            print(f"❌ Failed to log violation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _log_fire_violation(self, camera_id: str, category: str, confidence: float, bbox: Dict, annotated_frame=None):
        """
        ✅ OPTIMIZED: Async fire/smoke logging to eliminate delays
        """
        try:
            from pathlib import Path
            import cv2
            # import numpy as np # numpy is not used here
            from path_utils import get_resource_path
            from bson import ObjectId
            import threading # Added for async operations

            camera = self.db.get_camera(camera_id)
            display_name = category.upper()
            
            # 1. Pre-generate Log ID to include snapshot_url in the FIRST insertion
            log_id_obj = ObjectId()
            log_id = str(log_id_obj)
            snapshot_url = f"/alert_images/fire_{log_id}.jpg" if annotated_frame is not None else None

            violation_log = {
                '_id': log_id_obj,
                'log_type': 'fire_violation',
                'person_id': 'SYSTEM_FIRE',
                'person_name': f"FIRE: {display_name}",
                'role': 'DANGER',
                'is_unknown_person': True,
                'camera_id': camera_id,
                'camera_name': camera.get('name', camera_id) if camera else camera_id,
                'location': (camera.get('location') or camera.get('zone') or 'Unknown') if camera else 'Unknown',
                # ✅ Store zone_name explicitly for consistent querying by UI
                'zone_name': (camera.get('location') or camera.get('zone') or 'Unknown Zone') if camera else 'Unknown Zone',
                'fire_category': category,
                'confidence': confidence,
                'timestamp': datetime.utcnow(),
                'is_alert': True,
                'alert_sent': False,
                'resolved': False,
                'snapshot_url': snapshot_url
            }
            
            # Fast Insertion (Single roundtrip)
            self.db.recognition_logs.insert_one(violation_log)
            
            # 2. Offload Image saving to Background Thread
            if annotated_frame is not None:
                # Copy frame in main thread to ensure buffer stays valid
                frame_to_save = annotated_frame.copy()
                
                def save_snapshot_async(f, b, c, dn, lid):
                    try:
                        folder = Path(get_resource_path("alert_images"))
                        folder.mkdir(exist_ok=True)
                        path = folder / f"fire_{lid}.jpg"
                        
                        # Drawing is relatively light but keeping it here hides all latency
                        x1, y1 = int(b.get('x1', 0)), int(b.get('y1', 0))
                        x2, y2 = int(b.get('x2', 0)), int(b.get('y2', 0))
                        
                        if isinstance(b, list) and len(b) == 4:
                            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        
                        cv2.rectangle(f, (x1, y1), (x2, y2), (0, 0, 139), 3)
                        label = f"🔥 {dn}: {c*100:.1f}%"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(f, (x1, y1 - th - 10), (x1 + tw, y1), (0, 0, 139), -1)
                        cv2.putText(f, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        # Bottleneck (Disk I/O) happens here
                        cv2.imwrite(str(path), f)
                    except Exception as e:
                        print(f"⚠️ Async Fire snapshot failed: {e}")
                
                threading.Thread(target=save_snapshot_async, args=(frame_to_save, bbox, confidence, display_name, log_id), daemon=True).start()
                
            return log_id, snapshot_url
        except Exception as e:
            print(f"❌ Failed to log fire violation: {e}")
            return None, None

    def _log_fall_violation(self, camera_id: str, confidence: float, bbox: Dict, annotated_frame=None):
        """
        ✅ OPTIMIZED: Async fall logging to eliminate delays
        """
        try:
            from pathlib import Path
            import cv2
            from path_utils import get_resource_path
            from bson import ObjectId
            
            camera = self.db.get_camera(camera_id)
            
            # 1. Pre-generate Log ID
            log_id_obj = ObjectId()
            log_id = str(log_id_obj)
            snapshot_url = f"/alert_images/fall_{log_id}.jpg" if annotated_frame is not None else None

            violation_log = {
                '_id': log_id_obj,
                'log_type': 'fall_violation',
                'person_id': 'SYSTEM_FALL',
                'person_name': "PERSON LYING ON GROUND",
                'role': 'DANGER',
                'is_unknown_person': True,
                'camera_id': camera_id,
                'camera_name': camera.get('name', camera_id) if camera else camera_id,
                'location': camera.get('location', 'Unknown') if camera else 'Unknown',
                'zone_name': camera.get('location', 'Unknown Zone') if camera else 'Unknown Zone',
                'confidence': confidence,
                'timestamp': datetime.utcnow(),
                'is_alert': True,
                'alert_sent': False,
                'resolved': False,
                'snapshot_url': snapshot_url
            }
            
            self.db.recognition_logs.insert_one(violation_log)
            
            # 2. Async Snapshot Save
            if annotated_frame is not None:
                frame_to_save = annotated_frame.copy()
                
                def save_fall_async(f, b, c, lid):
                    try:
                        folder = Path(get_resource_path("alert_images"))
                        folder.mkdir(exist_ok=True)
                        path = folder / f"fall_{lid}.jpg"
                        
                        x1, y1 = int(b.get('x1', 0)), int(b.get('y1', 0))
                        x2, y2 = int(b.get('x2', 0)), int(b.get('y2', 0))
                        
                        cv2.rectangle(f, (x1, y1), (x2, y2), (0, 165, 255), 3)
                        label = f"🆘 LYING PERSON: {c*100:.1f}%"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(f, (x1, y1 - th - 10), (x1 + tw, y1), (0, 165, 255), -1)
                        cv2.putText(f, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        cv2.imwrite(str(path), f)
                    except Exception as e:
                        print(f"⚠️ Async Fall snapshot failed: {e}")
                
                threading.Thread(target=save_fall_async, args=(frame_to_save, bbox, confidence, log_id), daemon=True).start()
            
            return log_id, snapshot_url
        except Exception as e:
            print(f"❌ Failed to log fall violation: {e}")
            return None, None

    def _log_plate_detection(self, camera_id: str, plate_number: str, confidence: float, bbox: Dict, frame=None, entry_exit: str = "Entry"):
        """
        ✅ NEW: Log plate detection to MongoDB and save snapshot.
        """
        try:
            from pathlib import Path
            import cv2
            import numpy as np
            from path_utils import get_resource_path
            from bson import ObjectId

            camera = self.db.get_camera(camera_id)
            
            log_entry = {
                'log_type': 'plate_detection',
                'person_id': f"PLATE_{plate_number}",
                'person_name': f"Plate: {plate_number}",
                'role': 'VEHICLE',
                'is_unknown_person': True,
                'camera_id': camera_id,
                'camera_name': camera.get('name', camera_id) if camera else camera_id,
                'location': camera.get('location', 'Unknown') if camera else 'Unknown',
                'plate_number': plate_number,
                'confidence': confidence,
                'entry_exit': entry_exit,        # "Entry" or "Exit"
                'timestamp': datetime.utcnow(),     # Use UTC consistently
                'is_alert': False,
                'snapshot_url': None
            }
            
            result = self.db.recognition_logs.insert_one(log_entry)
            log_id = str(result.inserted_id)
            
            if frame is not None:
                try:
                    folder = Path(get_resource_path("alert_images"))
                    folder.mkdir(exist_ok=True)
                    filename = f"plate_{log_id}.jpg"
                    path = folder / filename
                    
                    annotated = frame.copy()
                    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"PLATE: {plate_number} ({confidence*100:.1f}%)"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                    cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    if cv2.imwrite(str(path), annotated):
                        self.db.recognition_logs.update_one(
                            {"_id": result.inserted_id},
                            {"$set": {"snapshot_url": f"/alert_images/{filename}"}}
                        )
                except Exception as e:
                    print(f"⚠️ Snapshot error for plate: {e}")
            
            return log_id
        except Exception as e:
            print(f"❌ Plate logging error: {e}")
            return None

    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        ✅ ENHANCED: Draw results with quality indicators
        🚨 EMERGENCY PRIORITY: If fire is detected, suppress drawing all other PPE/Zone overlays
        """
        annotated = frame.copy()
        
        has_fire = len(results.get('fire_smoke_detections', [])) > 0
        has_fall = len(results.get('fall_detections', [])) > 0
        
        # Draw PPE only if no fire/fall
        if not (has_fire or has_fall):
            annotated = self.ppe_system.draw_ppe_detections(annotated, results['ppe_detections'], show_unverified=False)
        
        for cr in results['compliance_results']:
            if 'face_bbox' not in cr:
                continue
            x1, y1, x2, y2 = cr['face_bbox']
            
            # ✅ NEW: Color priority (Zone Violation > Night Mode > PPE Violation > Unknown > Compliant)
            is_zone_violation = not cr.get('is_zone_authorized', True)
            is_night_violation = cr.get('log_type') == 'night_mode_violation'
            is_ppe_violation = cr.get('is_violation', False)
            
            if is_zone_violation:
                color = (0, 0, 255)  # Red for Wrong Zone
            elif is_night_violation:
                color = (255, 0, 255) # Magenta for Night Mode
                cr['compliance']['is_night_mode_violation'] = True
            elif is_ppe_violation:
                color = (0, 165, 255)  # Orange for PPE Violation
            elif cr.get('is_unknown'):
                color = (255, 165, 0)  # Sky Blue for Unknown
            else:
                color = (0, 255, 0)  # Green for Compliant
                
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Pass zone status to compliance info for label drawing
            cr['compliance']['is_zone_authorized'] = not is_zone_violation
            annotated = self.ppe_system.draw_face_label(
                annotated, (x1, y1, x2, y2), 
                cr['person_name'], cr['compliance'], cr['is_unknown']
            )
        
        # ✅ NEW: Draw fire and smoke detections
        for fd in results.get('fire_smoke_detections', []):
            x1, y1, x2, y2 = fd['bbox']['x1'], fd['bbox']['y1'], fd['bbox']['x2'], fd['bbox']['y2']
            cat = fd['category'].upper()
            conf = fd['confidence']
            
            # Fire/Smoke = Dark Red
            color = (0, 0, 139)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"🔥 {cat} {conf*100:.1f}%"
            
            # Add a more prominent label for fire/smoke
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1-th-10), (x1+tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ✅ NEW: Draw fall detections
        for fd in results.get('fall_detections', []):
            x1, y1, x2, y2 = fd['bbox']['x1'], fd['bbox']['y1'], fd['bbox']['x2'], fd['bbox']['y2']
            conf = fd['confidence']
            color = (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"🆘 LYING PERSON {conf*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1-th-10), (x1+tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ✅ NEW: Draw frame metrics
        metrics = results.get('frame_quality_metrics', {})
        h = annotated.shape[0]
        
        cv2.putText(annotated, f"Persons: {metrics.get('total_persons', 0)}", 
                   (10, h-65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"Compliant: {metrics.get('compliant_count', 0)}", 
                   (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        violations = metrics.get('violation_count', 0)
        # ✅ NEW: Draw License Plate results
        for pr in results.get('plate_results', []):
            bbox = pr.get('bbox', {})
            x1, y1 = int(bbox.get('x1', 0)), int(bbox.get('y1', 0))
            x2, y2 = int(bbox.get('x2', 0)), int(bbox.get('y2', 0))
            plate_num = pr.get('plate_number', 'UNKNOWN')
            conf = pr.get('confidence', 0)
            
            # Bright Green/Cyan for plates
            color = (0, 255, 127) # Spring Green
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            label = f"🔢 {plate_num} ({conf*100:.1f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return annotated

    def _log_night_mode_violation(self, person_id: str, person_name: str, camera_id: str,
                                 face_bbox: Tuple, is_unknown: bool = False, 
                                 track_id: str = None, annotated_frame = None):
        """
        ✅ NEW: Log security violations during Night Mode
        """
        try:
            camera = self.db.get_camera(camera_id)
            
            violation_log = {
                'log_type': 'night_mode_violation',
                'person_id': person_id,
                'person_name': person_name,
                'is_unknown_person': is_unknown,
                'track_id': track_id,
                'camera_id': camera_id,
                'camera_name': camera.get('name', camera_id) if camera else camera_id,
                'location': camera.get('location', 'Unknown') if camera else 'Unknown',
                'violation_reason': "Intrusion detected during Night Mode hours",
                'violation_type': 'night_mode_breach',
                'face_bbox': {'x1': int(face_bbox[0]), 'y1': int(face_bbox[1]), 
                             'x2': int(face_bbox[2]), 'y2': int(face_bbox[3])},
                'timestamp': datetime.utcnow(),
                'is_alert': True,
                'alert_sent': False,
                'resolved': False,
                'snapshot_url': None
            }
            
            result = self.db.recognition_logs.insert_one(violation_log)
            log_id = str(result.inserted_id)
            
            # Save snapshot
            snapshot_url = None
            annotated = annotated_frame  # Default to raw frame if no annotation happens
            
            if annotated_frame is not None:
                try:
                    from pathlib import Path
                    from path_utils import get_resource_path
                    import cv2
                    
                    folder = Path(get_resource_path("alert_images"))
                    folder.mkdir(exist_ok=True)
                    
                    filename = f"night_violation_{log_id}.jpg"
                    path = folder / filename
                    
                    annotated = annotated_frame.copy()
                    x1, y1, x2, y2 = int(face_bbox[0]), int(face_bbox[1]), int(face_bbox[2]), int(face_bbox[3])
                    
                    # Special Night Mode Styling: Purple/Neon style
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    
                    label = f"NIGHT BREACH: {person_name}"
                    cv2.putText(annotated, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
                    cv2.imwrite(str(path), annotated)
                    snapshot_url = f"/alert_images/{filename}"
                    
                    self.db.recognition_logs.update_one(
                        {"_id": result.inserted_id},
                        {"$set": {"snapshot_url": snapshot_url}}
                    )
                except Exception as e:
                    print(f"⚠️ Failed to save night mode snapshot: {e}")
            
            # Trigger alert via engine
            alert_data = {
                "camera_id": camera_id,
                "person_id": person_id,
                "person_name": person_name,
                "log_id": log_id,
                "annotated_frame": annotated  # ✅ Pass the ANNOTATED frame so the alert has the box
            }
            self.alert_engine.handle_night_mode_violation(alert_data)
            
            return log_id
            
        except Exception as e:
            print(f"❌ Failed to log night mode violation: {e}")
            return None

    def _get_dominant_color(self, image_crop: np.ndarray) -> str:
        """
        Determine the dominant color of an image crop (e.g. helmet)
        using HSV color ranges.
        """
        if image_crop is None or image_crop.size == 0:
            return None
            
        hsv_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
        
        max_pixels = 0
        dominant_color = None
        
        for color_name, (lower, upper) in self.COLOR_RANGES.items():
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(hsv_crop, lower_np, upper_np)
            
            # Special handling for red (it wraps around 180)
            if color_name == 'red1':
                continue # Skip red1, handle with red2 combined if needed, or just rely on ranges
            
            if color_name == 'red2':
                # Combine red1 and red2
                lower1 = np.array(self.COLOR_RANGES['red1'][0], dtype=np.uint8)
                upper1 = np.array(self.COLOR_RANGES['red1'][1], dtype=np.uint8)
                mask1 = cv2.inRange(hsv_crop, lower1, upper1)
                mask = cv2.bitwise_or(mask, mask1)
                final_name = 'red'
            else:
                final_name = color_name
                
            pixel_count = cv2.countNonZero(mask)
            
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                dominant_color = final_name
                
        # Must have significant color presence (e.g. > 10% of pixels) to be confident
        total_pixels = image_crop.shape[0] * image_crop.shape[1]
        if max_pixels < (total_pixels * 0.1):
            return None
            
        return dominant_color

    def _get_person_ppe(self, person_bbox: Tuple, ppe_detections: List[Dict]) -> List[str]:
        """Find PPE items overlapping with person bbox"""
        person_ppe = []
        px1, py1, px2, py2 = person_bbox
        
        for det in ppe_detections:
            cat = det.get("category")
            if cat == "person": continue
            
            dx1, dy1, dx2, dy2 = det["bbox"]["x1"], det["bbox"]["y1"], det["bbox"]["x2"], det["bbox"]["y2"]
            
            # Check overlap
            overlap_x1 = max(px1, dx1)
            overlap_y1 = max(py1, dy1)
            overlap_x2 = min(px2, dx2)
            overlap_y2 = min(py2, dy2)
            
            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                # If overlapped, associate
                person_ppe.append(det)
                
        return person_ppe

    def _infer_role_from_ppe(self, frame: np.ndarray, ppe_items: List[Dict]) -> str:
        """Infer role based on PPE color (e.g. Helmet)"""
        default_role = "worker" # Default assumption
        
        for item in ppe_items:
            if item["category"] in ["safety_helmet", "helmet", "hardhat"]:
                bbox = item["bbox"]
                x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
                
                # Validation checks
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x2 > frame.shape[1]: x2 = frame.shape[1]
                if y2 > frame.shape[0]: y2 = frame.shape[0]
                
                if x2 - x1 > 5 and y2 - y1 > 5:
                    helmet_crop = frame[y1:y2, x1:x2]
                    color = self._get_dominant_color(helmet_crop)
                    
                    if color and color in self.ROLE_COLOR_MAPPING:
                        return self.ROLE_COLOR_MAPPING[color]
        
        return default_role

    def process_single_frame(self, frame: np.ndarray, camera_id: str) -> Dict:
        """Process single frame (for API endpoint)"""
        results = self.process_frames_with_ppe_batch([frame], [camera_id], run_face=True)
        if results:
            result = results[0]
            result['processing_time_ms'] = 0
            return result
        return None
