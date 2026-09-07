"""
Enhanced REST API Server for InsightFace Recognition + PPE Detection System
Provides HTTP endpoints for all system operations including PPE compliance
"""

from path_utils import get_resource_path, get_app_data_dir

import os
# Set OpenCV/FFMPEG environment variables to fix H.264 decoding errors and reduce log spam
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp|'
    'fflags;nobuffer|'
    'flags;low_delay|'
    'max_delay;500000|'
    'reorder_queue_size;0|'
    'buffer_size;1024000'
)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import cv2
# cv2.setLogLevel(0)
from flask import send_from_directory
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

import cv2, threading, io, base64, os, time, json
from bson import ObjectId
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from werkzeug.utils import secure_filename
from threading import Thread, Lock
from mongo_db_manager import FaceRecognitionDB
from insightface_recognition_system import InsightFaceRecognitionSystem
from camera_detection_manager import CameraDetectionManager
from inference_controller import InferenceController
from alert_engine import AlertEngine

# ============================================================================
# Flask App Configuration
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for web clients

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max payload size
app.config['UPLOAD_FOLDER'] = os.path.join(get_app_data_dir(), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Suppress OpenCV warnings
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|analyzeduration;1000000|probesize;1000000'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
# cv2.setLogLevel(0)

# ============================================================================
# Load MongoDB Connection from JSON
# ============================================================================

def load_mongodb_connection():
    """Load MongoDB connection string from compass-connections.json"""
    try:
        json_path = get_resource_path('compass-connections.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                if 'connections' in data and len(data['connections']) > 0:
                    connection_string = data['connections'][0]['connectionOptions']['connectionString']
                    print(f"✅ Loaded MongoDB connection from {json_path}")
                    return connection_string
    except Exception as e:
        print(f"⚠️  Failed to load MongoDB connection from JSON: {e}")
    
    # Fallback to environment variable or default
    return os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')

# Initialize database with connection from JSON
mongodb_uri = load_mongodb_connection()
db = FaceRecognitionDB(
    connection_string=mongodb_uri,
    database_name=os.getenv('DATABASE_NAME', 'face_recognition')
)

# ============================================================================
# Initialize InsightFace Recognition System
# ============================================================================

print("🚀 Initializing InsightFace Recognition System...")
try:
    face_system = InsightFaceRecognitionSystem(
        db=db,
        similarity_threshold=float(os.getenv('FACE_THRESHOLD', '0.4')),  # Lowered to 0.3 for better recognition
        use_cuda=True,
        model_name='buffalo_l'
    )
    print("✅ InsightFace System Loaded Successfully")
    print(f"   Model: buffalo_l")
    print(f"   GPU: Enabled")
    print(f"   Threshold: {face_system.similarity_threshold}")
except Exception as e:
    print(f"❌ Error loading InsightFace: {e}")
    import traceback
    traceback.print_exc()
    raise

# ✅ OPTIMIZED: Use optimized PPE system with multi-tier confidence and temporal tracking
import os

ppe_model_path = os.getenv('PPE_MODEL_PATH', get_resource_path('models/best.onnx'))
# ✅ NEW: Specialized person detection model for Night Mode / Security
person_model_path = get_resource_path('models/person_detection_model/best.onnx')
if not os.path.exists(person_model_path):
    person_model_path = None # Fallback to PPE model for person detection if specialized model doesn't exist

# ✅ NEW: Specialized fire/smoke detection model (Updated to use newly trained model)
fire_model_path = get_resource_path('models/fire_and_smoke_model/best.onnx')
if not os.path.exists(fire_model_path):
    fire_model_path = None

# ✅ NEW: Specialized license plate detection model
plate_model_path = get_resource_path('models/plate_detection_model/best.onnx')
if not os.path.exists(plate_model_path):
    plate_model_path = None

# ✅ NEW: Specialized OCR recognition model
ocr_model_path = get_resource_path('models/ocr_model/best.onnx')
if not os.path.exists(ocr_model_path):
    ocr_model_path = None

# ✅ NEW: Specialized pose detection model
pose_model_path = get_resource_path('models/pose_detection_model/best.onnx')
if not os.path.exists(pose_model_path):
    pose_model_path = get_resource_path('yolov8n-pose.onnx')

# Check if custom model exists
if not os.path.exists(ppe_model_path):
    print(f"⚠️  Custom PPE model not found: {ppe_model_path}")
    print(f"📥 Using default YOLOv8n model (will auto-download)...")
    ppe_model_path = get_resource_path('yolov8n.onnx')  # Fallback to default

# Force use of Optimized PPE Detection
from optimized_ppe_detection import OptimizedPPEDetectionSystem, OptimizedIntegratedSystem
ppe_system = OptimizedPPEDetectionSystem(
    model_path=ppe_model_path,
    confidence_threshold=float(os.getenv('PPE_CONFIDENCE_THRESHOLD', '0.55')),  # Base threshold
    db=db, 
    use_cuda=True,
    person_model_path=person_model_path,
    fire_model_path=fire_model_path,
    plate_model_path=plate_model_path,  # ✅ NEW
    ocr_model_path=ocr_model_path,      # ✅ NEW
    pose_model_path=pose_model_path     # ✅ NEW
)
print(f"✅ OPTIMIZED PPE Detection initialized:")
print(f"   PPE Model: {ppe_model_path}")
if person_model_path:
    print(f"   Person Model: {person_model_path}")
if fire_model_path:
    print(f"   Fire Model: {fire_model_path}")
if plate_model_path:
    print(f"   Plate Model: {plate_model_path}")
if ocr_model_path:
    print(f"   OCR Model: {ocr_model_path}")
if pose_model_path:
    print(f"   Pose Model: {pose_model_path}")
print(f"   Base confidence: {ppe_system.base_confidence_threshold}")
print(f"   Features: Multi-tier confidence, Temporal tracking, Priority-based reporting")

# Use optimized integrated system
IntegratedSystemClass = OptimizedIntegratedSystem

# Initialize alert engine
alert_engine = AlertEngine(db)

# Create integrated system (using selected class)
integrated_system = IntegratedSystemClass(face_system, ppe_system, db, alert_engine)


# =============================
# Camera Detection Manager
# =============================
camera_manager = CameraDetectionManager()

inference_controller = InferenceController(
    camera_manager=camera_manager,
    integrated_system=integrated_system,
)
inference_controller.start()


# ============================================================================
# Background Heartbeat Task
# ============================================================================

def camera_heartbeat_task():
    """Background task to sync camera health to database"""
    print("💓 Camera heartbeat task started")
    while True:
        try:
            # Sync health for all cameras
            health_map = camera_manager.get_stream_health()
            for camera_id, health in health_map.items():
                is_online = health.get("status") == "healthy"
                db.update_camera_heartbeat(camera_id, is_online=is_online)
            
            # Sleep for 10 seconds between updates
            time.sleep(10)
        except Exception as e:
            print(f"⚠️ Heartbeat task error: {e}")
            time.sleep(10)

# Start heartbeat thread
heartbeat_thread = threading.Thread(target=camera_heartbeat_task, daemon=True)
heartbeat_thread.start()


# def start_active_cameras():
#     cameras = db.get_all_cameras(status="active")
#     for cam in cameras:
#         if cam.get("rtsp_url"):
#             print(f"[CameraManager] Starting {cam['camera_id']}")
#             camera_manager.start_detection(
#                 cam["camera_id"],
#                 cam["rtsp_url"]
#             )

# Active video streams
active_streams = {}
stream_lock = Lock()

def start_active_cameras():
    """Start all active cameras (RTSP or USB)"""
    cameras = db.get_all_cameras(status="active")
    
    for cam in cameras:
        camera_id = cam['camera_id']
        
        # Get video source (RTSP URL or USB index)
        source = cam.get("rtsp_url")
        if not source:
            source = cam.get("stream_index", 0)
        
        if source is not None:
            # ✅ For local file paths, check existence before starting
            import os as _os
            source_str = str(source)
            if _os.path.isabs(source_str) and not _os.path.isfile(source_str):
                print(f"[CameraManager] ⚠️ Skipping {camera_id} - file not found: {source_str}")
                continue

            print(f"[CameraManager] Starting {camera_id} (source: {source})")
            camera_manager.start_detection(
                camera_id, 
                source_str, 
                target_fps=cam.get('fps')
            )
            with stream_lock:
                active_streams[camera_id] = True
        else:
            print(f"[CameraManager] ⚠️ Skipping {camera_id} - no video source configured")



start_active_cameras()


# ============================================================================
# Utility Functions
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def decode_base64_image(base64_string):
    """Decode base64 image to numpy array"""
    try:
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None


def encode_image_base64(image):
    """Encode numpy image to base64"""
    try:
        success, buffer = cv2.imencode('.jpg', image)
        if success:
            return base64.b64encode(buffer).decode('utf-8')
        return None
    except Exception as e:
        return None


def require_admin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        role = request.headers.get("X-User-Role", "user")
        user = request.headers.get("X-User", "unknown")

        if role.lower() != "admin":
            return jsonify({
                "success": False,
                "error": "Admin access required"
            }), 403

        request.current_user = user
        request.current_role = role
        return fn(*args, **kwargs)

    return wrapper

def _json_ok(data=None, message=None):
    resp = {"success": True}
    if data is not None:
        resp["data"] = data
    if message:
        resp["message"] = message
    return jsonify(resp)

@app.route('/api/login', methods=['POST'])
def login():
    """Admin login endpoint"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required"}), 400
        
    user = db.verify_admin(username, password)
    if user:
        return jsonify({
            "success": True, 
            "message": "Login successful",
            "user": user
        })
    else:
        return jsonify({"success": False, "error": "Invalid credentials"}), 401


@app.route('/api/auth/reset-password', methods=['POST'])
def reset_password_self():
    """Allow a user to reset their password by verifying their registered Full Name."""
    data = request.json or {}
    username = data.get('username', '').strip()
    full_name_check = data.get('full_name', '').strip()
    new_password = data.get('new_password', '')

    if not username or not full_name_check or not new_password:
        return jsonify({"success": False, "error": "Username, Full Name and new password are required"}), 400

    if len(new_password) < 4:
        return jsonify({"success": False, "error": "New password must be at least 4 characters"}), 400

    # Verify user exists and Full Name matches exactly (case-sensitive)
    user = db.users.find_one({"username": username, "full_name": full_name_check})
    if not user:
        return jsonify({"success": False, "error": "Invalid username or verification name. Make sure full name matches your profile."}), 401

    # Update the password
    result = db.users.update_one(
        {"username": username},
        {"$set": {"password": new_password}}
    )

    if result.matched_count > 0:
        return jsonify({"success": True, "message": "Password updated successfully"})
    return jsonify({"success": False, "error": "User not found"}), 404



@app.route('/api/register', methods=['POST'])
def register():
    """User/Company registration endpoint"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    full_name = data.get('full_name')
    company_name = data.get('company_name')
    email = data.get('email')
    
    if not username or not password or not full_name:
        return jsonify({"success": False, "error": "Username, password and full name are required"}), 400
        
    success, message = db.register_user(
        username=username,
        password=password,
        full_name=full_name,
        company_name=company_name,
        email=email
    )
    
    if success:
        return jsonify({
            "success": True,
            "message": message
        }), 201
    else:
        return jsonify({
            "success": False,
            "error": message
        }), 400

# ============================================================================
# API Endpoints - System Status
# ============================================================================
@app.route("/api/stream/<camera_id>")
def stream_camera(camera_id):
    """✅ OPTIMIZED: Annotated or Raw MJPEG stream with improved performance"""
    # Force raw stream for certain views (e.g. OCR page) to ensure max real-time feel
    is_raw = request.args.get('raw', 'false').lower() == 'true'
    
    def generate():
        last_frame_count = -1
        frame_interval = 1.0 / 60.0  # limit to 60 FPS
        last_send_time = 0
        
        while True:
            try:
                current_time = time.time()
                
                if is_raw:
                    frame, current_count = inference_controller.get_latest_raw_frame_with_count(camera_id)
                else:
                    frame, current_count = inference_controller.get_latest_frame_with_count(camera_id)

                if frame is None:
                    time.sleep(0.016)
                    continue
                
                # Check if it's a new frame and if interval passed
                if current_count == last_frame_count:
                    time.sleep(0.005)
                    continue
                
                if current_time - last_send_time < frame_interval:
                    time.sleep(0.002)
                    continue

                # ✅ OPTIMIZED: Faster JPEG encoding
                ok, jpeg = cv2.imencode(
                    ".jpg", 
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 75,
                     int(cv2.IMWRITE_JPEG_OPTIMIZE), 0] # 0 = faster
                )
                
                if not ok:
                    continue

                jpg = jpeg.tobytes()
                
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                    + jpg
                    + b"\r\n"
                )
                
                last_frame_count = current_count
                last_send_time = current_time

            except GeneratorExit:
                break
            except Exception as e:
                print(f"[STREAM] {camera_id} error:", e)
                time.sleep(0.1)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'keep-alive'
        }
    )


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',
        'features': {
            'face_recognition': True,
            'ppe_detection': True,
            'rtsp_streaming': True,
            'spatial_verification': True
        }
    })


@app.route('/api/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics"""
    try:
        stats = db.get_database_stats()
        
        # Add PPE violation stats
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
        
        ppe_violations_today = db.recognition_logs.count_documents({
            'log_type': 'ppe_violation',
            'timestamp': {'$gte': today_start}
        })
        
        unknown_violations_today = db.recognition_logs.count_documents({
            'log_type': 'ppe_violation',
            'is_unknown_person': True,
            'timestamp': {'$gte': today_start}
        })
        
        stats['ppe_violations_today'] = ppe_violations_today
        stats['unknown_violations_today'] = unknown_violations_today
        stats['active_streams'] = len(active_streams)
        
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API Endpoints - PPE Detection & Analysis
# ============================================================================

@app.route('/api/ppe/detect', methods=['POST'])
def detect_ppe():
    """
    Detect PPE in uploaded image
    Returns PPE items detected with spatial verification
    """
    try:
        if 'image' not in request.files and 'image_base64' not in request.form:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Get image
        if 'image' in request.files:
            file = request.files['image']
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type'
                }), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = cv2.imread(filepath)
            os.remove(filepath)
        else:
            img = decode_base64_image(request.form['image_base64'])
        
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400
        
        # Detect PPE
        ppe_detections = ppe_system.detect_ppe(img)
        
        # Format results
        results = []
        for detection in ppe_detections:
            results.append({
                'category': detection['category'],
                'class_name': detection['class_name'],
                'confidence': float(detection['confidence']),
                'bounding_box': detection['bbox'],
                'center': detection['center']
            })
        
        return jsonify({
            'success': True,
            'ppe_items_detected': len(results),
            'detections': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ppe/analyze', methods=['POST'])
def analyze_ppe_compliance():
    """
    Analyze PPE compliance in uploaded image
    Combines face recognition + PPE detection
    """
    try:
        if 'image' not in request.files and 'image_base64' not in request.form:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        camera_id = request.form.get('camera_id', 'API_UPLOAD')
        
        # Get image
        if 'image' in request.files:
            file = request.files['image']
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type'
                }), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = cv2.imread(filepath)
            os.remove(filepath)
        else:
            img = decode_base64_image(request.form['image_base64'])
        
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400
        
        # Process frame with integrated system
        results = integrated_system.process_single_frame(img, camera_id)

        if not results:
            return jsonify({"success": False, "error": "No results available"}), 404

        
        # Format response
        compliance_results = []
        for result in results['compliance_results']:
            compliance_results.append({
                'person_id': result['person_id'],
                'person_name': result['person_name'],
                'role': result['role'],
                'is_unknown': result['is_unknown'],
                'face_confidence': float(result['face_confidence']),
                'compliance': {
                    'is_compliant': result['compliance']['is_compliant'],
                    'compliance_percentage': float(result['compliance']['compliance_percentage']),
                    'required_ppe': result['compliance']['required_ppe'],
                    'wearing_ppe': result['compliance']['wearing_ppe'],
                    'missing_ppe': result['compliance']['missing_ppe']
                },
                'is_violation': result['is_violation'],
                'bounding_box': {
                    'x1': int(result['face_bbox'][0]),
                    'y1': int(result['face_bbox'][1]),
                    'x2': int(result['face_bbox'][2]),
                    'y2': int(result['face_bbox'][3])
                }
            })
        
        # Optionally return annotated image
        include_image = request.form.get('include_image', 'false').lower() == 'true'
        annotated_image = None
        
        if include_image:
            annotated_frame = integrated_system.draw_results(img, results)
            annotated_image = f'data:image/jpeg;base64,{encode_image_base64(annotated_frame)}'
        
        response = {
            'success': True,
            'faces_detected': len(results['face_results']),
            'ppe_items_detected': len(results['ppe_detections']),
            'compliance_results': compliance_results,
            'processing_time_ms': float(results['processing_time_ms']),
            'timestamp': results['timestamp'].isoformat()
        }
        
        if annotated_image:
            response['annotated_image'] = annotated_image
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API Endpoints - PPE Violations
# ============================================================================
@app.route('/api/ppe/violations', methods=['GET'])
def get_ppe_violations():
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 20))
    search = request.args.get('search', '').strip()
    role = request.args.get('role', '').strip()
    start_date = request.args.get('start_date', '').strip()
    end_date = request.args.get('end_date', '').strip()
    camera_id = request.args.get('camera_id', '').strip()

    skip = (page - 1) * page_size

    # ✅ UPDATED: Support filtering by log_type
    log_type = request.args.get('log_type', '').strip()
    if log_type:
        query = {'log_type': log_type}
    else:
        # Default: show all violation types including fire and fall
        query = {'log_type': {'$in': ['ppe_violation', 'zone_violation', 'night_mode_violation', 'fire_violation', 'fall_violation']}}

    # Camera filter
    if camera_id:
        query['camera_id'] = camera_id

    # Role filter
    if role:
        if role.lower() == "worker":
            # virtual role = everyone except visitor
            query['role'] = {"$ne": "visitor"}
        else:
            query['role'] = role.lower()


    # Date range filter
    if start_date or end_date:
        query['timestamp'] = {}
        if start_date:
            try:
                # Expect ISO format YYYY-MM-DD
                dt_start = datetime.fromisoformat(start_date).replace(hour=0, minute=0, second=0)
                query['timestamp']['$gte'] = dt_start
            except ValueError:
                pass
        if end_date:
            try:
                dt_end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59)
                query['timestamp']['$lte'] = dt_end
            except ValueError:
                pass

    # Search filter
    if search:
        query['$or'] = [
            {'person_name': {'$regex': search, '$options': 'i'}},
            {'person_id': {'$regex': search, '$options': 'i'}},
            {'track_id': {'$regex': search, '$options': 'i'}},
            {'missing_ppe': {'$regex': search, '$options': 'i'}},
            {'camera_id': {'$regex': search, '$options': 'i'}}
        ]

    # Define priority for violation types (higher number = higher priority)
    violation_priority = {
        'fire_violation': 1000,      # 🔥 HIGHEST PRIORITY
        'fall_violation': 500,       # 🆘 HIGH PRIORITY
        'night_mode_violation': 100,
        'zone_violation': 10,
        'ppe_violation': 1
    }
    
    cursor = (
        db.recognition_logs
        .find(query)
        .sort('timestamp', -1)
        .skip(skip)
        .limit(page_size)
    )

    data = []
    for v in cursor:
        v['_id'] = str(v['_id'])
        v['timestamp'] = v['timestamp'].isoformat()
        data.append(v)
    
    # ✅ Sort by priority FIRST (fire_violation highest), then by timestamp
    data.sort(key=lambda x: (-violation_priority.get(x.get('log_type', ''), 0), 
                             -int(int(x.get('timestamp', '').replace(':', '').replace('-', '').replace('T', '').replace('.', '').replace('Z', ''), 36) if x.get('timestamp') else 0)))
    
    # Simpler approach: just use log_type priority + timestamp in ascending order by index after sorting
    def sort_key(item):
        log_type = item.get('log_type', '')
        priority = violation_priority.get(log_type, 0)
        # Return tuple: higher priority first, then newer timestamp first
        # We need to convert timestamp to sortable format
        try:
            ts = datetime.fromisoformat(item.get('timestamp', '').replace('Z', '+00:00'))
            ts_val = ts.timestamp()
        except:
            ts_val = 0
        return (-priority, -ts_val)  # Negative for descending order
    
    data.sort(key=sort_key)

    total = db.recognition_logs.count_documents(query)

    return jsonify({
        'success': True,
        'data': data,
        'page': page,
        'page_size': page_size,
        'total': total,
        'total_pages': (total + page_size - 1) // page_size
    })



@app.route('/api/ppe/violations/summary', methods=['GET'])
def get_violations_summary():
    """Get summary of PPE violations"""
    try:
        days = int(request.args.get('days', 7))
        start_date = datetime.utcnow() - timedelta(days=days)
        
        pipeline = [
            {
                '$match': {
                    'log_type': 'ppe_violation',
                    'timestamp': {'$gte': start_date}
                }
            },
            {
                '$group': {
                    '_id': {
                        'date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$timestamp'}},
                        'camera_id': '$camera_id'
                    },
                    'total_violations': {'$sum': 1},
                    'unknown_violations': {
                        '$sum': {'$cond': ['$is_unknown_person', 1, 0]}
                    }
                }
            },
            {'$sort': {'_id.date': -1}}
        ]
        
        summary = list(db.recognition_logs.aggregate(pipeline))
        
        # Also get violations by PPE type
        ppe_type_pipeline = [
            {
                '$match': {
                    'log_type': 'ppe_violation',
                    'timestamp': {'$gte': start_date}
                }
            },
            {'$unwind': '$missing_ppe'},
            {
                '$group': {
                    '_id': '$missing_ppe',
                    'count': {'$sum': 1}
                }
            },
            {'$sort': {'count': -1}}
        ]
        
        ppe_types = list(db.recognition_logs.aggregate(ppe_type_pipeline))
        
        return jsonify({
            'success': True,
            'period_days': days,
            'daily_summary': summary,
            'violations_by_ppe_type': ppe_types
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API Endpoints - Person Management (Enhanced)
# ============================================================================

@app.route('/api/persons', methods=['GET'])
def get_all_persons():
    """Get all persons"""
    try:
        # Default to all if not specified, previous default was 'active'
        status = request.args.get('status') 
        if status == 'all': status = None
        search = request.args.get('search')
        role = request.args.get('role')
        
        persons = db.get_all_persons(status=status, search=search, role=role)
        
        for person in persons:
            person['_id'] = str(person['_id'])
            person.pop('embeddings', None)
            
            # Add PPE requirements for role
            role = person.get('role', 'default')
            person['ppe_requirements'] = ppe_system.role_ppe_requirements.get(
                role.lower(),
                ppe_system.role_ppe_requirements.get("default", [])
            )

        
        return jsonify({
            'success': True,
            'data': persons,
            'count': len(persons)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/persons/<person_id>', methods=['GET'])
def get_person(person_id):
    """Get person by ID with PPE compliance stats"""
    try:
        person = db.get_person(person_id)

        if not person:
            return jsonify({
                'success': False,
                'error': 'Person not found'
            }), 404

        # -----------------------------
        # Convert Mongo _id
        # -----------------------------
        person['_id'] = str(person['_id'])

        # -----------------------------
        # Stats (handled in DB layer)
        # -----------------------------
        person['stats'] = db.get_person_stats(person_id, days=30)

        # -----------------------------
        # PPE requirements
        # -----------------------------
        role = person.get('role', 'default')
        person['ppe_requirements'] = ppe_system.role_ppe_requirements.get(
            role.lower(),
            ppe_system.role_ppe_requirements.get("default", [])
        )

        # -----------------------------
        # Recent PPE violations
        # -----------------------------
        violations = list(
            db.recognition_logs.find({
                'log_type': 'ppe_violation',
                'person_id': person_id
            })
            .sort('timestamp', -1)
            .limit(10)
        )

        for v in violations:
            v['_id'] = str(v['_id'])
            v['timestamp'] = v['timestamp'].isoformat()

        person['recent_violations'] = violations

        # -----------------------------
        # CLEAN embeddings safely
        # -----------------------------
        clean_embeddings = []

        for emb in person.get("embeddings", []):
            emb.pop("vector", None)

            img_id = emb.get("image_id")
            if isinstance(img_id, ObjectId):
                emb["image_id"] = str(img_id)

            clean_embeddings.append(emb)

        person["embeddings"] = clean_embeddings

        # -----------------------------
        # FINAL RESPONSE
        # -----------------------------
        return jsonify({
            'success': True,
            'data': person
        })

    except Exception as e:
        print(f"❌ get_person error ({person_id}):", e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# @app.route('/api/persons/register', methods=['POST'])
# def register_person():
#     """Register a new person with role-based PPE requirements"""
#     try:
#         if 'image' not in request.files:
#             return jsonify({
#                 'success': False,
#                 'error': 'No image provided'
#             }), 400
        
#         file = request.files['image']
        
#         if file.filename == '':
#             return jsonify({
#                 'success': False,
#                 'error': 'No image selected'
#             }), 400
        
#         if not allowed_file(file.filename):
#             return jsonify({
#                 'success': False,
#                 'error': 'Invalid file type'
#             }), 400
        
#         person_id = request.form.get('person_id')
#         name = request.form.get('name')
#         role = request.form.get('role', 'employee')
        
#         if not person_id or not name:
#             return jsonify({
#                 'success': False,
#                 'error': 'person_id and name are required'
#             }), 400
        
#         # Save uploaded file
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Register person
#         success = face_system.register_person_from_image(
#             image_path=filepath,
#             person_id=person_id,
#             name=name,
#             email=request.form.get('email', ''),
#             phone=request.form.get('phone', ''),
#             role=role,
#             department=request.form.get('department', ''),
#             tags=request.form.getlist('tags'),
#             access_level=int(request.form.get('access_level', 1)),
#             registered_by=request.form.get('registered_by', 'api')
#         )
        
#         os.remove(filepath)
        
#         if success:
#             ppe_requirements = ppe_system.role_ppe_requirements.get(
#                 role.lower(),
#                 ppe_system.role_ppe_requirements.get("default", [])
#             )
            
#             return jsonify({
#                 'success': True,
#                 'message': f'Person {name} registered successfully',
#                 'person_id': person_id,
#                 'role': role,
#                 'ppe_requirements': ppe_requirements
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Failed to register person'
#             }), 500
    
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500

@app.route('/api/persons/register-folder', methods=['POST'])
def register_person_from_folder_api():
    """
    Register a person using multiple uploaded face images
    (UI equivalent of register_person_from_folder)
    """
    try:
        # -------------------------
        # Validate inputs
        # -------------------------
        person_id = request.form.get('person_id')
        name = request.form.get('name')
        role = request.form.get('role', 'employee')
        department = request.form.get('department', '')

        if not person_id or not name:
            return jsonify({
                "success": False,
                "error": "person_id and name are required"
            }), 400

        if 'images' not in request.files:
            return jsonify({
                "success": False,
                "error": "No images uploaded"
            }), 400

        files = request.files.getlist('images')

        if len(files) < 3:
            return jsonify({
                "success": False,
                "error": "At least 3 face images are required"
            }), 400

        # -------------------------
        # Save images temporarily
        # -------------------------
        temp_dir = Path(app.config['UPLOAD_FOLDER']) / f"reg_{person_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        saved_images = []

        for f in files:
            if not allowed_file(f.filename):
                continue

            filename = secure_filename(f.filename)
            path = temp_dir / filename
            f.save(path)
            saved_images.append(path)

        if not saved_images:
            return jsonify({
                "success": False,
                "error": "No valid images found"
            }), 400

        # -------------------------
        # 🔥 PAUSE inference (CRITICAL)
        # -------------------------
        print("⏸ Pausing inference for registration...")
        inference_controller.pause()

        try:
            success = face_system.register_person_from_folder(
                folder_path=str(temp_dir),
                person_id=person_id,
                name=name,
                role=role,
                department=department,
                registered_by="ui"
            )
        finally:
            # -------------------------
            # ▶ RESUME inference (ALWAYS)
            # -------------------------
            inference_controller.resume()
            print("▶ Inference resumed")


        # Cleanup temp files
        import shutil

        # Cleanup temp directory safely (Windows-safe)
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"⚠️ Temp cleanup warning: {e}")


        if not success:
            return jsonify({
                "success": False,
                "error": "Face registration failed (no valid faces detected)"
            }), 500

        # ✅ Optimized: Embeddings are now reloaded internally by the registration function
        # print("🔄 Reloading embeddings into cache...")
        # try:
        #     face_system.reload_embeddings()
        #     print("✅ Embeddings reloaded successfully")
        # except Exception as e:
        #     print(f"⚠️ Warning: Error reloading embeddings: {e}")
        #     # Don't fail the registration if reload fails


        return jsonify({
            "success": True,
            "message": f"Person {name} registered successfully",
            "person_id": person_id,
            "images_used": len(saved_images),
            "role": role
        })


    except Exception as e:
        # Make sure to resume inference even if there's an error
        try:
            inference_controller.resume()
        except:
            pass
        print("❌ Register-folder error:", e)
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/persons/<person_id>', methods=['PUT'])
def update_person(person_id):
    """Update person information"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        data.pop('person_id', None)
        data.pop('embeddings', None)
        data.pop('_id', None)
        
        success = db.update_person(person_id, data)
        
        if success:
            # If role changed, return new PPE requirements
            new_role = data.get('role')
            response = {
                'success': True,
                'message': 'Person updated successfully'
            }
            
            if new_role:
                response['ppe_requirements'] = ppe_system.role_ppe_requirements.get(
                    new_role.lower(),
                    ppe_system.role_ppe_requirements.get("default", [])
                )

            
            return jsonify(response)
        else:
            return jsonify({
                'success': False,
                'error': 'Person not found or no changes made'
            }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/persons/<person_id>', methods=['DELETE'])
def delete_person(person_id):
    """Delete a person"""
    try:
        success = db.delete_person(person_id)
        
        if success:
            face_system.reload_embeddings()
            
            return jsonify({
                'success': True,
                'message': 'Person deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Person not found'
            }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API Endpoints - Camera Management
# ============================================================================

@app.route('/api/cameras', methods=['GET'])
def get_all_cameras():
    """Get all cameras"""
    try:
        status = request.args.get('status', 'active')
        cameras = db.get_all_cameras(status=status)
        
        for camera in cameras:
            camera['_id'] = str(camera['_id'])
            camera['is_streaming'] = camera['camera_id'] in active_streams
        
        return jsonify({
            'success': True,
            'data': cameras,
            'count': len(cameras)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cameras/<camera_id>', methods=['GET'])
def get_camera(camera_id):
    """Get camera by ID"""
    try:
        camera = db.get_camera(camera_id)
        
        if not camera:
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
        
        camera['_id'] = str(camera['_id'])
        camera['is_streaming'] = camera_id in active_streams
        
        stats = db.get_camera_stats(camera_id, days=7)
        camera['stats'] = stats
        
        return jsonify({
            'success': True,
            'data': camera
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cameras/register', methods=['POST'])
def register_camera():
    """Register a new camera"""
    try:
        data = request.json
        
        required_fields = ['camera_id', 'name']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'{field} is required'
                }), 400
        
        success = db.register_camera(**data)
        
        if success:
            # ✅ Automatically start detection after registration
            try:
                source = data.get("rtsp_url")
                if not source:
                    source = data.get("stream_index", 0)
                
                if source is not None:
                    print(f"[CameraManager] Auto-starting new camera: {data['camera_id']}")
                    camera_manager.start_detection(
                        data['camera_id'], 
                        str(source),
                        target_fps=data.get('fps')
                    )
                    with stream_lock:
                        active_streams[data['camera_id']] = True
            except Exception as e:
                print(f"⚠️ Failed to auto-start camera {data['camera_id']}: {e}")

            log_audit_event(f"Registered new camera: {data['camera_id']} ({data.get('name', 'Unnamed')})")
            return jsonify({
                'success': True,
                'message': 'Camera registered successfully',
                'camera_id': data['camera_id']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to register camera'
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/cameras/<camera_id>', methods=['DELETE'])
def remove_camera(camera_id):
    try:
        # 1️⃣ Check camera exists
        camera = db.get_camera(camera_id)
        if not camera:
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404

        # 2️⃣ Stop detection / streaming
        try:
            camera_manager.stop_detection(camera_id)
        except Exception:
            pass

        # 3️⃣ Remove from active streams
        with stream_lock:
            active_streams.pop(camera_id, None)

        # 4️⃣ HARD DELETE FROM DB
        result = db.cameras.delete_one({'camera_id': camera_id})

        if result.deleted_count == 0:
            return jsonify({
                'success': False,
                'error': 'Failed to delete camera from database'
            }), 500

        log_audit_event(f"Removed camera: {camera_id}")
        return jsonify({
            'success': True,
            'message': f'Camera {camera_id} deleted permanently'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cameras/<camera_id>', methods=['PUT'])
def update_camera_api(camera_id):
    """Update camera configuration"""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Check if status is being updated to 'active' or 'inactive'
        if 'status' in data:
            new_status = data['status']
            if new_status == 'active':
                # Fetch current camera data and merge with new data to get correct connection source
                camera = db.get_camera(camera_id)
                if camera:
                    # Use new source if provided in data, else fallback to existing
                    source = (data.get("rtsp_url") or data.get("stream_index") or 
                             camera.get("rtsp_url") or camera.get("stream_index", 0))
                    
                    # Ensure it's not already running before starting
                    if camera_id not in active_streams:
                        print(f"[API] Starting camera {camera_id} with source: {source}")
                        camera_manager.start_detection(camera_id, str(source))
                        with stream_lock:
                            active_streams[camera_id] = True
            elif new_status == 'inactive':
                # Stop camera
                camera_manager.stop_detection(camera_id)
                with stream_lock:
                    active_streams.pop(camera_id, None)

        success = db.update_camera(camera_id, data)
        
        if success:
            log_audit_event(f"Updated camera settings: {camera_id}")
            return jsonify({
                'success': True,
                'message': f'Camera {camera_id} updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API Endpoints - Zones
# ============================================================================

@app.route('/api/zones', methods=['GET'])
def get_zones():
    """Get all zones"""
    try:
        zones = db.get_all_zones()
        
        # Convert ObjectId to string
        for zone in zones:
            zone['_id'] = str(zone['_id'])
            if 'created_at' in zone:
                zone['created_at'] = zone['created_at'].isoformat()
            if 'updated_at' in zone:
                zone['updated_at'] = zone['updated_at'].isoformat()
        
        return jsonify({
            'success': True,
            'data': zones
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/zones', methods=['POST'])
def create_zone():
    """Create a new zone with optional role restrictions"""
    try:
        data = request.json
        
        zone_name = data.get('zone_name', '').strip()
        description = data.get('description', '').strip()
        allowed_roles = data.get('allowed_roles', [])  # List of role names
        
        if not zone_name:
            return jsonify({
                'success': False,
                'error': 'zone_name is required'
            }), 400
        
        # Validate allowed_roles is a list
        if not isinstance(allowed_roles, list):
            return jsonify({
                'success': False,
                'error': 'allowed_roles must be an array'
            }), 400
        
        person_limit = data.get('person_limit')
        
        success = db.create_zone(zone_name, description, allowed_roles, person_limit)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Zone "{zone_name}" created successfully',
                'zone_name': zone_name,
                'allowed_roles': allowed_roles
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Zone "{zone_name}" already exists'
            }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/zones/<zone_name>', methods=['PUT'])
def update_zone(zone_name):
    """Update zone settings including allowed roles"""
    try:
        data = request.json
        updates = {}
        
        # Fields we explicitly check but allow others
        editable_fields = ['description', 'allowed_roles', 'person_limit', 'status']
        
        for field in editable_fields:
            if field in data:
                if field == 'allowed_roles' and not isinstance(data[field], list):
                    continue # Skip invalid list
                updates[field] = data[field]
        
        # Also allow any other fields passed
        for k, v in data.items():
            if k not in updates and k != 'zone_name':
                updates[k] = v
        
        if not updates:
            return jsonify({
                'success': False,
                'error': 'No valid fields to update'
            }), 400
        
        success = db.update_zone(zone_name, updates)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Zone "{zone_name}" updated successfully',
                'updates': updates
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Zone "{zone_name}" not found'
            }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/zones/<zone_name>', methods=['DELETE'])
def delete_zone(zone_name):
    """Delete a zone"""
    try:
        # Check if any cameras are using this zone
        cameras_in_zone = list(db.cameras.find({"zone": zone_name}))
        
        if cameras_in_zone:
            # Update cameras to default zone
            db.cameras.update_many(
                {"zone": zone_name},
                {"$set": {"zone": "default"}}
            )
            print(f"📌 Updated {len(cameras_in_zone)} cameras from zone '{zone_name}' to 'default'")
        
        # Delete the zone
        success = db.delete_zone(zone_name)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Zone "{zone_name}" deleted successfully',
                'cameras_updated': len(cameras_in_zone)
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Zone "{zone_name}" not found'
            }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API Endpoints - Recognition Logs
# ============================================================================
@app.route('/api/logs', methods=['GET'])
def get_recognition_logs():
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('limit', 50))

    skip = (page - 1) * page_size

    query = {}

    camera_id = request.args.get('camera_id')
    person_id = request.args.get('person_id')

    if camera_id:
        query['camera_id'] = camera_id
    if person_id:
        query['person_id'] = person_id

    # Date filtering
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    
    if start_date_str or end_date_str:
        query['timestamp'] = {}
        if start_date_str:
            try:
                # Expecting YYYY-MM-DD
                start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
                query['timestamp']['$gte'] = start_dt
            except ValueError:
                pass
        
        if end_date_str:
            try:
                # Expecting YYYY-MM-DD, set to end of day
                end_dt = datetime.strptime(end_date_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
                query['timestamp']['$lte'] = end_dt
            except ValueError:
                pass

    cursor = (
        db.recognition_logs
        .find(query)
        .sort('timestamp', -1)
        .skip(skip)
        .limit(page_size)
    )

    data = []
    for log in cursor:
        log['_id'] = str(log['_id'])
        log['timestamp'] = log['timestamp'].isoformat()
        data.append(log)

    total = db.recognition_logs.count_documents(query)

    return jsonify({
        'success': True,
        'data': data,
        'total': total,
        'page': page,
        'page_size': page_size,
        'total_pages': (total + page_size - 1) // page_size
    })


@app.route('/api/logs/image/<image_id>', methods=['GET'])
def get_log_image(image_id):
    """Get face image from log"""
    try:
        from bson import ObjectId
        
        image = db.get_face_image(ObjectId(image_id))
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Image not found'
            }), 404
        
        base64_image = encode_image_base64(image)
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{base64_image}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API Endpoints - Face Recognition
# ============================================================================

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Recognize face from uploaded image"""
    try:
        if 'image' not in request.files and 'image_base64' not in request.form:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Get image
        if 'image' in request.files:
            file = request.files['image']
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type'
                }), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = cv2.imread(filepath)
            os.remove(filepath)
        else:
            img = decode_base64_image(request.form['image_base64'])
        
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400
        
        # Detect faces
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, probs = face_system.detector.detect(rgb_img)
        
        if boxes is None or len(boxes) == 0:
            return jsonify({
                'success': True,
                'faces_detected': 0,
                'results': []
            })
        
        results = []
        
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = [int(b) for b in box]
            
            padding = 20
            fx1 = max(0, x1 - padding)
            fy1 = max(0, y1 - padding)
            fx2 = min(img.shape[1], x2 + padding)
            fy2 = min(img.shape[0], y2 + padding)
            
            face_img = img[fy1:fy2, fx1:fx2]
            
            if face_img.size == 0:
                continue
            
            # Recognize
            person_id, confidence, distance = face_system.recognize_face(face_img)
            if prob < 0.75:
                continue
            
            result = {
                'bounding_box': {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                },
                'detection_confidence': float(prob),
                'person_id': person_id if person_id != "unknown" else None,
                'confidence': float(confidence),
                'distance': float(distance),
                'recognized': person_id != "unknown"
            }
            
            # Get person info if recognized
            if person_id and person_id != "unknown":
                person = db.get_person(person_id)
                if person:
                    role = person.get('role', 'default')
                    result['person'] = {
                        'name': person['name'],
                        'email': person.get('email'),
                        'role': role,
                        'department': person.get('department'),
                        'ppe_requirements': ppe_system.role_ppe_requirements.get(
                            role.lower(),
                            ppe_system.role_ppe_requirements.get("default", [])
                        )
                    }
            
            results.append(result)
        
        return jsonify({
            'success': True,
            'faces_detected': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/api/cameras/performance")
def camera_performance():
    return jsonify({
        "decode_fps": camera_manager.get_decode_fps(),
        "inference_fps": inference_controller.get_inference_fps()
    })

#system_config
@app.route("/api/settings/ppe", methods=["GET"])
def get_ppe_settings():
    try:
        config = db.system_config.find_one(
            {"config_type": "ppe_rules"},
            {"_id": 0}
        )

        if not config:
            return jsonify({
                "success": False,
                "error": "PPE rules not configured"
            }), 404

        return jsonify({
            "success": True,
            "data": config
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/settings/ppe", methods=["PUT"])
@require_admin
def update_ppe_settings():
    try:
        payload = request.json
        if not payload:
            return jsonify({
                "success": False,
                "error": "Payload is required"
            }), 400

        # 1️⃣ Load existing config
        config = db.system_config.find_one(
            {"config_type": "ppe_rules"}
        ) or {}

        existing_rules = config.get("role_rules", {})
        existing_colors = config.get("role_colors", {})

        # 2️⃣ Merge Rules (DO NOT REPLACE)
        if "role_rules" in payload:
            for role, ppe_list in payload["role_rules"].items():
                existing_rules[role.lower()] = ppe_list

        # 3️⃣ Merge Colors (DO NOT REPLACE)
        if "role_colors" in payload:
            for role, color in payload["role_colors"].items():
                existing_colors[role.lower()] = color

        # 4️⃣ Persist merged config
        update_data = {
            "config_type": "ppe_rules",
            "role_rules": existing_rules,
            "role_colors": existing_colors,
            "updated_at": datetime.utcnow(),
            "updated_by": request.current_user if hasattr(request, 'current_user') else 'system'
        }

        db.system_config.update_one(
            {"config_type": "ppe_rules"},
            {"$set": update_data},
            upsert=True
        )

        # 5️⃣ Hot reload PPE rules
        ppe_system.reload_ppe_rules()
        log_audit_event("Updated PPE compliance rules")

        return jsonify({
            "success": True,
            "message": "PPE settings updated successfully",
            "data": update_data
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/settings/ppe/rename", methods=["POST"])
@require_admin
def rename_ppe_role():
    """Rename a PPE role and migrate all workers assigned to it"""
    try:
        data = request.json
        old_name = data.get("old_name", "").lower()
        new_name = data.get("new_name", "").lower()
        new_color = data.get("new_color")

        if not old_name or not new_name:
            return jsonify({
                "success": False, 
                "error": "Both old_name and new_name are required"
            }), 400

        # Title case for display in logs
        old_display = old_name.replace("_", " ").title()
        new_display = new_name.replace("_", " ").title()

        # 1️⃣ Load current config to migrate rules
        config = db.system_config.find_one({"config_type": "ppe_rules"})
        if not config:
            return jsonify({"success": False, "error": "System configuration not found"}), 404

        rules = config.get("role_rules", {})
        colors = config.get("role_colors", {})

        if old_name not in rules:
            return jsonify({"success": False, "error": f"Role '{old_display}' does not exist"}), 404

        if new_name != old_name:
            if new_name in rules:
                return jsonify({"success": False, "error": f"Role '{new_display}' already exists"}), 400
            
            # Migrate Rules and Colors
            rules[new_name] = rules.pop(old_name)
            colors[new_name] = colors.pop(old_name)
        
        # Update color if provided
        if new_color:
            colors[new_name] = new_color

        # Persist updated config
        db.system_config.update_one(
            {"config_type": "ppe_rules"},
            {"$set": {
                "role_rules": rules,
                "role_colors": colors,
                "updated_at": datetime.utcnow(),
                "updated_by": request.current_user
            }}
        )

        # 2️⃣ Migrate Workers (Persons collection)
        # Find all persons with the old role and update to new role
        result = db.persons.update_many(
            {"role": {"$regex": f"^{old_display}$", "$options": "i"}}, # Case-insensitive match for security
            {"$set": {"role": new_display}}
        )
        # Also try exact match for the key if stored as key
        result_key = db.persons.update_many(
            {"role": old_name},
            {"$set": {"role": new_name}}
        )

        total_migrated = result.modified_count + result_key.modified_count
        log_audit_event(f"Renamed PPE role '{old_name}' to '{new_name}' ({total_migrated} workers updated)")
        print(f"🔄 Role Migration: '{old_display}' -> '{new_display}' ({total_migrated} workers updated)")

        # 3️⃣ Hot reload rules
        ppe_system.reload_ppe_rules()

        return jsonify({
            "success": True,
            "message": f"Role '{old_display}' successfully renamed to '{new_display}'",
            "migrated_workers": total_migrated
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Rename failed: {str(e)}"
        }), 500

@app.route("/api/settings/audit", methods=["GET"])
@require_admin
def get_settings_audit_logs():
    """
    Returns audit logs for system settings changes
    """
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 50))
        action = request.args.get("action")
        entity = request.args.get("entity")

        skip = (page - 1) * limit

        logs, total = db.get_audit_logs(
            action=action,
            entity=entity,
            limit=limit,
            skip=skip
        )

        return jsonify({
            "success": True,
            "data": logs,
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": (total + limit - 1) // limit
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def get_ppe_requirements_for_role(role: str):
    return ppe_system.role_ppe_requirements.get(
        role.lower(),
        ppe_system.role_ppe_requirements.get("default", [])
    )


# Add these endpoints to api_server_n.py

AUDIT_LOG_FILE = get_resource_path("audit_sessions.json")

def _migrate_old_log():
    """One-time migration from .log to .json format"""
    old_log = get_resource_path("audit_sessions.log")
    if os.path.exists(old_log) and not os.path.exists(AUDIT_LOG_FILE):
        try:
            print(f"📦 Migrating {old_log} to JSON format...")
            data = []
            with open(old_log, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if " | " in line:
                        parts = line.split(" | ")
                        ts = parts[0].strip("[]")
                        act = parts[1].strip()
                        user = parts[2].strip()
                        data.append({"timestamp": ts, "activity": act, "user": user})
            
            with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            # rename to .old
            os.rename(old_log, old_log + ".old")
        except Exception as e:
            print(f"⚠️ Migration failed: {e}")

# Call migration once on server start
_migrate_old_log()

def _load_audit_sessions():
    """Load session list from file. Returns a list of session objects."""
    if not os.path.exists(AUDIT_LOG_FILE):
        return []
    try:
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Support legacy flat format (list of {timestamp, activity, user})
        if data and isinstance(data[0], dict) and "activity" in data[0]:
            # Migrate on-the-fly
            return _migrate_flat_to_sessions(data)
        return data
    except Exception:
        return []

def _save_audit_sessions(sessions):
    """Persist session list to file with key order: start_time, user, activities, end_time."""
    try:
        ordered = [
            {
                "start_time": s.get("start_time"),
                "user":       s.get("user"),
                "activities": s.get("activities", []),
                "end_time":   s.get("end_time")
            }
            for s in sessions
        ]
        with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=4)
    except Exception as e:
        print(f"Failed to write audit log: {e}")

def _migrate_flat_to_sessions(entries):
    """Convert old flat event list to grouped session format."""
    sessions = []
    curr = None
    for entry in entries:
        ts, action, user = entry.get("timestamp"), entry.get("activity"), entry.get("user", "Unknown")
        if action == "Session START":
            if curr:
                sessions.append(curr)
            curr = {"start_time": ts, "user": user, "activities": [], "end_time": None}
        elif action == "Session END":
            if curr:
                curr["end_time"] = ts
                sessions.append(curr)
                curr = None
            else:
                sessions.append({"start_time": None, "user": user, "activities": [], "end_time": ts})
        else:
            if curr is None:
                curr = {"start_time": ts, "user": user, "activities": [], "end_time": None}
            curr["activities"].append({"time": ts, "action": action})
    if curr:
        curr["end_time"] = "Still active"
        sessions.append(curr)
    return sessions

def log_audit_event(activity):
    """Append an activity to the CURRENT open session in the grouped JSON file."""
    try:
        company = db.get_company_name()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sessions = _load_audit_sessions()

        # Find the last open session (no end_time or end_time == 'Still active')
        open_sess = next(
            (s for s in reversed(sessions)
             if not s.get("end_time") or s.get("end_time") == "Still active"),
            None
        )

        if open_sess is None:
            # No open session — create one implicitly so the activity is not lost
            open_sess = {"start_time": ts, "user": company, "activities": [], "end_time": None}
            sessions.append(open_sess)

        open_sess["activities"].append({"time": ts, "action": activity})
        _save_audit_sessions(sessions)
    except Exception as e:
        print(f"❌ Failed to write audit log: {e}")

@app.route("/api/audit/session/start", methods=["POST"])
def audit_session_start():
    try:
        company = db.get_company_name()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sessions = _load_audit_sessions()
        # Close any dangling open session first
        for s in reversed(sessions):
            if not s.get("end_time") or s.get("end_time") == "Still active":
                s["end_time"] = ts + " (auto-closed)"
                break
        sessions.append({"start_time": ts, "user": company, "activities": [], "end_time": None})
        _save_audit_sessions(sessions)
    except Exception as e:
        print(f"❌ Session start log error: {e}")
    return jsonify({"success": True})

@app.route("/api/audit/session/end", methods=["POST"])
def audit_session_end():
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sessions = _load_audit_sessions()
        for s in reversed(sessions):
            if not s.get("end_time") or s.get("end_time") == "Still active":
                s["end_time"] = ts
                break
        _save_audit_sessions(sessions)
    except Exception as e:
        print(f"❌ Session end log error: {e}")
    return jsonify({"success": True})


@app.route("/api/settings/audit/sessions", methods=["GET"])
def get_audit_sessions():
    """Return grouped session objects from audit_sessions.json, newest first."""
    try:
        sessions = _load_audit_sessions()
        # Sort newest first by start_time
        sessions_sorted = sorted(
            sessions,
            key=lambda s: s.get("start_time") or "",
            reverse=True
        )
        return jsonify({"success": True, "data": sessions_sorted})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==================================================
# ALERTS CONFIGURATION ENDPOINTS
# ==================================================

@app.route("/api/settings/alerts", methods=["GET"])
def get_alerts_settings():
    """Get alert configuration"""
    try:
        config = db.system_config.find_one(
            {"config_type": "alerts"},
            {"_id": 0}
        )

        if not config:
            # Return default configuration
            default_config = {
                "config_type": "alerts",
                "enable_alerts": True,
                "delivery_mode": "whatsapp_with_sms_fallback",
                "cooldown_seconds": 30,
                "whatsapp_cloud": {
                    "access_token": "",
                    "instance_id": "",
                    "recipient_number": ""
                },
                "sms_twilio": {
                    "account_sid": "",
                    "auth_token": "",
                    "twilio_number": "",
                    "recipient_number": ""
                }
            }
            return jsonify({
                "success": True,
                "data": default_config
            })

        return jsonify({
            "success": True,
            "data": config
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# Add these endpoints to api_server_n.py

@app.route("/api/settings/alerts", methods=["PUT"])
@require_admin
def update_alerts_settings():
    """Update alert configuration with WhatsApp settings"""
    try:
        payload = request.json
        if not payload:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        # Update configuration
        db.system_config.update_one(
            {"config_type": "alerts"},
            {
                "$set": {
                    "config_type": "alerts",
                    "enable_alerts": payload.get("enable_alerts", True),
                    "enable_buzzer": payload.get("enable_buzzer", True),
                    "delivery_mode": payload.get("delivery_mode", "whatsapp_with_sms_fallback"),
                    "cooldown_seconds": payload.get("cooldown_seconds", 30),
                    "whatsapp_cloud": {
                        "access_token": payload.get("whatsapp_cloud", {}).get("access_token", ""),
                        "instance_id": payload.get("whatsapp_cloud", {}).get("instance_id", ""),
                        "recipient_number": payload.get("whatsapp_cloud", {}).get("recipient_number", "")
                    },
                    "sms_twilio": {
                        "account_sid": payload.get("sms_twilio", {}).get("account_sid", ""),
                        "auth_token": payload.get("sms_twilio", {}).get("auth_token", ""),
                        "twilio_number": payload.get("sms_twilio", {}).get("twilio_number", ""),
                        "recipient_number": payload.get("sms_twilio", {}).get("recipient_number", "")
                    },
                    "updated_at": datetime.utcnow(),
                    "updated_by": request.current_user
                }
            },
            upsert=True
        )

        # Reload alert engine configuration
        alert_engine.reload()

        # Log audit
        log_audit_event("Updated WhatsApp/SMS alert configuration")
        db.insert_audit_log({
            "action": "update_alerts",
            "entity": "alerts_config",
            "performed_by": request.current_user,
            "performed_at": datetime.utcnow(),
            "details": {
                "enable_alerts": payload.get("enable_alerts"),
                "delivery_mode": payload.get("delivery_mode"),
                "cooldown": payload.get("cooldown_seconds")
            }
        })

        return jsonify({
            "success": True,
            "message": "Alert settings updated successfully"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/settings/night_mode", methods=["GET"])
def get_night_mode_settings():
    """Get night mode configuration"""
    try:
        config = db.system_config.find_one(
            {"config_type": "night_mode"},
            {"_id": 0}
        )
        if not config:
            # Default
            config = {
                "config_type": "night_mode",
                "enabled": False,
                "automatic": True,
                "manual_on": False,
                "start_time": "20:00",
                "end_time": "06:00",
                "whatsapp_alerts": True
            }
        return jsonify({"success": True, "data": config})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/settings/night_mode/status", methods=["GET"])
def get_night_mode_status():
    """Check if night mode is currently active (auto or manual)"""
    try:
        is_active = db.is_night_mode_active()
        return jsonify({"success": True, "active": is_active})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/settings/night_mode", methods=["PUT"])
@require_admin
def update_night_mode_settings():
    """Update night mode configuration"""
    try:
        payload = request.json
        if not payload:
            return jsonify({"success": False, "error": "No data provided"}), 400

        db.system_config.update_one(
            {"config_type": "night_mode"},
            {
                "$set": {
                    "enabled": payload.get("enabled", False),
                    "automatic": payload.get("automatic", True),
                    "manual_on": payload.get("manual_on", False),
                    "start_time": payload.get("start_time", "20:00"),
                    "end_time": payload.get("end_time", "06:00"),
                    "whatsapp_alerts": payload.get("whatsapp_alerts", True),
                    "updated_at": datetime.utcnow()
                }
            },
            upsert=True
        )

        log_audit_event("Updated Night Mode schedule/settings")
        return jsonify({"success": True, "message": "Night mode settings updated"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/settings/night_mode/toggle", methods=["POST"])
@require_admin
def toggle_night_mode_manual():
    """Manually toggle night mode on/off"""
    try:
        data = request.json
        manual_status = data.get("manual_on", False)
        
        db.system_config.update_one(
            {"config_type": "night_mode"},
            {"$set": {"manual_on": manual_status, "updated_at": datetime.utcnow()}},
            upsert=True
        )
        
        status_text = "enabled" if manual_status else "disabled"
        log_audit_event(f"Manually {status_text} Night Mode")
        return jsonify({"success": True, "message": f"Night mode manually {status_text}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/cameras/<camera_id>/live_status")
def live_status(camera_id):
    return jsonify({
        "success": True,
        "data": inference_controller.get_live_status(camera_id)
    })

@app.route("/api/cameras/live_status_all")
def live_status_all():
    """✅ OPTIMIZED: Return live status for all active cameras in a single request to prevent UI blocking."""
    status_map = {}
    
    with stream_lock:
        active_cams = list(active_streams.keys())
        
    for cam_id in active_cams:
        status_map[cam_id] = inference_controller.get_live_status(cam_id)
        
    return jsonify({
        "success": True,
        "data": status_map
    })


@app.route("/api/cameras/<camera_id>/recent_violations")
def recent_violations(camera_id):
    """
    Returns violations logged in the last 60 seconds for the given camera.
    Used by the Active Detections panel to auto-refresh without needing a manual refresh.
    ✅ PRIORITIZED: Fire violations appear first, followed by other violations
    """
    try:
        seconds = int(request.args.get("seconds", 60))
        since = datetime.utcnow() - timedelta(seconds=seconds)

        query = {
            "camera_id": camera_id,
            "timestamp": {"$gte": since},
            # ✅ FIX: Include fire_smoke_alert so live fire events are captured too
            "log_type": {"$in": ["ppe_violation", "zone_violation", "night_mode_violation",
                                  "fire_violation", "fire_smoke_alert", "fall_violation"]}
        }

        cursor = db.recognition_logs.find(query).sort("timestamp", -1).limit(30)

        results = []
        for doc in cursor:
            results.append({
                "_id": str(doc["_id"]),
                "log_type": doc.get("log_type"),
                "person_id": doc.get("person_id"),
                "person_name": doc.get("person_name"),
                "track_id": doc.get("track_id"),
                "role": doc.get("role"),
                "camera_id": doc.get("camera_id"),
                "missing_ppe": doc.get("missing_ppe", []),
                "is_compliant": False,
                "is_unknown": doc.get("is_unknown_person", False),
                "is_zone_authorized": doc.get("log_type") != "zone_violation",
                # ✅ FIX: Fire/Fall violations store zone as 'location'. Fall back gracefully.
                "zone_name": doc.get("zone") or doc.get("location") or doc.get("zone_name"),
                "fire_category": doc.get("fire_category"),
                "violation_log_id": str(doc["_id"]),
                "snapshot_url": doc.get("snapshot_url"),
                "timestamp": doc["timestamp"].isoformat() if hasattr(doc.get("timestamp"), "isoformat") else str(doc.get("timestamp", "")),
            })
        
        # ✅ Re-sort results by violation type priority (fire/fall first)
        violation_priority = {
            'fire_violation': 1000,      # 🔥 HIGHEST PRIORITY
            'fire_smoke_alert': 1000,    # 🔥 Same as fire_violation
            'fall_violation': 500,       # 🆘 HIGH PRIORITY
            'night_mode_violation': 100,
            'zone_violation': 10,
            'ppe_violation': 1
        }
        
        def sort_key(item):
            log_type = item.get('log_type', '')
            priority = violation_priority.get(log_type, 0)
            # Return tuple: higher priority first, then newer timestamp first
            try:
                ts = datetime.fromisoformat(item.get('timestamp', '').replace('Z', '+00:00'))
                ts_val = ts.timestamp()
            except:
                ts_val = 0
            return (-priority, -ts_val)  # Negative for descending order
        
        results.sort(key=sort_key)

        return jsonify({"success": True, "data": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/emergencies/active")
def get_active_emergencies():
    """
    ✅ NEW: Cross-camera emergency endpoint.
    Returns any active fire/fall/smoke detections across ALL cameras
    for the last `seconds` seconds (default 30s).
    Used by the UI to show an alert banner even when viewing a different camera.
    """
    try:
        seconds = int(request.args.get("seconds", 30))
        since = datetime.utcnow() - timedelta(seconds=seconds)

        query = {
            "timestamp": {"$gte": since},
            "log_type": {"$in": ["fire_violation", "fire_smoke_alert", "fall_violation"]}
        }

        cursor = db.recognition_logs.find(query).sort("timestamp", -1).limit(20)

        results = []
        for doc in cursor:
            cam_id = doc.get("camera_id", "")
            # Fetch friendly camera name from DB if available
            cam_name = cam_id
            try:
                cam_info = db.get_camera(cam_id)
                if cam_info:
                    cam_name = cam_info.get("name") or cam_info.get("camera_name") or cam_id
            except Exception:
                pass

            results.append({
                "_id": str(doc["_id"]),
                "log_type": doc.get("log_type"),
                "camera_id": cam_id,
                "camera_name": cam_name,
                "zone_name": doc.get("zone") or doc.get("location") or doc.get("zone_name") or "Unknown Zone",
                "fire_category": doc.get("fire_category"),
                "snapshot_url": doc.get("snapshot_url"),
                "timestamp": doc["timestamp"].isoformat() if hasattr(doc.get("timestamp"), "isoformat") else str(doc.get("timestamp", "")),
            })

        return jsonify({"success": True, "data": results, "count": len(results)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



# ✅ Global cache for encoded MJPEG frames to avoid redundant work
_mjpeg_cache = {} # (camera_id, w, h) -> (count, jpg_bytes)
_cache_lock = Lock()

@app.route("/api/stream/raw/<camera_id>")
def stream_raw(camera_id):
    """
    🚀 ULTRA-OPTIMIZED: Raw MJPEG stream with maximum performance
    - Response-side caching to prevent redundant encoding
    - Zero latency design with FAST resizing
    """
    target_width = request.args.get('width', type=int)
    target_height = request.args.get('height', type=int)
    cache_key = (camera_id, target_width, target_height)
    
    print(f"🎬 [RAW STREAM] Request: {camera_id} (w={target_width}, h={target_height})")
    
    # Quick sanity check
    with camera_manager.lock:
        if camera_id not in camera_manager.running_flags:
            print(f"⚠️ [RAW STREAM] {camera_id} NOT found in running_flags. Known: {list(camera_manager.running_flags.keys())}")
        if camera_id not in camera_manager.latest_frames:
            print(f"⚠️ [RAW STREAM] {camera_id} HAS NO FRAME in latest_frames cache.")

    def generate():
        last_frame_count = -1
        
        while True:
            try:
                # 1. Check if we have a fresh frame without full lock
                # We still need to know the count to see if we should re-encode
                with camera_manager.lock:
                    current_count = camera_manager.frame_counters.get(camera_id, 0)
                
                if current_count == last_frame_count:
                    time.sleep(0.005)
                    continue

                # 2. Check cache first
                with _cache_lock:
                    cached_entry = _mjpeg_cache.get(cache_key)
                    if cached_entry and cached_entry[0] == current_count:
                        jpg = cached_entry[1]
                        last_frame_count = current_count
                    else:
                        jpg = None

                # 3. If no cache, encode it
                if jpg is None:
                    with camera_manager.lock:
                        frame = camera_manager.latest_frames.get(camera_id)
                    
                    if frame is None:
                        # Log every 100 iterations (aprox 1s)
                        if int(time.time() * 100) % 100 == 0:
                            print(f"⚠️ [RAW STREAM] {camera_id} loop: frame is still None")
                        time.sleep(0.01)
                        continue

                    # Process frame
                    process_frame = frame
                    if target_width or target_height:
                        h, w = frame.shape[:2]
                        # Use INTER_NEAREST for maximum speed
                        if target_width and target_height:
                            process_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
                        elif target_width:
                            scale = target_width / w
                            process_frame = cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_NEAREST)
                        elif target_height:
                            scale = target_height / h
                            process_frame = cv2.resize(frame, (int(w * scale), target_height), interpolation=cv2.INTER_NEAREST)

                    success, buffer = cv2.imencode(
                        ".jpg",
                        process_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 60,      # Reduced quality = much faster
                         int(cv2.IMWRITE_JPEG_OPTIMIZE), 0]
                    )

                    if not success:
                        continue
                    
                    jpg = buffer.tobytes()
                    last_frame_count = current_count

                    # Update cache for other clients
                    with _cache_lock:
                        _mjpeg_cache[cache_key] = (current_count, jpg)
                        
                        # Cleanup cache - keep it small
                        if len(_mjpeg_cache) > 50:
                            _mjpeg_cache.clear()

                # 4. Yield the frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                    jpg +
                    b"\r\n"
                )

            except GeneratorExit:
                break
            except Exception as e:
                print(f"[RAW STREAM] {camera_id} error:", e)
                time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Disable nginx buffering if behind proxy
        }
    )

@app.route("/api/settings/alerts/test/whatsapp", methods=["POST"])
@require_admin
def test_whatsapp_cloud_alert():
    """Send test WhatsApp message via Cloud API"""
    try:
        payload = request.json
        
        # Validate required fields
        required = ["access_token", "instance_id", "recipient_number"]
        for field in required:
            if not payload.get(field):
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400

        token = payload["access_token"]
        instance_id = payload["instance_id"]
        recipient = payload["recipient_number"].replace("+", "").replace(" ", "").strip()
        
        url = "https://wspanel.in/api/send"
        message_body = payload.get(
            "message",
            "🔔 *Test Alert from PPE Detection System*\n\n"
            "If you received this, WhatsApp WS Panel API alerts are working correctly!"
        )
        
        params = {
            "number": recipient,
            "type": "text",
            "message": message_body,
            "instance_id": instance_id,
            "access_token": token
        }

        import requests
        r = requests.get(url, params=params, timeout=15)
        
        if r.status_code == 200:
            resp_data = r.json()
            if resp_data.get("status") == "error" and "Invalidated" in resp_data.get("message", ""):
                 return jsonify({
                    "success": False,
                    "error": "❌ Your WS Panel Instance ID is INVALIDATED. Please check wspanel.in, reconnect your device, and update your Instance ID."
                }), 400
            
            return jsonify({
                "success": True,
                "message": "Test WhatsApp sent successfully",
                "response": resp_data
            })
        else:
            error_text = r.text
            if "Invalidated" in error_text:
                error_text = "Your WS Panel Instance ID is INVALIDATED. Please check wspanel.in."
            return jsonify({
                "success": False,
                "error": f"WhatsApp API Error ({r.status_code}): {error_text}"
            }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/settings/alerts/test/sms", methods=["POST"])
@require_admin
def test_sms_alert():
    """Send test SMS message via Twilio"""
    try:
        payload = request.json
        
        # Validate required fields
        required = ["account_sid", "auth_token", "twilio_number", "recipient_number"]
        for field in required:
            if not payload.get(field):
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400

        # Try to import Twilio
        try:
            from twilio.rest import Client
        except ImportError:
            return jsonify({
                "success": False,
                "error": "Twilio library not installed. Run: pip install twilio"
            }), 500

        # Send test message
        try:
            client = Client(payload["account_sid"], payload["auth_token"])
            
            message_body = payload.get(
                "message",
                "🔔 Test SMS Alert from PPE Detection System. Alerts are working!"
            )
            
            message = client.messages.create(
                body=message_body,
                from_=payload["twilio_number"],
                to=payload["recipient_number"]
            )
            
            return jsonify({
                "success": True,
                "message": "Test SMS sent successfully",
                "message_sid": message.sid,
                "status": message.status
            })
            
        except Exception as twilio_error:
            return jsonify({
                "success": False,
                "error": f"Twilio error: {str(twilio_error)}"
            }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/settings/alerts", methods=["PUT"])
@require_admin
def update_alert_settings():
    """Update alert system configuration"""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        # Update database
        db.system_config.update_one(
            {"config_type": "alerts"},
            {"$set": {
                "enable_alerts": data.get("enable_alerts", True),
                "delivery_mode": data.get("delivery_mode", "whatsapp_with_sms_fallback"),
                "cooldown_seconds": data.get("cooldown_seconds", 30),
                "whatsapp_cloud": data.get("whatsapp_cloud", {}),
                "sms_twilio": data.get("sms_twilio", {}),
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )

        # Reload settings in engine
        alert_engine.reload()
        log_audit_event("Updated Alert contact/system settings")

        return jsonify({"success": True, "message": "Alert settings updated successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/settings/alerts/status", methods=["GET"])
def get_alert_status():
    """Get current alert system status"""
    try:
        config = db.system_config.find_one(
            {"config_type": "alerts"},
            {"_id": 0}
        )
        
        if not config:
            return jsonify({
                "success": True,
                "data": {
                    "configured": False,
                    "enabled": False,
                    "delivery_mode": "whatsapp_with_sms_fallback"
                }
            })
        
        # Check if WhatsApp Cloud is properly configured
        wc = config.get("whatsapp_cloud", {})
        whatsapp_configured = all([wc.get("access_token"), wc.get("phone_number_id"), wc.get("recipient_number")])
        
        # Check if SMS Twilio is properly configured
        st = config.get("sms_twilio", {})
        sms_configured = all([st.get("account_sid"), st.get("auth_token"), st.get("twilio_number"), st.get("recipient_number")])
        
        return jsonify({
            "success": True,
            "data": {
                "configured": True,
                "enabled": config.get("enable_alerts", False),
                "delivery_mode": config.get("delivery_mode", "whatsapp_with_sms_fallback"),
                "cooldown_seconds": config.get("cooldown_seconds", 30),
                "whatsapp_cloud": {
                    "configured": whatsapp_configured,
                    "has_token": bool(wc.get("access_token")),
                    "phone_id": wc.get("phone_number_id", ""),
                    "recipient": wc.get("recipient_number", "")
                },
                "sms_twilio": {
                    "configured": sms_configured,
                    "has_token": bool(st.get("auth_token")),
                    "from": st.get("twilio_number", ""),
                    "recipient": st.get("recipient_number", "")
                }
            }
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
# ==================================================
# ROLE DELETION ENDPOINT
# ==================================================

@app.route("/api/settings/ppe/<role>", methods=["DELETE"])
@require_admin
def delete_ppe_role(role):
    """Delete a PPE role"""
    try:
        role = role.lower()

        # Prevent deletion of default roles
        if role in ["default", "visitor"]:
            return jsonify({
                "success": False,
                "error": f"Cannot delete default role '{role}'"
            }), 400

        # Load current config
        config = db.system_config.find_one(
            {"config_type": "ppe_rules"}
        )

        if not config or role not in config.get("role_rules", {}):
            return jsonify({
                "success": False,
                "error": f"Role '{role}' not found"
            }), 404

        # Remove role
        role_rules = config.get("role_rules", {})
        del role_rules[role]

        # Update database
        db.system_config.update_one(
            {"config_type": "ppe_rules"},
            {
                "$set": {
                    "role_rules": role_rules,
                    "updated_at": datetime.utcnow(),
                    "updated_by": request.current_user
                }
            }
        )

        # Reload PPE system
        ppe_system.reload_ppe_rules()
        log_audit_event(f"Updated PPE compliance rules for role: {role}")

        # Log audit
        db.insert_audit_log({
            "action": "delete_role",
            "entity": f"role:{role}",
            "performed_by": request.current_user,
            "performed_at": datetime.utcnow()
        })

        return jsonify({
            "success": True,
            "message": f"Role '{role}' deleted successfully"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Attendance
@app.route("/api/persons/<person_id>/attendance")
def attendance(person_id):
    """
    ✅ FIXED: Proper date range filtering for attendance
    Returns attendance records for specified period
    """
    try:
        start_str = request.args.get("start_date")
        end_str = request.args.get("end_date")

        if start_str and end_str:
            start = datetime.fromisoformat(start_str.replace('Z', ''))
            end = datetime.fromisoformat(end_str.replace('Z', ''))
        else:
            days = int(request.args.get("days", 30))
            
            # Calculate date range
            end = datetime.utcnow()
            start = end - timedelta(days=days)
            
            # ✅ CRITICAL: Normalize to start/end of day in UTC
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        print(f"📅 Fetching attendance for {person_id}: {start} to {end}")
        
        # Query attendance collection with proper date filtering
        data = db.get_attendance(person_id, start, end)
        
        # Convert MongoDB documents to JSON-serializable format
        result = []
        for record in data:
            # Convert ObjectId
            record['_id'] = str(record['_id'])
            
            # ✅ CRITICAL: Convert date to ISO string
            if 'date' in record and isinstance(record['date'], datetime):
                record['date'] = record['date'].isoformat()
            
            # Convert event timestamps
            for event in record.get("events", []):
                if 'timestamp' in event and isinstance(event['timestamp'], datetime):
                    event['timestamp'] = event['timestamp'].isoformat()
            
            result.append(record)
        
        print(f"✅ Returning {len(result)} attendance records")
        
        return _json_ok(result)
    
    except Exception as e:
        print(f"❌ Attendance endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    




@app.route("/api/performance/fps")
def get_fps_metrics():
    return jsonify({
        "decode_fps": camera_manager.get_decode_fps(),
        "inference_fps": inference_controller.get_inference_fps()
    })


@app.route("/api/cameras/fps")
def camera_inference_fps():
    """
    Frontend-compatible endpoint
    Returns INFERENCE FPS per camera
    """
    return jsonify(inference_controller.get_inference_fps(per_camera=True))


# Add these endpoints to api_server_n.py

@app.route("/api/cameras/health", methods=["GET"])
def get_cameras_health():
    """
    Get health status for all cameras
    Useful for monitoring and diagnostics
    """
    try:
        health_data = camera_manager.get_all_camera_status()
        
        return jsonify({
            "success": True,
            "data": health_data,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/cameras/<camera_id>/health", methods=["GET"])
def get_camera_health(camera_id):
    """Get health status for specific camera"""
    try:
        health = camera_manager.get_stream_health(camera_id)
        
        return jsonify({
            "success": True,
            "camera_id": camera_id,
            "health": health,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/cameras/<camera_id>/restart", methods=["POST"])
@require_admin
def restart_camera(camera_id):
    """
    Manually restart a camera
    Useful when a camera is stuck
    """
    try:
        success = camera_manager.restart_camera(camera_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Camera {camera_id} restarted"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to restart camera"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/cameras/test-rtsp", methods=["POST"])
@require_admin
def test_rtsp_url():
    """
    Test an RTSP URL without adding it to the system
    Useful for validating camera URLs before registration
    """
    try:
        data = request.json
        rtsp_url = data.get("rtsp_url")
        
        if not rtsp_url:
            return jsonify({
                "success": False,
                "error": "rtsp_url is required"
            }), 400
        
        # Try to connect
        import cv2
        print(f"🧪 Testing RTSP URL: {rtsp_url}")
        
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5s timeout
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        
        if not cap.isOpened():
            cap.release()
            return jsonify({
                "success": False,
                "error": "Failed to open RTSP stream",
                "details": "Could not connect to the stream. Check URL, network, and camera status."
            }), 400
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({
                "success": False,
                "error": "Failed to read frame from stream",
                "details": "Connected but could not retrieve video frames."
            }), 400
        
        # Get frame info
        height, width = frame.shape[:2]
        
        return jsonify({
            "success": True,
            "message": "RTSP stream is accessible",
            "resolution": {
                "width": width,
                "height": height
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/cameras/status")
def camera_status():
    cameras = db.get_all_cameras(status="active")
    status = []

    for cam in cameras:
        has_alert = db.recognition_logs.find_one({
            "camera_id": cam["camera_id"],
            "log_type": "ppe_violation",
            "resolved": False
        }) is not None

        status.append({
            "camera_id": cam["camera_id"],
            "has_alert": has_alert
        })

    return jsonify({"success": True, "data": status})


"""
Add these diagnostic endpoints to api_server_n.py
to help debug detection and logging issues
"""

# Add these imports at the top if not already present:
from datetime import datetime, timedelta

# Add these endpoints to your Flask app:

@app.route("/api/diagnostics/detection", methods=["GET"])
def diagnostics_detection():
    """
    Get detection system status
    Helps debug why detections might not be working
    """
    try:
        camera_id = request.args.get("camera_id")
        
        # Check inference controller status
        inference_running = inference_controller.running
        
        # Get latest results
        with inference_controller.lock:
            if camera_id:
                latest_result = inference_controller.latest_results.get(camera_id)
                violations = inference_controller.live_violations.get(camera_id, [])
            else:
                latest_result = dict(inference_controller.latest_results)
                violations = dict(inference_controller.live_violations)
        
        # Check camera manager status
        with camera_manager.lock:
            frames = {k: v is not None for k, v in camera_manager.latest_frames.items()}
        
        # Check face recognition system
        face_system_status = {
            "embeddings_loaded": len(face_system.embeddings_cache),
            "persons_registered": len(db.get_all_persons("active")),
            "threshold": face_system.threshold
        }
        
        # Check PPE system
        ppe_system_status = {
            "model_loaded": ppe_system.model is not None,
            "confidence_threshold": ppe_system.confidence_threshold,
            "role_rules_loaded": len(ppe_system.role_ppe_requirements)
        }
        
        return jsonify({
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "inference_controller": {
                "running": inference_running,
                "fps": inference_controller.get_inference_fps(),
                "per_camera_fps": inference_controller.get_inference_fps(per_camera=True)
            },
            "camera_manager": {
                "active_cameras": list(frames.keys()),
                "frames_available": frames,
                "decode_fps": camera_manager.get_decode_fps()
            },
            "face_recognition": face_system_status,
            "ppe_detection": ppe_system_status,
            "latest_results": latest_result,
            "recent_violations": violations
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/diagnostics/violations", methods=["GET"])
def diagnostics_violations():
    """
    Check violation logging status
    Shows recent violations and logging stats
    """
    try:
        # Get violations from last 24 hours
        yesterday = datetime.utcnow() - timedelta(days=1)
        
        recent_violations = list(
            db.recognition_logs.find({
                "log_type": "ppe_violation",
                "timestamp": {"$gte": yesterday}
            })
            .sort("timestamp", -1)
            .limit(20)
        )
        
        # Convert ObjectId to string
        for v in recent_violations:
            v["_id"] = str(v["_id"])
            v["timestamp"] = v["timestamp"].isoformat()
        
        # Get stats
        total_violations = db.recognition_logs.count_documents({
            "log_type": "ppe_violation"
        })
        
        violations_today = db.recognition_logs.count_documents({
            "log_type": "ppe_violation",
            "timestamp": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0)}
        })
        
        violations_last_hour = db.recognition_logs.count_documents({
            "log_type": "ppe_violation",
            "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=1)}
        })
        
        # Check cooldown status
        cooldown_status = {}
        for key, timestamp in integrated_system.violation_cooldown.items():
            time_left = integrated_system.VIOLATION_COOLDOWN_SEC - (time.time() - timestamp)
            if time_left > 0:
                cooldown_status[key] = f"{time_left:.1f}s remaining"
        
        return jsonify({
            "success": True,
            "statistics": {
                "total_violations": total_violations,
                "violations_today": violations_today,
                "violations_last_hour": violations_last_hour
            },
            "recent_violations": recent_violations,
            "cooldown_status": cooldown_status,
            "cooldown_seconds": integrated_system.VIOLATION_COOLDOWN_SEC
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# @app.route("/api/diagnostics/reset-cooldown", methods=["POST"])
# def reset_violation_cooldown():
#     """
#     Reset violation cooldown (for testing)
#     Use this if you want to force re-logging of violations
#     """
#     try:
#         integrated_system.violation_cooldown.clear()
        
#         return jsonify({
#             "success": True,
#             "message": "Violation cooldown reset. All violations will be logged again."
#         })
    
#     except Exception as e:
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500


@app.route("/api/diagnostics/ppe-rules", methods=["GET"])
def diagnostics_ppe_rules():
    """
    Check current PPE rules configuration
    """
    try:
        # Get from PPE system
        rules = ppe_system.role_ppe_requirements
        
        # Get from database
        db_config = db.system_config.find_one({"config_type": "ppe_rules"})
        
        return jsonify({
            "success": True,
            "loaded_rules": rules,
            "database_config": db_config.get("role_rules") if db_config else None,
            "rules_match": rules == (db_config.get("role_rules") if db_config else {})
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/alert_images/<filename>")
def serve_alert_image(filename):
    return send_from_directory(get_resource_path("alert_images"), filename)


@app.route("/incident_videos/<filename>")
def serve_incident_video(filename):
    return send_from_directory(get_resource_path("incident_videos"), filename)


@app.route("/api/incidents/videos", methods=["GET"])
def list_incident_videos():
    """List all recorded incident videos"""
    try:
        from pathlib import Path
        video_dir = Path(get_resource_path("incident_videos"))
        if not video_dir.exists():
            return jsonify({"success": True, "videos": []})
        
        videos = []
        for file in video_dir.glob("*.mp4"):
            stats = file.stat()
            videos.append({
                "filename": file.name,
                "url": f"/incident_videos/{file.name}",
                "size": stats.st_size,
                "created_at": datetime.fromtimestamp(stats.st_mtime).isoformat()
            })
            
        # Sort by recency
        videos.sort(key=lambda x: x["created_at"], reverse=True)
        return jsonify({"success": True, "videos": videos})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/dashboard/stats", methods=["GET"])
def get_dashboard_stats():
    """Retrieve comprehensive system stats for the dashboard"""
    try:
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_start = today_start - timedelta(days=1)
        
        # 1. CAMERA STATS
        # Use active_streams (in-memory, always current) for the live active count.
        # is_online in MongoDB is only updated by the heartbeat task and can lag.
        total_cameras = db.cameras.count_documents({"status": "active"})
        active_cameras = len(active_streams)
        
        # 2. WORKER STATS
        total_workers = db.persons.count_documents({})
        # Attendance today?
        attendance_today = db.attendance.count_documents({"date": today_start})
        
        # 3. VIOLATION STATS (Today vs Yesterday)
        today_violations = db.recognition_logs.count_documents({
            "timestamp": {"$gte": today_start},
            "log_type": {"$in": ["ppe_violation", "zone_violation", "night_mode_violation"]}
        })
        yesterday_violations = db.recognition_logs.count_documents({
            "timestamp": {"$gte": yesterday_start, "$lt": today_start},
            "log_type": {"$in": ["ppe_violation", "zone_violation", "night_mode_violation"]}
        })
        
        # 4. EMERGENCY STATS (Fire, Falls)
        today_emergencies = db.recognition_logs.count_documents({
            "timestamp": {"$gte": today_start},
            "log_type": {"$in": ["fire_violation", "fall_violation"]}
        })
        
        # 5. VIOLATION TREND (Last 7 days)
        trend = []
        for i in range(6, -1, -1):
            day_start = today_start - timedelta(days=i)
            day_end = day_start + timedelta(days=1)
            count = db.recognition_logs.count_documents({
                "timestamp": {"$gte": day_start, "$lt": day_end},
                "log_type": {"$in": ["ppe_violation", "zone_violation", "night_mode_violation"]}
            })
            trend.append({
                "date": day_start.strftime("%m-%d"),
                "count": count
            })
            
        # 6. ROLE BREAKDOWN (Violations)
        role_counts = list(db.recognition_logs.aggregate([
            {"$match": {"timestamp": {"$gte": today_start}}},
            {"$group": {"_id": "$role", "count": {"$sum": 1}}}
        ]))
        roles = {r["_id"]: r["count"] for r in role_counts if r["_id"]}
        
        # 7. TOP VIOLATIONS TYPES
        # If missing_ppe is a list, flatly count them
        pipeline = [
            {"$match": {"timestamp": {"$gte": today_start}, "log_type": "ppe_violation"}},
            {"$unwind": "$missing_ppe"},
            {"$group": {"_id": "$missing_ppe", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        top_missing_ppe = list(db.recognition_logs.aggregate(pipeline))
        violation_types = {item["_id"]: item["count"] for item in top_missing_ppe}
        
        # 8. LATEST VIDEOS
        from pathlib import Path
        video_dir = Path(get_resource_path("incident_videos"))
        latest_videos = []
        if video_dir.exists():
            # Filter: Sort by time, reverse, then take files that ARE NOT temporary or test files
            all_files = sorted(video_dir.glob("*.mp4"), key=os.path.getmtime, reverse=True)
            for file in all_files:
                if file.name.startswith("REC_TEMP_") or file.name.startswith("test_"):
                    continue
                
                # Basic check for valid incident filename format
                if "_Violation_" not in file.name:
                    continue
                
                latest_videos.append({
                    "filename": file.name,
                    "url": f"/incident_videos/{file.name}",
                    "created_at": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                })
                
                if len(latest_videos) >= 5:
                    break

        return jsonify({
            "success": True,
            "stats": {
                "cameras": {"total": total_cameras, "active": active_cameras},
                "workers": {"total": total_workers, "attendance": attendance_today},
                "violations": {
                    "today": today_violations, 
                    "yesterday": yesterday_violations,
                    "trend": trend
                },
                "emergencies": {"today": today_emergencies},
                "role_breakdown": roles,
                "violation_types": violation_types,
                "latest_videos": latest_videos
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/system/roles")
def get_roles():
    cfg = db.system_config.find_one({"config_type": "ppe_rules"}) or {}
    roles = list(cfg.get("role_rules", {}).keys())
    return jsonify({"success": True, "roles": roles})


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


@app.route('/api/settings/users', methods=['GET'])
@require_admin
def get_users():
    """Get all users (admin only)"""
    users = list(db.users.find({}, {"password": 0}))
    for u in users:
        u["_id"] = str(u["_id"])
    return jsonify({"success": True, "data": users})

@app.route('/api/settings/users', methods=['PUT'])
@require_admin
def update_user_password():
    """Change user password (admin only or self)"""
    data = request.json
    username = data.get("username")
    new_password = data.get("password")
    current_password = data.get("current_password") # Added current_password
    
    if not username or not new_password or not current_password:
        return jsonify({"success": False, "error": "Username, current password, and new password are required"}), 400
        
    # Verify current password
    user = db.users.find_one({
        "username": username,
        "password": current_password
    })
    
    if not user:
        return jsonify({"success": False, "error": "Invalid current password"}), 401

    if len(new_password) < 4:
        return jsonify({"success": False, "error": "New password must be at least 4 characters"}), 400

    result = db.users.update_one(
        {"username": username},
        {"$set": {"password": new_password}}
    )
    
    if result.matched_count > 0:
        # Log audit
        db.insert_audit_log({
            "action": "change_password",
            "entity": f"user:{username}",
            "performed_by": request.current_user,
            "performed_at": datetime.utcnow()
        })
        return jsonify({"success": True, "message": f"Password for {username} updated"})
    return jsonify({"success": False, "error": "User not found"}), 404




# ============================================================================
# OCR / Number Plate API
# ============================================================================

@app.route('/api/ocr/plates', methods=['GET'])
def get_plate_logs():
    """Fetch plate detection history from DB, newest first."""
    try:
        limit = int(request.args.get('limit', 100))
        camera_id = request.args.get('camera_id')
        query = {'log_type': 'plate_detection'}
        if camera_id:
            query['camera_id'] = camera_id
        logs = list(
            db.recognition_logs.find(
                query,
                {'_id': 1, 'plate_number': 1, 'confidence': 1,
                 'entry_exit': 1, 'timestamp': 1, 'camera_id': 1, 'snapshot_url': 1}
            ).sort('timestamp', -1).limit(limit)
        )
        result = []
        for log in logs:
            ts = log.get('timestamp')
            result.append({
                'id': str(log['_id']),
                'plate_number': log.get('plate_number', ''),
                'confidence': log.get('confidence', 0),
                'entry_exit': log.get('entry_exit', 'Entry'),
                'timestamp': ts.isoformat() if ts else '',
                'camera_id': log.get('camera_id', ''),
                'snapshot_url': log.get('snapshot_url', '')
            })
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/ocr/camera', methods=['GET'])
def get_ocr_camera():
    """Return the camera currently designated for OCR processing."""
    cam_id = integrated_system.ocr_camera_id
    return jsonify({'success': True, 'camera_id': cam_id})


@app.route('/api/ocr/camera', methods=['POST'])
def set_ocr_camera():
    """
    Set which camera should run OCR/plate detection.
    POST body: { "camera_id": "CAMV2" }  or  { "camera_id": null } to disable.
    """
    data = request.json or {}
    camera_id = data.get('camera_id')          # None = disable OCR on all
    integrated_system.ocr_camera_id = camera_id
    # Flush the cache for the old camera so stale plates don't linger
    integrated_system.ocr_cache = {}
    integrated_system.ocr_last_check = {}
    print(f"📷 OCR camera set to: {camera_id or 'DISABLED'}")
    return jsonify({'success': True, 'camera_id': camera_id})


# ============================================================================
# Main
# ============================================================================




if __name__ == '__main__':
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print(f"""
    {'='*70}
    🚀 Enhanced Face Recognition + PPE Detection API Server
    {'='*70}
    Server running on: http://{HOST}:{PORT}
    Database: {db.db.name}
    Registered persons: {len(face_system.embeddings_cache)}
    Press Ctrl+C to stop
    {'='*70}
    """)
    
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)