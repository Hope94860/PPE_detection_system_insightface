"""
GPU-enabled Face Encoder - Patched version for proper GPU support
"""
import os
import numpy as np
from typing import List, Dict, Tuple
from path_utils import get_resource_path
import insightface
from insightface.model_zoo import model_zoo
import onnxruntime as ort


class GPUFaceAnalysis:
    """
    GPU-enabled FaceAnalysis that properly uses CUDA
    """
    def __init__(self, name, root=None, use_gpu=True):
        self.models = {}
        if root is None:
            # Default to bundled models if available, else ~/.insightface/models
            root = get_resource_path('insightface_models')
            if not os.path.exists(root):
                root = os.path.expanduser('~/.insightface/models')
        else:
            root = os.path.expanduser(root)
        
        # Set providers based on GPU preference
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("🚀 GPU Providers: CUDAExecutionProvider enabled")
        else:
            providers = ['CPUExecutionProvider']
            print("💻 CPU Provider enabled")
        
        import glob
        import os.path as osp
        
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_') > 0:
                continue
            
            # Create session with GPU providers
            session = ort.InferenceSession(onnx_file, providers=providers)
            
            # Verify which provider is being used
            active_providers = session.get_providers()
            print(f"  Model: {osp.basename(onnx_file)}")
            print(f"  Active provider: {active_providers[0]}")
            
            # Get model from session
            model = self._get_model_from_session(onnx_file, session)
            
            if model and model.taskname not in self.models:
                print(f'find model: {onnx_file} {model.taskname}')
                self.models[model.taskname] = model
            elif model:
                print(f'duplicated model task type, ignore: {onnx_file} {model.taskname}')
                del model
        
        assert 'detection' in self.models, "Detection model not found!"
        self.det_model = self.models['detection']
    
    def _get_model_from_session(self, onnx_file, session):
        """Get appropriate model class from ONNX session"""
        from insightface.model_zoo import SCRFD, ArcFaceONNX
        
        input_cfg = session.get_inputs()[0]
        input_shape = input_cfg.shape
        outputs = session.get_outputs()
        
        # Determine model type
        if len(outputs) >= 5:
            # Detection model (SCRFD)
            model = SCRFD(model_file=onnx_file, session=session)
            return model
        elif len(input_shape) >= 4 and input_shape[2] == 112 and input_shape[3] == 112:
            # Recognition model (ArcFace)
            model = ArcFaceONNX(model_file=onnx_file, session=session)
            return model
        else:
            return None
    
    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        """Prepare models for inference"""
        self.det_thresh = det_thresh

        # FORCE tuple for SCRFD compatibility
        if isinstance(det_size, int):
            det_size = (det_size, det_size)

        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        
        for taskname, model in self.models.items():
            if taskname == 'detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    
    def get(self, img, max_num=0):
        """Detect and recognize faces in image"""
        try:
            # SCRFD.detect() signature varies by version
            # Based on error, trying: detect(img, input_size, max_num, thresh)
            # Ensure max_num is an integer
            max_num_int = int(max_num) if max_num else 0
            det_size = self.det_size
            if isinstance(det_size, int):
                det_size = (det_size, det_size)

            # Fixed: Use explicit keyword arguments or correct order to match SCRFD.detect signature:
            # detect(img, threshold=0.5, input_size=None, max_num=0, metric='default')
            result = self.det_model.detect(img, threshold=self.det_thresh, input_size=det_size, max_num=max_num_int)

            # Check if result is a tuple (bboxes, kpss) or something else
            if isinstance(result, tuple) and len(result) == 2:
                bboxes, kpss = result
            else:
                # If not a tuple, result might be just bboxes
                bboxes = result
                kpss = None
            
            # Check if bboxes is valid
            if bboxes is None or (hasattr(bboxes, 'shape') and bboxes.shape[0] == 0):
                return []
            
            # If bboxes is a float or scalar, no faces detected
            if isinstance(bboxes, (float, int)):
                return []
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            
            embedding = None
            normed_embedding = None
            embedding_norm = None
            
            if 'recognition' in self.models:
                if kps is not None:
                    rec_model = self.models['recognition']
                    from insightface.utils import face_align
                    aimg = face_align.norm_crop(img, landmark=kps)
                    embedding = rec_model.get_feat(aimg).flatten()
                    from numpy.linalg import norm
                    embedding_norm = norm(embedding)
                    normed_embedding = embedding / embedding_norm
            
            # Create Face object - some versions don't accept all parameters in constructor
            from insightface.app.face_analysis import Face
            try:
                # Try with all parameters
                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                # Set additional attributes after creation
                face.embedding = embedding
                face.normed_embedding = normed_embedding
                face.embedding_norm = embedding_norm
                face.gender = None
                face.age = None
            except Exception as e:
                # Fallback: create a simple object with the attributes we need
                class SimpleFace:
                    def __init__(self):
                        self.bbox = bbox
                        self.kps = kps
                        self.det_score = det_score
                        self.embedding = embedding
                        self.normed_embedding = normed_embedding
                        self.embedding_norm = embedding_norm
                        self.gender = None
                        self.age = None
                face = SimpleFace()
            
            ret.append(face)
        
        return ret


class FaceEncoder:
    """
    Face encoder using InsightFace with proper GPU support
    """
    
    def __init__(self, model_name: str = 'buffalo_l', ctx_id: int = 0, 
                 det_size: Tuple[int, int] = (640, 640), use_gpu: bool = True):
        """
        Initialize the face encoder
        
        Args:
            model_name: Name of the InsightFace model
            ctx_id: Context ID (0 for GPU 0, -1 for CPU)
            det_size: Detection size for face detection
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        print("Initializing FaceEncoder...")
        
        # Use custom GPU-enabled FaceAnalysis
        self.app = GPUFaceAnalysis(name=model_name, use_gpu=use_gpu)
        
        # Set context ID based on GPU preference
        if use_gpu:
            actual_ctx_id = 0  # GPU 0
        else:
            actual_ctx_id = -1  # CPU
        
        self.app.prepare(ctx_id=actual_ctx_id, det_size=det_size)
        print(f"✓ FaceEncoder initialized with models: {list(self.app.models.keys())}")
    
    def get_face_embedding(self, image: np.ndarray, return_largest: bool = True) -> Tuple[np.ndarray, Dict]:
        """Extract face embedding from an image"""
        faces = self.app.get(image)
        
        if len(faces) == 0:
            return None, None
        
        if return_largest and len(faces) > 1:
            areas = [(face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]) for face in faces]
            largest_idx = np.argmax(areas)
            face = faces[largest_idx]
        else:
            face = faces[0]
        
        face_info = {
            'bbox': face.bbox,
            'landmarks': face.kps,
            'det_score': face.det_score,
            'embedding': face.normed_embedding
        }
        
        return face.normed_embedding, face_info
    
    def get_all_face_embeddings(self, image: np.ndarray) -> List[Tuple[np.ndarray, Dict]]:
        """Extract embeddings for all faces in an image"""
        faces = self.app.get(image)
        
        results = []
        for face in faces:
            face_info = {
                'bbox': face.bbox,
                'landmarks': face.kps,
                'det_score': face.det_score,
                'embedding': face.normed_embedding
            }
            results.append((face.normed_embedding, face_info))
        
        return results
    
    def encode_face_from_path(self, image_path: str, return_largest: bool = True) -> Tuple[np.ndarray, Dict]:
        """Extract face embedding from image file path"""
        from utils import load_image
        image = load_image(image_path)
        return self.get_face_embedding(image, return_largest)
    
    def build_database_from_directory(self, known_faces_dir: str, database_path: str) -> Dict:
        """Build a face database from a directory of known faces"""
        from utils import get_image_files, save_database
        
        print(f"\nBuilding face database from {known_faces_dir}...")
        database = {}
        
        if not os.path.exists(known_faces_dir):
            print(f"Directory not found: {known_faces_dir}")
            return database
        
        person_dirs = [d for d in os.listdir(known_faces_dir) 
                      if os.path.isdir(os.path.join(known_faces_dir, d))]
        
        if len(person_dirs) == 0:
            print("No person directories found!")
            return database
        
        total_faces = 0
        for person_name in person_dirs:
            person_dir = os.path.join(known_faces_dir, person_name)
            image_files = get_image_files(person_dir)
            
            if len(image_files) == 0:
                print(f"  ⚠ No images found for {person_name}")
                continue
            
            person_embeddings = []
            print(f"\n  Processing {person_name}:")
            
            for img_path in image_files:
                try:
                    embedding, face_info = self.encode_face_from_path(img_path)
                    
                    if embedding is not None:
                        person_embeddings.append(embedding)
                        print(f"    ✓ {os.path.basename(img_path)}")
                        total_faces += 1
                    else:
                        print(f"    ✗ No face detected in {os.path.basename(img_path)}")
                        
                except Exception as e:
                    print(f"    ✗ Error processing {os.path.basename(img_path)}: {e}")
            
            if len(person_embeddings) > 0:
                database[person_name] = {
                    'embeddings': person_embeddings,
                    'num_images': len(person_embeddings)
                }
                print(f"    → Added {len(person_embeddings)} embeddings for {person_name}")
            else:
                print(f"    ⚠ No valid embeddings for {person_name}")
        
        if len(database) > 0:
            save_database(database, database_path)
            print(f"\n✓ Database created with {len(database)} identities and {total_faces} total face embeddings")
        else:
            print("\n✗ No faces were encoded. Database is empty.")
        
        return database
