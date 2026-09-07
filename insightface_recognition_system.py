"""
InsightFace Recognition System integrated with MongoDB
Combines face detection/recognition with PPE detection pipeline
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time
from collections import defaultdict, deque
from threading import Lock

from face_encoder import FaceEncoder
from face_recognizer import FaceRecognizer
from mongo_db_manager import FaceRecognitionDB


class InsightFaceRecognitionSystem:
    """
    InsightFace-based face recognition system integrated with MongoDB
    """
    
    def __init__(self, db: FaceRecognitionDB, similarity_threshold: float = 0.4, 
                 use_cuda: bool = True, model_name: str = 'buffalo_l'):
        """
        Initialize InsightFace recognition system
        
        Args:
            db: MongoDB database manager
            similarity_threshold: Minimum similarity score for recognition (0-1)
            use_cuda: Whether to use GPU acceleration
            model_name: InsightFace model name
        """
        print("🚀 Initializing InsightFace Recognition System...")
        
        self.db = db
        self.similarity_threshold = similarity_threshold
        self.use_cuda = use_cuda
        
        # Initialize face encoder
        self.encoder = FaceEncoder(model_name=model_name, use_gpu=use_cuda)
        
        # Load embeddings from database
        self.embeddings_cache = {}
        self._load_embeddings_from_db()
        
        # Prepare database for fast matching
        self._prepare_database()
        
        # Tracking for unknown faces
        self.unknown_trackers = {}
        self.track_id_counter = defaultdict(int)
        self.track_last_seen = {}
        self.lock = Lock()
        
        print(f"✅ InsightFace System initialized with {len(self.embeddings_cache)} persons")
        print(f"   Similarity threshold: {self.similarity_threshold}")
        print(f"   GPU enabled: {self.use_cuda}")
    
    def _load_embeddings_from_db(self):
        """Load face embeddings from MongoDB"""
        persons = self.db.get_all_persons(status='active')
        
        for person in persons:
            person_id = person['person_id']
            embeddings = person.get('embeddings', [])
            
            if len(embeddings) > 0:
                # Extract embedding vectors
                embedding_vectors = []
                for emb in embeddings:
                    if 'vector' in emb:
                        embedding_vectors.append(np.array(emb['vector']))
                
                if len(embedding_vectors) > 0:
                    self.embeddings_cache[person_id] = {
                        'name': person['name'],
                        'role': person.get('role', 'default'),
                        'embeddings': embedding_vectors,
                        'person_data': person
                    }
        
        print(f"📥 Loaded {len(self.embeddings_cache)} persons from database")
    
    def _prepare_database(self):
        """Prepare database embeddings for efficient matching"""
        self.db_person_ids = []
        self.db_embeddings = []
        
        for person_id, data in self.embeddings_cache.items():
            for embedding in data['embeddings']:
                self.db_person_ids.append(person_id)
                self.db_embeddings.append(embedding)
        
        if len(self.db_embeddings) > 0:
            self.db_embeddings = np.array(self.db_embeddings)
            print(f"✅ Prepared {len(self.db_embeddings)} embeddings for matching")
    
    def recognize_face(self, embedding: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """
        Recognize a face from its embedding
        
        Args:
            embedding: Face embedding vector
            
        Returns:
            Tuple of (person_id, person_name, similarity_score)
        """
        if len(self.db_embeddings) == 0:
            return None, None, 0.0
        
        # Calculate similarities with all database embeddings
        similarities = np.dot(self.db_embeddings, embedding)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        # Check if similarity meets threshold
        if best_similarity >= self.similarity_threshold:
            person_id = self.db_person_ids[best_idx]
            person_name = self.embeddings_cache[person_id]['name']
            return person_id, person_name, float(best_similarity)
        else:
            return None, None, float(best_similarity)
    
    def process_frame(self, frame: np.ndarray, camera_id: str, 
                     store_logs: bool = True) -> List[Dict]:
        """
        Process a frame for face detection and recognition
        
        Args:
            frame: Input frame (BGR format)
            camera_id: Camera identifier
            store_logs: Whether to store recognition logs in database
            
        Returns:
            List of face detection results
        """
        # Get all face embeddings from frame
        face_results = self.encoder.get_all_face_embeddings(frame)
        
        results = []
        timestamp = datetime.utcnow()
        
        for embedding, face_info in face_results:
            # Recognize the face
            person_id, person_name, similarity = self.recognize_face(embedding)
            
            # Prepare bounding box
            bbox = face_info['bbox']
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Build result
            result = {
                'box': (x1, y1, x2, y2),
                'person_id': person_id if person_id else 'unknown',
                'person_name': person_name if person_name else 'Unknown',
                'confidence': similarity,
                'det_score': face_info['det_score'],
                'is_recognized': person_id is not None,
                'landmarks': face_info['landmarks'],
                'embedding': embedding
            }
            
            # Get person role if recognized
            if person_id and person_id in self.embeddings_cache:
                result['role'] = self.embeddings_cache[person_id]['role']
            else:
                result['role'] = 'visitor'
            
            results.append(result)
            
            # Store recognition log if enabled and person is recognized
            if store_logs and person_id:
                self._store_recognition_log(person_id, camera_id, similarity, timestamp)
        
        return results
    
    def _store_recognition_log(self, person_id: str, camera_id: str, 
                               confidence: float, timestamp: datetime):
        """Store recognition event in database"""
        try:
            log_data = {
                'person_id': person_id,
                'camera_id': camera_id,
                'confidence': confidence,
                'timestamp': timestamp,
                'log_type': 'face_recognition'
            }
            self.db.recognition_logs.insert_one(log_data)
        except Exception as e:
            print(f"⚠️ Failed to store recognition log: {e}")
    
    def _resize_image_if_needed(self, image: np.ndarray, max_size: int = 1280) -> np.ndarray:
        """Resize image if it exceeds max_size while maintaining aspect ratio"""
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # print(f"  ⬇️  Resized image from {w}x{h} to {new_w}x{new_h}")
        return image

    def register_person_from_image(self, image_path: str, person_id: str, name: str,
                                   email: str = '', phone: str = '', role: str = 'employee',
                                   department: str = '', tags: List[str] = None,
                                   access_level: int = 1, registered_by: str = 'system') -> bool:
        """
        Register a new person from an image file
        
        Args:
            image_path: Path to the person's image
            person_id: Unique person identifier
            name: Person's name
            email: Email address
            phone: Phone number
            role: Person's role
            department: Department name
            tags: List of tags
            access_level: Access level (1-5)
            registered_by: Who registered this person
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Failed to load image: {image_path}")
                return False

            # Resize if too large
            image = self._resize_image_if_needed(image)

            # Extract face embedding
            embedding, face_info = self.encoder.get_face_embedding(image, return_largest=True)
            
            if embedding is None:
                print(f"❌ No face detected in {image_path}")
                return False
            
            # Store image in database
            image_id = self.db.store_face_image(image, metadata={'person_id': person_id, 'type': 'registration'})
            
            # Prepare embedding data
            embedding_data = {
                'vector': embedding.tolist(),
                'image_id': image_id,
                'registered_at': datetime.utcnow(),
                'registered_by': registered_by,
                'quality_score': float(face_info['det_score'])
            }
            
            # Check if person exists
            existing_person = self.db.get_person(person_id)
            
            if existing_person:
                # Update existing person with new embedding
                self.db.persons.update_one(
                    {'person_id': person_id},
                    {
                        '$push': {'embeddings': embedding_data},
                        '$set': {'updated_at': datetime.utcnow()}
                    }
                )
                print(f"✅ Added new embedding for existing person: {name}")
            else:
                # Create new person
                person_data = {
                    'person_id': person_id,
                    'name': name,
                    'email': email,
                    'phone': phone,
                    'role': role,
                    'department': department,
                    'tags': tags or [],
                    'access_level': access_level,
                    'embeddings': [embedding_data],
                    'status': 'active',
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow(),
                    'registered_by': registered_by
                }
                
                self.db.persons.insert_one(person_data)
                print(f"✅ Registered new person: {name}")
            
            # Reload embeddings cache
            self._load_embeddings_from_db()
            self._prepare_database()
            
            return True
            
        except Exception as e:
            print(f"❌ Error registering person: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def register_person_from_frame(self, frame: np.ndarray, person_id: str, name: str,
                                   email: str = '', phone: str = '', role: str = 'employee',
                                   department: str = '', tags: List[str] = None,
                                   access_level: int = 1, registered_by: str = 'system') -> bool:
        """
        Register a new person from a video frame
        
        Args:
            frame: Input frame containing the person's face
            person_id: Unique person identifier
            name: Person's name
            (other args same as register_person_from_image)
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Extract face embedding
            embedding, face_info = self.encoder.get_face_embedding(frame, return_largest=True)
            
            if embedding is None:
                print(f"❌ No face detected in frame")
                return False
            
            # Store frame in database (frame is already a numpy array)
            image_id = self.db.store_face_image(frame, metadata={'person_id': person_id, 'type': 'registration'})
            
            # Prepare embedding data
            embedding_data = {
                'vector': embedding.tolist(),
                'image_id': image_id,
                'registered_at': datetime.utcnow(),
                'registered_by': registered_by,
                'quality_score': float(face_info['det_score'])
            }
            
            # Check if person exists
            existing_person = self.db.get_person(person_id)
            
            if existing_person:
                # Update existing person with new embedding
                self.db.persons.update_one(
                    {'person_id': person_id},
                    {
                        '$push': {'embeddings': embedding_data},
                        '$set': {'updated_at': datetime.utcnow()}
                    }
                )
                print(f"✅ Added new embedding for existing person: {name}")
            else:
                # Create new person
                person_data = {
                    'person_id': person_id,
                    'name': name,
                    'email': email,
                    'phone': phone,
                    'role': role,
                    'department': department,
                    'tags': tags or [],
                    'access_level': access_level,
                    'embeddings': [embedding_data],
                    'status': 'active',
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow(),
                    'registered_by': registered_by
                }
                
                self.db.persons.insert_one(person_data)
                print(f"✅ Registered new person: {name}")
            
            # Reload embeddings cache
            self._load_embeddings_from_db()
            self._prepare_database()
            
            return True
            
        except Exception as e:
            print(f"❌ Error registering person from frame: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def register_person_from_folder(self, folder_path: str, person_id: str, name: str,
                                     email: str = '', phone: str = '', role: str = 'employee',
                                     department: str = '', tags: List[str] = None,
                                     access_level: int = 1, registered_by: str = 'system') -> bool:
        """
        Register a new person from a folder of images
        
        Args:
            folder_path: Path to folder containing person's images
            person_id: Unique person identifier
            name: Person's name
            email: Email address
            phone: Phone number
            role: Person's role
            department: Department name
            tags: List of tags
            access_level: Access level (1-5)
            registered_by: Who registered this person
            
        Returns:
            True if registration successful, False otherwise
        """
        import os
        from pathlib import Path
        
        try:
            folder = Path(folder_path)
            if not folder.exists():
                print(f"❌ Folder not found: {folder_path}")
                return False
            
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(folder.glob(f'*{ext}'))
                image_files.extend(folder.glob(f'*{ext.upper()}'))
            
            if len(image_files) == 0:
                print(f"❌ No images found in {folder_path}")
                return False
            
            print(f"📸 Found {len(image_files)} images for {name}")
            
            # Process each image
            embeddings_data = []
            successful_count = 0
            
            for img_path in image_files:
                try:
                    # Load image first
                    image = cv2.imread(str(img_path))
                    if image is None:
                        print(f"  ❌ Failed to load image: {img_path.name}")
                        continue
                        
                    # Resize if too large
                    image = self._resize_image_if_needed(image)
                    
                    # Extract face embedding
                    embedding, face_info = self.encoder.get_face_embedding(image, return_largest=True)
                    
                    if embedding is None:
                        print(f"  ⚠️  No face detected in {img_path.name}")
                        continue
                    
                    # Store image in database
                    image_id = self.db.store_face_image(image, metadata={'person_id': person_id, 'type': 'registration'})
                    
                    # Prepare embedding data
                    embedding_data = {
                        'vector': embedding.tolist(),
                        'image_id': image_id,
                        'registered_at': datetime.utcnow(),
                        'registered_by': registered_by,
                        'quality_score': float(face_info['det_score'])
                    }
                    
                    embeddings_data.append(embedding_data)
                    successful_count += 1
                    print(f"  ✅ Processed {img_path.name} (quality: {face_info['det_score']:.3f})")
                    
                except Exception as e:
                    print(f"  ❌ Error processing {img_path.name}: {e}")
                    continue
            
            if successful_count == 0:
                print(f"❌ No faces detected in any images")
                return False
            
            # Check if person exists
            existing_person = self.db.get_person(person_id)
            
            if existing_person:
                # Update existing person with new embeddings
                self.db.persons.update_one(
                    {'person_id': person_id},
                    {
                        '$push': {'embeddings': {'$each': embeddings_data}},
                        '$set': {'updated_at': datetime.utcnow()}
                    }
                )
                print(f"✅ Added {successful_count} embeddings for existing person: {name}")
            else:
                # Create new person
                person_data = {
                    'person_id': person_id,
                    'name': name,
                    'email': email,
                    'phone': phone,
                    'role': role,
                    'department': department,
                    'tags': tags or [],
                    'access_level': access_level,
                    'embeddings': embeddings_data,
                    'status': 'active',
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow(),
                    'registered_by': registered_by
                }
                
                self.db.persons.insert_one(person_data)
                print(f"✅ Registered new person: {name} with {successful_count} embeddings")
            
            # Reload embeddings cache
            self._load_embeddings_from_db()
            self._prepare_database()
            
            return True
            
        except Exception as e:
            print(f"❌ Error registering person from folder: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def reload_embeddings(self):
        """Reload embeddings from database"""
        print("🔄 Reloading embeddings from database...")
        self.embeddings_cache = {}
        self._load_embeddings_from_db()
        self._prepare_database()
        print("✅ Embeddings reloaded")
    
    def update_threshold(self, new_threshold: float):
        """Update similarity threshold"""
        self.similarity_threshold = new_threshold
        print(f"✅ Similarity threshold updated to {self.similarity_threshold}")
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        total_embeddings = sum(len(data['embeddings']) for data in self.embeddings_cache.values())
        
        return {
            'total_persons': len(self.embeddings_cache),
            'total_embeddings': total_embeddings,
            'similarity_threshold': self.similarity_threshold,
            'gpu_enabled': self.use_cuda
        }
