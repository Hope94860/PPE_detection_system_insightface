"""
Face Recognizer - Recognize faces using pre-built database
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from face_encoder import FaceEncoder
from utils import load_database, cosine_similarity


class FaceRecognizer:
    """
    Face recognizer that matches detected faces against a database
    """
    
    def __init__(self, database_path: str, model_name: str = 'buffalo_l', 
                 similarity_threshold: float = 0.4, use_gpu: bool = True):
        """
        Initialize the face recognizer
        
        Args:
            database_path: Path to the face database pickle file
            model_name: Name of the InsightFace model
            similarity_threshold: Minimum similarity score to consider a match (0-1)
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        print("Initializing FaceRecognizer...")
        
        # Initialize encoder with GPU support
        self.encoder = FaceEncoder(model_name=model_name, use_gpu=use_gpu)
        
        # Load database
        self.database = load_database(database_path)
        self.similarity_threshold = similarity_threshold
        
        # Prepare database embeddings for faster matching
        self._prepare_database()
        
        print(f"FaceRecognizer initialized with {len(self.database)} identities")
        print(f"Similarity threshold: {self.similarity_threshold}")
    
    def _prepare_database(self):
        """
        Prepare database for efficient matching
        """
        self.db_names = []
        self.db_embeddings = []
        
        for name, data in self.database.items():
            embeddings = data['embeddings']
            for embedding in embeddings:
                self.db_names.append(name)
                self.db_embeddings.append(embedding)
        
        if len(self.db_embeddings) > 0:
            self.db_embeddings = np.array(self.db_embeddings)
            print(f"Prepared {len(self.db_embeddings)} embeddings for matching")
    
    def recognize_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face from its embedding
        
        Args:
            embedding: Face embedding to recognize
            
        Returns:
            Tuple of (name, similarity_score) or (None, 0.0) if no match
        """
        if len(self.db_embeddings) == 0:
            return None, 0.0
        
        # Calculate similarities with all database embeddings
        similarities = np.dot(self.db_embeddings, embedding)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        # Check if similarity meets threshold
        if best_similarity >= self.similarity_threshold:
            return self.db_names[best_idx], float(best_similarity)
        else:
            return None, float(best_similarity)
    
    def recognize_faces_in_image(self, image: np.ndarray) -> List[Dict]:
        """
        Detect and recognize all faces in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of dictionaries containing recognition results for each face
        """
        # Get all face embeddings
        face_results = self.encoder.get_all_face_embeddings(image)
        
        recognition_results = []
        for embedding, face_info in face_results:
            # Recognize the face
            name, similarity = self.recognize_face(embedding)
            
            # Prepare result
            result = {
                'bbox': face_info['bbox'],
                'landmarks': face_info['landmarks'],
                'det_score': face_info['det_score'],
                'name': name if name else 'Unknown',
                'similarity': similarity,
                'is_recognized': name is not None
            }
            
            recognition_results.append(result)
        
        return recognition_results
    
    def recognize_from_path(self, image_path: str) -> List[Dict]:
        """
        Recognize faces from an image file path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of recognition results
        """
        from utils import load_image
        image = load_image(image_path)
        return self.recognize_faces_in_image(image)
    
    def update_threshold(self, new_threshold: float):
        """
        Update the similarity threshold
        
        Args:
            new_threshold: New threshold value (0-1)
        """
        self.similarity_threshold = new_threshold
        print(f"Similarity threshold updated to {self.similarity_threshold}")
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the loaded database
        
        Returns:
            Dictionary with database statistics
        """
        stats = {
            'num_identities': len(self.database),
            'total_embeddings': len(self.db_embeddings),
            'identities': {}
        }
        
        for name, data in self.database.items():
            stats['identities'][name] = {
                'num_images': data['num_images']
            }
        
        return stats


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Check if database exists
    database_path = "../database/faces.pkl"
    
    try:
        recognizer = FaceRecognizer(database_path, similarity_threshold=0.4)
        
        # Print database stats
        stats = recognizer.get_database_stats()
        print("\nDatabase Statistics:")
        print(f"  Total identities: {stats['num_identities']}")
        print(f"  Total embeddings: {stats['total_embeddings']}")
        print("\nIdentities:")
        for name, info in stats['identities'].items():
            print(f"  - {name}: {info['num_images']} images")
        
        # Test recognition on a sample image if provided
        if len(sys.argv) > 1:
            test_image_path = sys.argv[1]
            print(f"\nTesting recognition on: {test_image_path}")
            results = recognizer.recognize_from_path(test_image_path)
            
            print(f"\nDetected {len(results)} face(s):")
            for i, result in enumerate(results):
                print(f"  Face {i+1}:")
                print(f"    Name: {result['name']}")
                print(f"    Similarity: {result['similarity']:.3f}")
                print(f"    Detection score: {result['det_score']:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease build the database first using face_encoder.py")
