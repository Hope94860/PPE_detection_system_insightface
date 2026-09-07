"""
Utility functions for face recognition project
"""
import os
import cv2
import numpy as np
from typing import List, Tuple, Dict
import pickle


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array in BGR format
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return img


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an image to file
    
    Args:
        image: Image as numpy array
        output_path: Path to save the image
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files from a directory
    
    Args:
        directory: Directory path
        extensions: List of valid extensions (default: ['.jpg', '.jpeg', '.png'])
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)


def draw_face_box(image: np.ndarray, bbox: np.ndarray, label: str = None, 
                  confidence: float = None, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw bounding box and label on face
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        label: Label text to display
        confidence: Confidence score
        color: Box color in BGR
        
    Returns:
        Image with drawn box
    """
    img = image.copy()
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label text
    if label:
        text = label
        if confidence is not None:
            text += f" ({confidence:.2f})"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(img, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return img


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, 
                   color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """
    Draw facial landmarks on image
    
    Args:
        image: Input image
        landmarks: Facial landmarks array of shape (N, 2)
        color: Color for landmarks in BGR
        
    Returns:
        Image with drawn landmarks
    """
    img = image.copy()
    for point in landmarks:
        x, y = point.astype(int)
        cv2.circle(img, (x, y), 2, color, -1)
    
    return img


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    # Normalize embeddings
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2)
    
    return float(similarity)


def save_database(database: Dict, filepath: str) -> None:
    """
    Save face database to pickle file
    
    Args:
        database: Dictionary containing face embeddings
        filepath: Path to save the database
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(database, f)
    print(f"Database saved to {filepath}")


def load_database(filepath: str) -> Dict:
    """
    Load face database from pickle file
    
    Args:
        filepath: Path to the database file
        
    Returns:
        Dictionary containing face embeddings
    """
    if not os.path.exists(filepath):
        print(f"Database not found at {filepath}, returning empty database")
        return {}
    
    with open(filepath, 'rb') as f:
        database = pickle.load(f)
    
    print(f"Database loaded from {filepath} with {len(database)} identities")
    return database


def create_video_writer(output_path: str, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """
    Create a video writer object
    
    Args:
        output_path: Path to save the video
        fps: Frames per second
        frame_size: (width, height) of the video
        
    Returns:
        VideoWriter object
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    return writer


def resize_image(image: np.ndarray, max_size: int = 1920) -> np.ndarray:
    """
    Resize image if it's too large while maintaining aspect ratio
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
    
    if height > width:
        new_height = max_size
        new_width = int(width * (max_size / height))
    else:
        new_width = max_size
        new_height = int(height * (max_size / width))
    
    resized = cv2.resize(image, (new_width, new_height))
    return resized
