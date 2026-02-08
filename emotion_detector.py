import cv2
import numpy as np

class EmotionDetector:
    """Simple emotion detection based on facial features"""
    
    def __init__(self):
        # Load Haar Cascades for facial features
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
    
    def detect_emotion(self, face_roi):
        """
        Detect emotion from face ROI using simple heuristics
        Returns: emotion string with emoji
        """
        if face_roi.size == 0:
            return "😐 Neutral"
        
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_roi
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            gray_face, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # Detect smile
        smiles = self.smile_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        
        # Calculate brightness (can indicate surprise/fear)
        brightness = np.mean(gray_face)
        
        # Simple rule-based emotion detection
        if len(smiles) > 0:
            return "😊 Happy"
        elif len(eyes) == 0:
            return "😲 Surprised"
        elif brightness < 80:
            return "😢 Sad"
        elif brightness > 150:
            return "😲 Surprised"
        else:
            # Use face aspect ratio for more emotions
            h, w = gray_face.shape
            aspect_ratio = h / w if w > 0 else 1
            
            if aspect_ratio > 1.3:
                return "😮 Surprised"
            elif aspect_ratio < 1.1:
                return "😠 Angry"
            else:
                return "😐 Neutral"
    
    def get_emotion_color(self, emotion):
        """Get color for emotion"""
        if "Happy" in emotion or "😊" in emotion:
            return (0, 255, 0)  # Green
        elif "Sad" in emotion or "😢" in emotion:
            return (255, 0, 0)  # Blue
        elif "Angry" in emotion or "😠" in emotion:
            return (0, 0, 255)  # Red
        elif "Surprised" in emotion or "😲" in emotion or "😮" in emotion:
            return (0, 255, 255)  # Yellow
        else:
            return (200, 200, 200)  # Gray for neutral
