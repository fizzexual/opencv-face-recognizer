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
        
        # Resize for better detection
        gray_face = cv2.resize(gray_face, (200, 200))
        
        # Detect eyes with more lenient parameters
        eyes = self.eye_cascade.detectMultiScale(
            gray_face, 
            scaleFactor=1.05, 
            minNeighbors=3,
            minSize=(15, 15)
        )
        
        # Detect smile with more sensitive parameters
        smiles = self.smile_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(20, 20)
        )
        
        # Calculate brightness and contrast
        brightness = np.mean(gray_face)
        contrast = np.std(gray_face)
        
        # Analyze lower half of face for mouth/smile
        lower_half = gray_face[100:, :]
        lower_brightness = np.mean(lower_half)
        
        # Analyze upper half for eyes
        upper_half = gray_face[:100, :]
        upper_brightness = np.mean(upper_half)
        
        # More sophisticated emotion detection
        if len(smiles) > 0:
            # Strong smile detected
            return "😊 Happy"
        elif lower_brightness > upper_brightness + 10:
            # Mouth area brighter (possible smile/open mouth)
            return "😊 Happy"
        elif len(eyes) < 2 and contrast > 40:
            # Eyes not clearly visible, high contrast
            return "😠 Angry"
        elif brightness < 70:
            # Very dark face (squinting/sad)
            return "😢 Sad"
        elif len(eyes) == 0 or len(eyes) > 3:
            # Eyes detection failed or too many (wide eyes)
            return "😲 Surprised"
        elif contrast < 30:
            # Low contrast (flat expression)
            return "😐 Neutral"
        else:
            # Check face proportions
            h, w = gray_face.shape
            
            # Analyze vertical thirds
            top_third = gray_face[:66, :]
            mid_third = gray_face[66:133, :]
            bot_third = gray_face[133:, :]
            
            top_bright = np.mean(top_third)
            mid_bright = np.mean(mid_third)
            bot_bright = np.mean(bot_third)
            
            if bot_bright < mid_bright - 5:
                return "😢 Sad"
            elif bot_bright > mid_bright + 5:
                return "😊 Happy"
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
