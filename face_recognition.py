import cv2
import numpy as np
from pathlib import Path
import urllib.request
import os
from datetime import datetime
import json
from emotion_detector import EmotionDetector

class FaceRecognition:
    def __init__(self):
        self.known_faces = {}
        self.face_detector = None
        self.face_recognizer = None
        self.emotion_detector = EmotionDetector()
        self.face_cascade = None
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'recognized_faces': 0,
            'unknown_faces': 0,
            'session_start': datetime.now()
        }
        
        # Settings
        self.settings = {
            'show_landmarks': True,
            'show_confidence': True,
            'show_stats': True,
            'show_emotion': True,
            'alert_unknown': True,
            'detection_threshold': 0.5,
            'recognition_threshold': 100
        }
        
        # Emotion detection (simple rule-based)
        self.emotions = ['😊 Happy', '😐 Neutral', '😢 Sad', '😠 Angry', '😲 Surprised']
        
        # Recording
        self.is_recording = False
        self.video_writer = None
        
        self.setup_models()
    
    def setup_models(self):
        """Download and setup face detection and recognition models"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Face detection model (Caffe)
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        prototxt_path = models_dir / "deploy.prototxt"
        caffemodel_path = models_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        
        # Download if not exists
        if not prototxt_path.exists():
            print("📥 Downloading face detection model (prototxt)...")
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
        
        if not caffemodel_path.exists():
            print("📥 Downloading face detection model (weights - ~10MB)...")
            urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
        
        # Load face detector
        self.face_detector = cv2.dnn.readNetFromCaffe(
            str(prototxt_path), 
            str(caffemodel_path)
        )
        
        # Use OpenCV's face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Load Haar Cascade for landmarks
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print("✓ Models loaded successfully")
    
    def detect_faces(self, image):
        """Detect faces in an image"""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.settings['detection_threshold']:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                faces.append((x1, y1, x2, y2, confidence))
        
        return faces
    
    def register_faces(self):
        """Load and train on faces from the faces directory"""
        faces_dir = Path("faces")
        if not faces_dir.exists():
            print(f"⚠️  'faces' directory not found")
            return False
        
        face_files = list(faces_dir.glob("*.*"))
        face_files = [f for f in face_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if not face_files:
            print(f"⚠️  No images found in 'faces'")
            return False
        
        print(f"\n📁 Registering {len(face_files)} face(s)...")
        
        faces_data = []
        labels_data = []
        label_names = {}
        current_label = 0
        
        for img_path in face_files:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"✗ Could not load {img_path.name}")
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected_faces = self.detect_faces(image)
            
            if detected_faces:
                x1, y1, x2, y2, conf = detected_faces[0]
                face_roi = gray[y1:y2, x1:x2]
                
                if face_roi.size > 0:
                    face_roi = cv2.resize(face_roi, (200, 200))
                    faces_data.append(face_roi)
                    labels_data.append(current_label)
                    
                    person_name = img_path.stem
                    label_names[current_label] = person_name
                    self.known_faces[current_label] = person_name
                    
                    print(f"✓ Registered: {person_name}")
                    current_label += 1
            else:
                print(f"✗ No face detected in {img_path.name}")
        
        if faces_data:
            self.face_recognizer.train(faces_data, np.array(labels_data))
            return True
        
        return False
    
    def draw_stats_panel(self, frame):
        """Draw statistics panel on frame"""
        if not self.settings['show_stats']:
            return frame  # Return original frame if stats disabled
        
        h, w = frame.shape[:2]
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        # Session duration
        duration = datetime.now() - self.stats['session_start']
        duration_str = str(duration).split('.')[0]
        
        # Draw stats
        cv2.putText(panel, f"Session: {duration_str}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(panel, f"Total Detections: {self.stats['total_detections']}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        cv2.putText(panel, f"Recognized: {self.stats['recognized_faces']}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        cv2.putText(panel, f"Unknown: {self.stats['unknown_faces']}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
        
        # Recording indicator
        if self.is_recording:
            cv2.circle(panel, (w - 30, 25), 10, (0, 0, 255), -1)
            cv2.putText(panel, "REC", (w - 70, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Combine with frame
        combined = np.vstack([panel, frame])
        return combined
    
    def draw_confidence_bar(self, frame, x, y, confidence, max_confidence=100):
        """Draw confidence meter - Lower number = Better match!"""
        if not self.settings['show_confidence']:
            return
        
        bar_width = 100
        bar_height = 10
        
        # Lower confidence = better match, so invert for display
        # 0 = perfect match (100% fill), 100 = poor match (0% fill)
        fill_width = int((1 - min(confidence, max_confidence) / max_confidence) * bar_width)
        
        # Background
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
        
        # Fill (green = good match, red = poor match)
        if fill_width > 66:
            color = (0, 255, 0)  # Green - excellent match
        elif fill_width > 33:
            color = (0, 255, 255)  # Yellow - okay match
        else:
            color = (0, 0, 255)  # Red - poor match
        
        cv2.rectangle(frame, (x, y), (x + fill_width, y + bar_height), color, -1)
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 1)
        
        # Add label
        cv2.putText(frame, "Match:", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def save_screenshot(self, frame):
        """Save screenshot"""
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = screenshots_dir / f"screenshot_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def toggle_recording(self, frame):
        """Toggle video recording"""
        if not self.is_recording:
            # Start recording
            recordings_dir = Path("recordings")
            recordings_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = recordings_dir / f"recording_{timestamp}.avi"
            
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(str(filename), fourcc, 20.0, (w, h))
            self.is_recording = True
            print(f"🎥 Recording started: {filename}")
        else:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.is_recording = False
            print("⏹️  Recording stopped")

    def process_webcam(self):
        """Real-time face recognition from webcam with enhanced features"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return
        
        print("\n🎥 Starting enhanced webcam mode...")
        print("\n⌨️  Controls:")
        print("   Q - Quit")
        print("   S - Screenshot")
        print("   R - Toggle Recording")
        print("   L - Toggle Landmarks")
        print("   C - Toggle Confidence Bars")
        print("   T - Toggle Stats Panel")
        print("   E - Toggle Emotion Detection")
        print("   A - Toggle Unknown Alerts")
        print("   + - Increase Detection Threshold")
        print("   - - Decrease Detection Threshold\n")
        print("💡 Confidence Bar: Green (full) = Perfect Match, Red (empty) = Poor Match\n")
        
        face_tracker = {}  # Track faces across frames
        next_face_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            original_frame = frame.copy()
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Update stats
            self.stats['total_detections'] = len(faces)
            
            recognized_count = 0
            unknown_count = 0
            
            for idx, (x1, y1, x2, y2, conf) in enumerate(faces):
                # Extract face for recognition
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_roi = gray[y1:y2, x1:x2]
                
                if face_roi.size > 0:
                    face_roi_resized = cv2.resize(face_roi, (200, 200))
                    
                    # Recognize face
                    label, confidence = self.face_recognizer.predict(face_roi_resized)
                    
                    # Determine name and color
                    if confidence < self.settings['recognition_threshold']:
                        name = self.known_faces.get(label, "Unknown")
                        color = (0, 255, 0)
                        recognized_count += 1
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)
                        unknown_count += 1
                        
                        # Alert for unknown faces
                        if self.settings['alert_unknown']:
                            cv2.putText(frame, "⚠️ UNKNOWN PERSON", (10, frame.shape[0] - 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Draw fancy rectangle with rounded corners effect
                    thickness = 3
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw corner accents
                    corner_length = 20
                    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness + 2)
                    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness + 2)
                    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness + 2)
                    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness + 2)
                    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness + 2)
                    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness + 2)
                    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness + 2)
                    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness + 2)
                    
                    # Name tag background
                    tag_height = 35
                    cv2.rectangle(frame, (x1, y2), (x2, y2 + tag_height), color, -1)
                    
                    # Name text
                    text = f"{name} #{idx+1}"
                    cv2.putText(frame, text, (x1 + 6, y2 + 23), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Detect emotion
                    emotion = "😐 Neutral"
                    if self.settings['show_emotion']:
                        face_roi_color = frame[y1:y2, x1:x2]
                        emotion = self.emotion_detector.detect_emotion(face_roi_color)
                        emotion_color = self.emotion_detector.get_emotion_color(emotion)
                        
                        # Draw emotion above name
                        cv2.putText(frame, emotion, (x1, y1 - 35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
                    
                    # Confidence bar
                    if self.settings['show_confidence']:
                        self.draw_confidence_bar(frame, x1, y1 - 20, confidence)
                        cv2.putText(frame, f"{confidence:.0f}", (x1 + 105, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Draw landmarks
                    if self.settings['show_landmarks']:
                        # Simple landmark points (eyes, nose, mouth approximation)
                        face_width = x2 - x1
                        face_height = y2 - y1
                        
                        # Eyes
                        left_eye = (x1 + int(face_width * 0.3), y1 + int(face_height * 0.35))
                        right_eye = (x1 + int(face_width * 0.7), y1 + int(face_height * 0.35))
                        cv2.circle(frame, left_eye, 3, (255, 255, 0), -1)
                        cv2.circle(frame, right_eye, 3, (255, 255, 0), -1)
                        
                        # Nose
                        nose = (x1 + int(face_width * 0.5), y1 + int(face_height * 0.55))
                        cv2.circle(frame, nose, 3, (255, 255, 0), -1)
                        
                        # Mouth
                        mouth = (x1 + int(face_width * 0.5), y1 + int(face_height * 0.75))
                        cv2.circle(frame, mouth, 3, (255, 255, 0), -1)
            
            # Update session stats
            self.stats['recognized_faces'] = recognized_count
            self.stats['unknown_faces'] = unknown_count
            
            # Add stats panel
            display_frame = self.draw_stats_panel(frame)
            
            # Write to video if recording
            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)
            
            cv2.imshow('Face Recognition System - Enhanced', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_screenshot(original_frame)
            elif key == ord('r'):
                self.toggle_recording(frame)
            elif key == ord('l'):
                self.settings['show_landmarks'] = not self.settings['show_landmarks']
                print(f"Landmarks: {'ON' if self.settings['show_landmarks'] else 'OFF'}")
            elif key == ord('c'):
                self.settings['show_confidence'] = not self.settings['show_confidence']
                print(f"Confidence bars: {'ON' if self.settings['show_confidence'] else 'OFF'}")
            elif key == ord('t'):
                self.settings['show_stats'] = not self.settings['show_stats']
                print(f"Stats panel: {'ON' if self.settings['show_stats'] else 'OFF'}")
            elif key == ord('e'):
                self.settings['show_emotion'] = not self.settings['show_emotion']
                print(f"Emotion detection: {'ON' if self.settings['show_emotion'] else 'OFF'}")
            elif key == ord('a'):
                self.settings['alert_unknown'] = not self.settings['alert_unknown']
                print(f"Unknown alerts: {'ON' if self.settings['alert_unknown'] else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                self.settings['detection_threshold'] = min(0.9, self.settings['detection_threshold'] + 0.05)
                print(f"Detection threshold: {self.settings['detection_threshold']:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.settings['detection_threshold'] = max(0.1, self.settings['detection_threshold'] - 0.05)
                print(f"Detection threshold: {self.settings['detection_threshold']:.2f}")
        
        # Cleanup
        if self.is_recording:
            self.toggle_recording(frame)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path):
        """Process a single image with enhanced visualization"""
        if not Path(image_path).exists():
            print(f"✗ Image not found: {image_path}")
            return
        
        print(f"\n🔍 Analyzing {image_path}...")
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"✗ Could not load image")
            return
        
        faces = self.detect_faces(image)
        
        if not faces:
            print("✗ No faces detected")
            return
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        print(f"\n✓ Found {len(faces)} face(s):\n")
        
        for idx, (x1, y1, x2, y2, conf) in enumerate(faces):
            face_roi = gray[y1:y2, x1:x2]
            
            if face_roi.size > 0:
                face_roi_resized = cv2.resize(face_roi, (200, 200))
                label, confidence = self.face_recognizer.predict(face_roi_resized)
                
                if confidence < self.settings['recognition_threshold']:
                    name = self.known_faces.get(label, "Unknown")
                    color = (0, 255, 0)
                    print(f"  Face {idx+1}: {name} (confidence: {confidence:.1f})")
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    print(f"  Face {idx+1}: Unknown (confidence: {confidence:.1f})")
                
                # Draw on image
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(image, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(image, f"{name} #{idx+1}", (x1 + 6, y2 - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Save annotated image
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"annotated_{Path(image_path).name}"
        cv2.imwrite(str(output_path), image)
        print(f"\n💾 Annotated image saved: {output_path}")
        
        # Display
        cv2.imshow('Face Recognition Result', image)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
