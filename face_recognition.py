import cv2
import numpy as np
from pathlib import Path
import urllib.request
import os

class FaceRecognition:
    def __init__(self):
        self.known_faces = {}
        self.face_detector = None
        self.face_recognizer = None
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
            
            if confidence > 0.5:
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

    def process_webcam(self):
        """Real-time face recognition from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return
        
        print("\n🎥 Starting webcam... Press 'q' to quit\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            for (x1, y1, x2, y2, conf) in faces:
                # Extract face for recognition
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_roi = gray[y1:y2, x1:x2]
                
                if face_roi.size > 0:
                    face_roi = cv2.resize(face_roi, (200, 200))
                    
                    # Recognize face
                    label, confidence = self.face_recognizer.predict(face_roi)
                    
                    # Lower confidence = better match (it's actually distance)
                    if confidence < 100:
                        name = self.known_faces.get(label, "Unknown")
                        color = (0, 255, 0)
                        text = f"{name} ({confidence:.1f})"
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)
                        text = "Unknown"
                    
                    # Draw rectangle and name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                    cv2.putText(frame, text, (x1 + 6, y2 - 6), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition - Press Q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path):
        """Process a single image"""
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
        
        for idx, (x1, y1, x2, y2, conf) in enumerate(faces):
            face_roi = gray[y1:y2, x1:x2]
            
            if face_roi.size > 0:
                face_roi = cv2.resize(face_roi, (200, 200))
                label, confidence = self.face_recognizer.predict(face_roi)
                
                if confidence < 100:
                    name = self.known_faces.get(label, "Unknown")
                    print(f"✓ Face {idx+1}: {name} (confidence: {confidence:.1f})")
                else:
                    print(f"✓ Face {idx+1}: Unknown")
