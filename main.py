from face_recognition import FaceRecognition
from pathlib import Path

def main():
    print("=" * 50)
    print("Face Recognition System")
    print("=" * 50)
    
    fr = FaceRecognition()
    
    # Check for registered faces
    if not fr.register_faces():
        print("\n💡 Create a 'faces' directory and add images")
        print("   Name files as: person_name.jpg")
        return
    
    print("\n" + "=" * 50)
    print("Options:")
    print("1. Start webcam recognition")
    print("2. Process an image")
    print("=" * 50)
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        fr.process_webcam()
    elif choice == "2":
        img_path = input("Enter image path: ").strip()
        fr.process_image(img_path)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
