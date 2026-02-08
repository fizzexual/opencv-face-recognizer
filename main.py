from face_recognition import FaceRecognition
from pathlib import Path

def print_banner():
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║        🎭 FACE RECOGNITION SYSTEM - ENHANCED 🎭       ║
    ║                                                       ║
    ║              Powered by OpenCV & Python               ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    print_banner()
    
    fr = FaceRecognition()
    
    # Check for registered faces
    if not fr.register_faces():
        print("\n💡 Create a 'faces' directory and add images")
        print("   Name files as: person_name.jpg")
        return
    
    print("\n" + "=" * 60)
    print("📋 OPTIONS:")
    print("=" * 60)
    print("  1. 🎥 Real-time Webcam Recognition (Enhanced)")
    print("  2. 🖼️  Process Single Image")
    print("  3. ⚙️  View/Edit Settings")
    print("  4. 📊 View Statistics")
    print("  5. ❌ Exit")
    print("=" * 60)
    
    choice = input("\n👉 Enter choice (1-5): ").strip()
    
    if choice == "1":
        fr.process_webcam()
    elif choice == "2":
        img_path = input("📁 Enter image path: ").strip()
        fr.process_image(img_path)
    elif choice == "3":
        print("\n⚙️  Current Settings:")
        print("=" * 60)
        for key, value in fr.settings.items():
            print(f"  {key}: {value}")
        print("=" * 60)
    elif choice == "4":
        print("\n📊 Session Statistics:")
        print("=" * 60)
        for key, value in fr.stats.items():
            print(f"  {key}: {value}")
        print("=" * 60)
    elif choice == "5":
        print("\n👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
