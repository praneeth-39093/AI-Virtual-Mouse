AI-Powered Virtual Mouse Interface
📌 Project Overview
This project implements an AI-powered virtual mouse system that allows users to control a computer cursor using real-time hand gesture recognition and computer vision.
By leveraging a webcam and machine learning techniques, the system enables touch-free, intuitive, and hygienic interaction with digital devices.

🎯 Key Features

Real-time hand tracking using MediaPipe and OpenCV.
Detects 21 key hand landmarks for accurate gesture recognition.
Supports gesture-based mouse control:
Cursor movement
Single-click, double-click
Drag-and-drop
Scrolling
Multi-finger gestures
Contactless operation — improves hygiene and accessibility.
Works with any standard webcam — no special hardware required.

🛠️ Tech Stack

Programming Language: Python
Libraries & Frameworks:
OpenCV – image processing
MediaPipe – hand tracking
PyAutoGUI – mouse control
NumPy – numerical processing


      
🚀 Installation & Setup

1️⃣ Clone the repository: 

git clone https://github.com/your-username/AI-Virtual-Mouse.git
cd AI-Virtual-Mouse

2️⃣ Install dependencies: 

pip install -r requirements.txt

3️⃣ Run the project: 
python main.py

🎮 How It Works
The webcam captures the live video feed.
MediaPipe detects 21 hand landmarks.
Gesture recognition maps these landmarks to predefined gestures.
PyAutoGUI converts gestures into mouse actions.
The cursor moves or performs clicks/scrolls without physical contact.

🏆 Acknowledgments
MediaPipe
OpenCV
PyAutoGUI
