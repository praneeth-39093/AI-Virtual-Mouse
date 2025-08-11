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

📂 Project Structure
bash
Copy
Edit
AI-Virtual-Mouse/
│── README.md                 # Project documentation
│── requirements.txt          # Dependencies
│── main.py                   # Entry point
│── modules/
│     ├── vision_tracking.py      # Hand tracking (Member 1)
│     ├── gesture_recognition.py  # Gesture detection (Member 2)
│     ├── mouse_control.py        # Mouse actions (Member 3)
│     ├── integration.py          # System integration (Member 4)
│── docs/
      ├── design_doc.md
      ├── user_manual.pdf
      
🚀 Installation & Setup

1️⃣ Clone the repository

git clone https://github.com/your-username/AI-Virtual-Mouse.git
cd AI-Virtual-Mouse

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Run the project
python main.py

🎮 How It Works
The webcam captures the live video feed.
MediaPipe detects 21 hand landmarks.
Gesture recognition maps these landmarks to predefined gestures.
PyAutoGUI converts gestures into mouse actions.
The cursor moves or performs clicks/scrolls without physical contact.

👨‍💻 Team Members & Roles

Student ID	Role
2200031331	Computer Vision & Hand Landmark Detection
2200031483	Gesture Recognition & Mapping
2200032540	Mouse Action Control & PyAutoGUI Integration
2200039093	System Integration, Testing & Documentation

📌 Applications
Assistive technology for people with mobility impairments.
Touchless interfaces in healthcare, public kiosks, and smart homes.
Interactive displays and VR/AR control.

📜 License
This project is for academic and research purposes under the Department of Computer Science and Engineering, Koneru Lakshmaiah Education Foundation.

🏆 Acknowledgments
MediaPipe
OpenCV
PyAutoGUI
