# YOLOv8 People Counter 👁️‍🗨️

This project uses the **Ultralytics YOLOv8** model with **OpenCV** to detect and count people in a live webcam feed or video file in real-time.

## 🚀 Features
- Real-time object detection using **YOLOv8 Nano (yolov8n.pt)**
- Automatically counts the number of people visible in each frame
- Displays bounding boxes, confidence scores, and FPS
- Works with both **webcams** and **video files**

---

## 🧠 Requirements
Install the required Python libraries:
```bash
pip install ultralytics opencv-python
```

---

## 🧩 How It Works
1. Loads the **YOLOv8 Nano** model (smallest & fastest version).
2. Reads frames from a webcam or video file.
3. Performs object detection on each frame.
4. Draws bounding boxes and counts the number of detected persons.
5. Displays the total count and FPS in real-time.

---

## 💻 Usage
1. Place your video file (e.g., `The Fate of the Furious ｜ Harpooning Dom's Car.mp4`) in the same directory.
2. Run the script:
```bash
python people_counter.py
```
3. Press **'q'** to quit the window.

> 💡 To use a webcam instead of a video file, replace:
> ```python
> cap = cv2.VideoCapture("The Fate of the Furious ｜ Harpooning Dom's Car.mp4")
> ```
> with:
> ```python
> cap = cv2.VideoCapture(0)
> ```

---

## 🧾 Code Overview
```python
from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO("yolov8n.pt")

# Capture video
cap = cv2.VideoCapture("video.mp4")

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 400))
    results = model(frame)
    detections = results[0].boxes.data

    count = 0
    for box in detections:
        x1, y1, x2, y2, score, cls_id = box[:6]
        cls_id = int(cls_id)
        if model.names[cls_id] == 'person':
            count += 1
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = f"Person {count} ({score:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    fps = 1 / (time.time() - start)
    cv2.putText(frame, f"People Count: {count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.imshow("ESP People Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📸 Output Example
- Green boxes highlight detected people.
- Real-time **People Count** and **FPS** are displayed.

---

## 🧑‍💻 Author
**Amit Kadam**  
📧 kadamamit462@gmail.com  
📍 Bhalki, India  

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
