from ultralytics import YOLO      # Import YOLOv8 model from Ultralytics
import cv2                        # OpenCV for video capture and drawing
import time                       # Time module for calculating FPS

# Load YOLOv8 nano model (smallest and fastest version)
model = YOLO("yolov8n.pt")        # Pretrained on COCO dataset

# Open webcam or IP camera stream
cap = cv2.VideoCapture("The Fate of the Furious ｜ Harpooning Dom's Car.mp4")  # Replace with 0 for local webcam

# Start the main loop for real-time detection
while True:
    start_time = time.time()      # Record start time for FPS calculation

    ret, frame = cap.read()       # Read a frame from the video stream
    if not ret:
        break                     # Exit loop if frame not received

    # Resize frame to reduce processing load and improve speed
    frame = cv2.resize(frame, (640, 400))  # Resize to 640x400 pixels

    # Run YOLOv8 detection on the frame
    results = model(frame)        # Returns detection results
    detections = results[0].boxes.data  # Extract bounding box data

    count = 0                     # Initialize person counter

    # Loop through each detected object
    for box in detections:
        x1, y1, x2, y2, score, cls_id = box[:6]  # Get box coordinates, confidence, and class ID
        cls_id = int(cls_id)      # Convert class ID to integer

        # Check if the detected object is a person
        if model.names[cls_id] == 'person':
            count += 1            # Increment person count

            # Convert coordinates to integers for drawing
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Draw a thin green bounding box around the person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Draw label with person number and confidence score
            label = f"Person {count} ({score:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Calculate FPS based on time taken for one loop
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Display total people count on screen
    cv2.putText(frame, f"People Count: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    # Display FPS on screen
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Show the final frame with annotations
    cv2.imshow("ESP People Counter", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video stream and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
