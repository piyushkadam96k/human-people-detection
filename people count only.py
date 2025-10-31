from ultralytics import YOLO      # YOLOv8 model from Ultralytics
import cv2                        # OpenCV for video capture and drawing
import time                       # Time module for FPS calculation

# Load YOLOv8 nano model (fastest version)
model = YOLO("yolov8n.pt")

# Move model to GPU (RTX 2050) for faster inference
model.to('cuda')  # If CUDA is available, this uses your GPU

# Open webcam or IP camera stream
cap = cv2.VideoCapture("The Fate of the Furious ｜ Harpooning Dom's Car.mp4")  # Replace with 0 for local webcam

while True:
    start_time = time.time()  # Start timer for FPS calculation

    ret, frame = cap.read()   # Read a frame from the video stream
    if not ret:
        break  # Exit loop if no frame is received

    # Resize frame to medium resolution for faster processing
    frame = cv2.resize(frame, (640, 400))

    # Run YOLOv8 detection on the frame (on GPU)
    results = model(frame)

    # Extract bounding box data from results
    detections = results[0].boxes.data
    count = 0  # Initialize person counter

    # Loop through each detected object
    for box in detections:
        # Extract box coordinates, confidence score, and class ID
        x1, y1, x2, y2, score, cls_id = box[:6]
        cls_id = int(cls_id)

        # Check if the detected object is a person
        if model.names[cls_id] == 'person':
            count += 1  # Increment person count

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
    cv2.imshow("People Counter", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video stream and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()