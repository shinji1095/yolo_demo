import os
import cv2
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

base = r'C:\Users\shinj\Projects\Python\Sasaki\share\inputs\dataset'
name = 'data47'
video = os.path.join(base, name, 'movie.mp4')
model_path = r'vidvipo_yolov8n_2023-05-19.pt'
print(video)


# Load the YOLOv8 model
model = YOLO(model_path)

# Open the video file
cap = cv2.VideoCapture(video)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()