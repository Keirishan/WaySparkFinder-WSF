import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import pyttsx3

# Load the YOLOv8 model
model = YOLO('pedestrian_light.pt')

# Path to Video
video_path = "test_data\Test_pedestrian_status.MOV"
if not Path(video_path).exists():
    raise FileNotFoundError(f"Source path {video_path} does not exist.")

names = model.model.names
cap = cv2.VideoCapture(video_path)

# Initialize TTS engine
engine = pyttsx3.init()

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.predict(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.tolist()
        confidences = results[0].boxes.conf.tolist()
        annotator = Annotator(frame, line_width=2, example=str(names))

        for box, cls, conf in zip(boxes, classes, confidences):
            if names[int(cls)] == "stop":
                annotator.box_label(box, "", (255, 0, 0))
                cv2.putText(frame, "You can't cross now", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                engine.say("You can't cross now")
            elif names[int(cls)] == "go":
                annotator.box_label(box, "", (0, 255, 0))
                cv2.putText(frame, "Now you can cross", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                engine.say("Now you can cross")

        cv2.imshow("Pedestrian_light_detection", frame)
        engine.runAndWait()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()