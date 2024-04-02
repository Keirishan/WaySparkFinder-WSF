import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import pyttsx3

# Load the YOLOv8 model
model = YOLO('newbest.pt')

# Path to Video
video_path = "test_data\Test_object.MOV"
if not Path(video_path).exists():
    raise FileNotFoundError(f"Source path {video_path} does not exist.")

names = model.model.names
cap = cv2.VideoCapture(video_path)

# Set frame width and height (adjust as needed)
frame_width = 1920
frame_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Initialize TTS engine
engine = pyttsx3.init()

while True:
    success, frame = cap.read()

    if success:
        results = model.predict(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.tolist()
        confidences = results[0].boxes.conf.tolist()
        annotator = Annotator(frame, line_width=2, example=str(names))

        for box, cls, conf in zip(boxes, classes, confidences):
            if names[int(cls)] == "person":
                annotator.box_label(box, "", (255, 102, 188))
                cv2.putText(frame, "Person detected", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 102, 188), 2)
                engine.say("Person detected")
            elif names[int(cls)] == "car" or names[int(cls)] == "truck" or names[int(cls)] == "bus":
                annotator.box_label(box, "", (0, 255, 0))
                cv2.putText(frame, "Vehicle detected", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                engine.say("Vehicle detected")
            elif names[int(cls)] == "potted plant" or names[int(cls)] == "tree":
                annotator.box_label(box, "", (255, 0, 0))
                cv2.putText(frame, "Plant detected", (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                engine.say("Plant detected")
            elif names[int(cls)] == "traffic light" or names[int(cls)] == "traffic-light":
                annotator.box_label(box, "", (255, 0, 0))
                cv2.putText(frame, "Traffic light detected", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                engine.say("traffic light detected")
            elif names[int(cls)] == "post":
                annotator.box_label(box, "", (0, 0, 255))
                cv2.putText(frame, "post detected", (1400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 23, 255), 2)
                engine.say("post detected")

        cv2.imshow("Object detection", frame)
        engine.runAndWait()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
engine.shutdown()