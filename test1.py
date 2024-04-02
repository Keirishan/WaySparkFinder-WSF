import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from gtts import gTTS
import os

model = YOLO('yolov8n.pt')

# Path to Video
video_path = "test_data\Test1.mp4"
if not Path(video_path).exists():
    raise FileNotFoundError(f"Source path {video_path} does not exist.")

names = model.model.names
cap = cv2.VideoCapture(video_path)

# Set the desired frame width and height
frame_width = 1920
frame_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

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
                message = "Person detected"
                cv2.putText(frame, message, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 102, 188), 2)
                tts = gTTS(message, lang='en')
                tts.save('output.mp3')
                os.system('start output.mp3')
            elif names[int(cls)] == "car" or names[int(cls)] == "truck" or names[int(cls)] == "bus":
                annotator.box_label(box, "", (0, 255, 0))
                message = "Vehicle detected"
                cv2.putText(frame, message, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                tts = gTTS(message, lang='en')
                tts.save('output.mp3')
                os.system('start output.mp3')
            elif names[int(cls)] == "potted plant":
                annotator.box_label(box, "", (255, 0, 0))
                message = "Plant detected"
                cv2.putText(frame, message, (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                tts = gTTS(message, lang='en')
                tts.save('output.mp3')
                os.system('start output.mp3')
            elif names[int(cls)] == "traffic light":
                annotator.box_label(box, "", (0, 0, 255))
                message = "Traffic light detected"
                cv2.putText(frame, message, (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                tts = gTTS(message, lang='en')
                tts.save('output.mp3')
                os.system('start output.mp3')

        cv2.imshow("Object detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()