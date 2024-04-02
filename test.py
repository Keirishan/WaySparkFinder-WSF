import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from gtts import gTTS
import pygame
import os
import time

# Initialize pygame mixer
pygame.mixer.init()

model = YOLO('pathway_pedestrian.pt')

# Path to Video
video_path = "test_data\pedestrian.jpg"
if not Path(video_path).exists():
    raise FileNotFoundError(f"Source path {video_path} does not exist.")

names = model.model.names
cap = cv2.VideoCapture(video_path)

def speak(message):
    tts = gTTS(text=message, lang='en')
    output_path = "speach\output.mp3"
    tts.save(output_path)
    pygame.mixer.music.load(output_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.predict(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.tolist()
        confidences = results[0].boxes.conf.tolist()
        annotator = Annotator(frame, line_width=2, example=str(names))

        for box, cls, conf in zip(boxes, classes, confidences):
            if names[int(cls)] == "pathway":
                annotator.box_label(box, "", (255, 42, 4))
                cv2.putText(frame, "pathway Detected", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 42, 4), 2)
                time.sleep(5)
                speak("Pathway detected.")
            if names[int(cls)] == "pedestrian_crossing":
                annotator.box_label(box, "", (0, 128, 0))
                cv2.putText(frame, "pedestrian Detected", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2)
                time.sleep(5)
                speak("Pedestrian detected.")

        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
