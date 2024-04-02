import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load the YOLOv8 model
model = YOLO('pathway_pedestrian.pt')

names = model.model.names
cap = cv2.VideoCapture(0)

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
            if names[int(cls)] == "pathway":
                annotator.box_label(box, "", (255, 42, 4))
                cv2.putText(frame, "pathway Detected", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 42, 4), 2)
            elif names[int(cls)] == "pedestrian_crossing":
                annotator.box_label(box, "", (0, 128, 0))
                cv2.putText(frame, "pedestrian Detected", (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2)

        cv2.imshow("Object detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()