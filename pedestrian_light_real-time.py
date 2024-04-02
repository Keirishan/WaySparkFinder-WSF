import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load the YOLOv8 model
model = YOLO('pedestrian_light.pt')

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
            if names[int(cls)] == "stop":
                annotator.box_label(box, "", (0, 0, 255))
                cv2.putText(frame, "You can't cross the road", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            elif names[int(cls)] == "go" or names[int(cls)] == "truck" or names[int(cls)] == "bus":
                annotator.box_label(box, "", (0, 255, 0))
                cv2.putText(frame, "Now you can cross the road", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("pedestrian_light_detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
