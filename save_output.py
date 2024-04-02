import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load the YOLOv8 model
model = YOLO('pathway_pedestrian.pt')

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

# Define the codec and create a VideoWriter object
output_video_path = "output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

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
                cv2.putText(frame, "pathway Detected", (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 42, 4), 2)
            elif names[int(cls)] == "pedestrian_crossing" or names[int(cls)] == "truck" or names[int(cls)] == "bus":
                annotator.box_label(box, "", (0, 128, 0))
                cv2.putText(frame, "pedestrian Detected", (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 0), 2)

        out.write(frame)
        cv2.imshow("Object detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to: {output_video_path}")
