from ultralytics import YOLO

model = YOLO('newbest.pt')

results = model.predict(source = 'test_data\Test_pedestrian_status.MOV', show=True)

print(results)