from collections import defaultdict
import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO("C:/Users/User/Documents/zebra_fish/YOLOV8X/weights/best.pt")
video_path = "C:/Users/User/Documents/zebra_fish/source/test_vid30s.mp4"
cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True,tracker="bytetrack.yaml")
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

