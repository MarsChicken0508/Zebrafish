from collections import defaultdict
import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO("C:/Users/User/Documents/zebra_fish/trained_v8n.pt")
video_path = "C:/Users/User/Documents/zebra_fish/source/test_vid30s.mp4"
cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])
next_id = 1  # 初始化id

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        boxes = results[0].boxes.xywh.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        #設置annotated_frame
        annotated_frame = frame.copy()
        #用來紀錄現在id的中心點
        #current_centers = {}

        for box, conf, track_id in zip(boxes, confs, track_ids):
            x, y, w, h = box
            center = (float(x), float(y))

            # 檢查此id是否存在track_history中
            if track_id not in track_history:
                assigned_id = None

                # 將此id與先前的紀錄比對，若第n隻魚的boxes中心與其相差低於50像素，則判斷為同一隻魚
                for existing_id, track in track_history.items():
                    if track:
                        last_center = track[-1]
                        if np.linalg.norm(np.array(center) - np.array(last_center)) < 50:
                            assigned_id = existing_id
                            break

                # 若無，則為新的魚
                if assigned_id is None:
                    assigned_id = next_id
                    next_id += 1

                # 更新track_id
                track_id = assigned_id

            # 更新 track_history
            track_history[track_id].append(center)
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            # 更新current_centers
            #current_centers[track_id] = center

            # 將結果(bounding box,frame,id及conf分數)打印出來
            color = (0, 255, 0)  # 顏色設置
            cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), 
                          (int(x + w / 2), int(y + h / 2)), color, 2)
            cv2.putText(annotated_frame, f"ID: {track_id}", (int(x - w / 2), int(y - h / 2) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(annotated_frame, f"Conf: {conf:.2f}", (int(x - w / 2), int(y - h / 2) - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # 繪製軌跡
            points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=5)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
