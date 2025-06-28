import os
import cv2
import csv
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker

MODEL_PATH = 'yolov8s.pt'
VIDEO_PATH = 'highway.mp4'
OUTPUT_DIR = 'detected_frames'
OUTPUT_VIDEO = 'output.avi'
CSV_PATH = 'vehicle_log.csv'
FRAME_WIDTH = 1020
FRAME_HEIGHT = 500
RED_LINE_Y = 198
BLUE_LINE_Y = 268
OFFSET = 6
DISTANCE_BETWEEN_LINES = 20

TARGET_CLASSES = ['car', 'bus', 'truck', 'motorcycle']

model = YOLO(MODEL_PATH)
tracker = Tracker()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'XVID'), fps, (FRAME_WIDTH, FRAME_HEIGHT))

down_frame, up_frame = {}, {}
counter_down, counter_up = [], []
final_id_map = {}
vehicle_counter = 1

with open(CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame', 'Vehicle_ID', 'Vehicle_Type', 'Direction', 'Speed_Km_h', 'cx', 'cy', 'x1', 'y1', 'x2', 'y2'])

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    results = model.predict(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()
    boxes_df = pd.DataFrame(detections).astype("float")

    car_boxes = []
    types = {}
    for _, row in boxes_df.iterrows():
        x1, y1, x2, y2, _, cls_id = row[:6]
        cls_id = int(cls_id)
        class_list = model.names
        vehicle_type = class_list[cls_id]
        if vehicle_type in TARGET_CLASSES:
            car_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            types[(int(x1), int(y1), int(x2), int(y2))] = vehicle_type

    tracked = tracker.update(car_boxes)

    with open(CSV_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)

        for x1, y1, x2, y2, tid in tracked:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            vehicle_type = types.get((x1, y1, x2, y2), "unknown")

            if tid not in down_frame and RED_LINE_Y - OFFSET < cy < RED_LINE_Y + OFFSET:
                down_frame[tid] = frame_count

            if tid in down_frame and tid not in counter_down and BLUE_LINE_Y - OFFSET < cy < BLUE_LINE_Y + OFFSET:
                counter_down.append(tid)
                elapsed_frames = frame_count - down_frame[tid]
                elapsed_time = elapsed_frames / fps if fps > 0 else 0.01
                if elapsed_time == 0:
                    continue
                speed = DISTANCE_BETWEEN_LINES / elapsed_time * 3.6

                if tid not in final_id_map:
                    final_id_map[tid] = f'vehicle_{vehicle_counter}'
                    vehicle_counter += 1

                vid_str = final_id_map[tid]
                writer.writerow([frame_count, vid_str, vehicle_type, 'Down', round(speed, 2), cx, cy, x1, y1, x2, y2])
                cv2.putText(frame, f'{int(speed)} Km/h', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if tid not in up_frame and BLUE_LINE_Y - OFFSET < cy < BLUE_LINE_Y + OFFSET:
                up_frame[tid] = frame_count

            if tid in up_frame and tid not in counter_up and RED_LINE_Y - OFFSET < cy < RED_LINE_Y + OFFSET:
                counter_up.append(tid)
                elapsed_frames = frame_count - up_frame[tid]
                elapsed_time = elapsed_frames / fps if fps > 0 else 0.01
                if elapsed_time == 0:
                    continue
                speed = DISTANCE_BETWEEN_LINES / elapsed_time * 3.6

                if tid not in final_id_map:
                    final_id_map[tid] = f'vehicle_{vehicle_counter}'
                    vehicle_counter += 1

                vid_str = final_id_map[tid]
                writer.writerow([frame_count, vid_str, vehicle_type, 'Up', round(speed, 2), cx, cy, x1, y1, x2, y2])
                cv2.putText(frame, f'{int(speed)} Km/h', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    text_color = (0, 0, 0)
    cv2.rectangle(frame, (0, 0), (250, 90), (0, 255, 255), -1)
    cv2.line(frame, (172, RED_LINE_Y), (774, RED_LINE_Y), (0, 0, 255), 2)
    cv2.putText(frame, 'Red Line', (172, RED_LINE_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.line(frame, (8, BLUE_LINE_Y), (927, BLUE_LINE_Y), (255, 0, 0), 2)
    cv2.putText(frame, 'Blue Line', (8, BLUE_LINE_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(frame, f'Going Down - {len(counter_down)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(frame, f'Going Up - {len(counter_up)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    cv2.imwrite(f'{OUTPUT_DIR}/frame_{frame_count}.jpg', frame)
    out.write(frame)
    cv2.imshow('Vehicle Speed Estimation', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
