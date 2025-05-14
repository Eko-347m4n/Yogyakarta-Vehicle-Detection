import cv2
from ultralytics import YOLO
import time
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

# Stream CCTV
stream_url = "https://mam.jogjaprov.go.id:1937/cctv-public/ViewDepanBeringharjo.stream/playlist.m3u8"
cap = cv2.VideoCapture(stream_url)
frame_interval = 1
start_time = time.time()

# Folder penyimpanan dataset
os.makedirs("dataset", exist_ok=True)

print("Tekan huruf untuk memberi label (misal: 'a', 'b', dll), 's' untuk skip, atau 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil frame.")
        break

    now = time.time()
    if now - start_time >= frame_interval:
        # Deteksi objek
        results = model(frame)[0]

        # Tampilkan bounding box pada frame asli
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Resize frame agar lebar 720px
        desired_width = 720
        height, width = frame.shape[:2]
        scale_ratio = desired_width / width
        resized_frame = cv2.resize(frame, (desired_width, int(height * scale_ratio)))

        # Tampilkan UI
        cv2.imshow("Labeling Frame [Tekan huruf / s / q]", resized_frame)

        key = cv2.waitKey(0)
        if key == ord('q'):
            print("Keluar.")
            break
        elif key == ord('s'):
            print("Frame dilewati.")
        elif 32 <= key <= 126:
            label = chr(key)
            filename = f"dataset/{label}_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame disimpan: {filename}")

        start_time = time.time()

cap.release()
cv2.destroyAllWindows()
