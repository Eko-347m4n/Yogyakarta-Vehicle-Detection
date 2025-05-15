import cv2
from ultralytics import YOLO
import time

# Load model YOLO
model = YOLO("yolov8n.pt")

# URL stream (ganti jika perlu)
stream_url = "https://mam.jogjaprov.go.id:1937/cctv-public/ViewDepanBeringharjo.stream/playlist.m3u8"
cap = cv2.VideoCapture(stream_url)

# Cek apakah stream berhasil dibuka
if not cap.isOpened():
    print("Gagal membuka stream.")
    exit()

print("Menjalankan monitoring... Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    # Deteksi objek dengan YOLO
    results = model(frame)[0]

    # Tampilkan bounding boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, cls_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Resize agar UI tidak berat
    desired_width = 720
    height, width = frame.shape[:2]
    scale_ratio = desired_width / width
    resized_frame = cv2.resize(frame, (desired_width, int(height * scale_ratio)))

    # Tampilkan frame
    cv2.imshow("Live Monitoring YOLO", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
