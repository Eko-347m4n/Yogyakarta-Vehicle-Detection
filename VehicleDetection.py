import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")  # Ganti jika pakai model kustom

# Kendaraan yang ingin dideteksi
vehicle_classes = ["car", "bus", "truck", "motorcycle", "bicycle"]

# Buka stream video
stream_url = "https://mam.jogjaprov.go.id:1937/cctv-public/ViewDepanBeringharjo.stream/playlist.m3u8"
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Gagal membuka stream.")
    exit()

print("Monitoring kendaraan... Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    # Deteksi objek
    results = model(frame)[0]

    vehicle_count = 0

    for box in results.boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]

        # Hanya deteksi kendaraan
        if cls_name in vehicle_classes:
            vehicle_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Tambahkan overlay jumlah kendaraan
    cv2.putText(frame, f"Jumlah Kendaraan: {vehicle_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    # Resize agar tampil efisien
    desired_width = 720
    height, width = frame.shape[:2]
    scale_ratio = desired_width / width
    resized_frame = cv2.resize(frame, (desired_width, int(height * scale_ratio)))

    # Tampilkan
    cv2.imshow("Deteksi Kendaraan", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
