from ultralytics import YOLO
import cv2
from geopy.distance import geodesic
import numpy as np

# === ТВОИ ДАННЫЕ ===
video_path = "ai/input/IMG_9109.MOV"  # <-- сюда путь к видео
lat1, lon1 = 55.996508, 92.792385     # <-- координаты начала
lat2, lon2 = 55.995836, 92.785775     # <-- координаты конца
segment_length_m = 10                 # длина сегмента в метрах

# === АНАЛИЗ ВИДЕО ===
model = YOLO("ai/best.pt")
model.to("cuda")
cap = cv2.VideoCapture(video_path)

frame_results = []
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    results = model(frame)
    # Если есть хотя бы один detection — считаем, что разметка есть
    has_marking = int(len(results[0].boxes) > 0)
    frame_results.append(has_marking)
    frame_count += 1

cap.release()

print(f"Обработано кадров: {frame_count}")

# === ИНТЕРПОЛЯЦИЯ КООРДИНАТ ===
N = len(frame_results)
frame_coords = [
    (
        lat1 + (lat2 - lat1) * i / (N - 1),
        lon1 + (lon2 - lon1) * i / (N - 1)
    )
    for i in range(N)
]

# Можно считать покрытие определенной разметки (например можно выбрать определенную разметку на нужном участке дороги)
# === РАЗБИВАЕМ НА СЕГМЕНТЫ ===
total_distance = geodesic((lat1, lon1), (lat2, lon2)).meters
num_segments = int(np.ceil(total_distance / segment_length_m))
segment_frames = [[] for _ in range(num_segments)]

for idx, (lat, lon) in enumerate(frame_coords):
    dist_from_start = geodesic((lat1, lon1), (lat, lon)).meters
    seg_idx = min(int(dist_from_start // segment_length_m), num_segments - 1)
    segment_frames[seg_idx].append(frame_results[idx])

# === СЧИТАЕМ ПОКРЫТИЕ ===
segment_coverage = []
for i, frames in enumerate(segment_frames):
    if frames:
        coverage = 100 * sum(frames) / len(frames)
        print(f"Сегмент {i+1}: {len(frames)} кадров, покрытие = {coverage:.1f}%")
    else:
        print(f"Сегмент {i+1}: нет данных")
    segment_coverage.append(coverage if frames else None)

# === ИТОГОВЫЙ ВЫВОД ===
print("\nИтоговая статистика по сегментам:")
for i, coverage in enumerate(segment_coverage):
    if coverage is not None:
        print(f"  Сегмент {i+1}: {coverage:.1f}% покрытия")
    else:
        print(f"  Сегмент {i+1}: нет данных")