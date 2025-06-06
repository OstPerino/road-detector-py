from ultralytics import YOLO
import cv2

# model = YOLO("/content/drive/MyDrive/best.pt")
# video_path = "/content/drive/MyDrive/nissan.mp4"
model = YOLO("ai/best.pt")
video_path = "ai/input/IMG_9008.MOV"

model.to("cuda")

cap = cv2.VideoCapture(video_path)

pathSplitted = video_path.split('/')
name = pathSplitted[len(pathSplitted) - 1].split('.')[0] + '.mp4'

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()