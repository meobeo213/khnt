import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms, models
import socket
import json

HOST = '127.0.0.1'
PORT = 65432

# =====================================================
# HÀM GỬI LỆNH VỀ ROBOT (WEbOTS SERVER)
# =====================================================
last_cmd = ""

def send(cmd: str):
    global last_cmd
    if cmd == last_cmd:
        return
    last_cmd = cmd

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            s.connect((HOST, PORT))
            s.sendall(json.dumps({"command": cmd}).encode())
        print(f"→ ĐÃ GỬI: {cmd}")
    except Exception as e:
        print("❌ Không gửi được:", e)


# =====================================================
# KHỞI TẠO MODEL FACE + EMOTION
# =====================================================
cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
detector = mp_face.FaceDetection(min_detection_confidence=0.65)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
model.load_state_dict(torch.load("models/efficientnet_b0_best_latest.pth",
                                 weights_only=False, map_location=device))

model.to(device)
model.eval()

labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise', 'Neutral', 'Disgust']

EMOTION_TO_CMD = {
    "Happy": "WAVE",
    "Sad": "LOOK_DOWN",
    "Angry": "MOVE_BACK",
    "Suprise": "LOOK_AROUND",
    "Neutral": "IDLE",
    "Fear": "STOP_ALL",
    "Disgust": "STOP_ALL"
}

transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# =====================================================
# VÒNG LẶP CHÍNH
# =====================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if results.detections:
        for det in results.detections:

            # Lấy bbox
            bbox = det.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y = int(bbox.xmin*iw), int(bbox.ymin*ih)
            w, h = int(bbox.width*iw), int(bbox.height*ih)

            x, y = max(0,x), max(0,y)
            face = frame[y:y+h, x:x+w]

            mp_draw.draw_detection(frame, det)

            if face.size > 0:
                face_tensor = transformer(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(face_tensor)
                    _, pred = torch.max(output, 1)

                emotion = labels[pred.item()]

                # Gửi lệnh về Webots
                cmd = EMOTION_TO_CMD.get(emotion, "IDLE")
                send(cmd)

                cv2.putText(frame, f"{emotion}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Emotion → Webots", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
send("IDLE")
