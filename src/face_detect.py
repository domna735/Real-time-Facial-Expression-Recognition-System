# src/face_detect.py

import cv2, os

# 1. 建 outputs/faces 資料夾
os.makedirs("outputs/faces", exist_ok=True)

# 2. 載入 Haar Cascade
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 3. 讀圖 + 轉灰階
img = cv2.imread("data/test.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. 偵測人臉
faces = cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)

# 讀取偵測結果後，取 len 判斷
if faces is None or len(faces) == 0:
    print("No faces detected.")
else:
    for i, (x, y, w, h) in enumerate(faces):
        roi = img[y:y+h, x:x+w]
        path = f"outputs/faces/face_{i}.jpg"
        cv2.imwrite(path, roi)
        print(f"Saved face_{i}.jpg → {path}")
