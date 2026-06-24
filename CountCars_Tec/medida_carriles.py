# Herramienta de calibración - corre esto aparte
import cv2
import numpy as np

puntos = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos.append((x, y))
        print(f"Punto {len(puntos)}: ({x}, {y})")
        if len(puntos) == 2:
            dist = np.linalg.norm(np.array(puntos[0]) - np.array(puntos[1]))
            metros_reales = 3.5  # ancho de carril estándar
            ppm = dist / metros_reales
            print(f"\n>>> PIXELS_PER_METER = {ppm:.2f}")

cap = cv2.VideoCapture("Video_TECmin.mp4")
ret, frame = cap.read()
frame = cv2.resize(frame, (1280, 720))
cv2.imshow("Calibrar", frame)
cv2.setMouseCallback("Calibrar", click)
cv2.waitKey(0)