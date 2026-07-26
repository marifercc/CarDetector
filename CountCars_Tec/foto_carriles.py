import cv2
import numpy as np

COORD_ZONA_A = np.array([

    [400, 280],   # 0 - arriba izquierda (donde está el carro negro)
    [750, 280],   # 1 - arriba derecha
    [1100, 450],  # 2 - derecha media
    [1100, 720],  # 3 - esquina inferior derecha
    [600, 720],   # 4 - inferior centro
    [300, 500],   # 5 - izquierda media
], np.int32)


cap = cv2.VideoCapture("Video_TECmin.mp4")
ret, frame = cap.read()
frame = cv2.resize(frame, (1280, 720))

cv2.polylines(frame, [COORD_ZONA_A], True, (0, 255, 0), 3)

for i, punto in enumerate(COORD_ZONA_A):
    cv2.circle(frame, tuple(punto), 6, (0, 0, 255), -1)
    cv2.putText(frame, str(i), tuple(punto), 0, 0.8, (255, 255, 0), 2)

cv2.imshow("Verificar zona", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()