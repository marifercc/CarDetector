import cv2 #abrir videos
import json #guardar las cajas para el etiquetador
import os #rutas de archivos
import argparse #leer argumentos de la terminal
import torchvision #NMS para los cuadros de RT-DETR
from ultralytics import RTDETR #pre-etiquetar con el modelo actual

# Prepara frames para entrenar el detector con imagenes de esta camara: saca frames
# de los videos, tapa en gris todo fuera de la calle (FRANJA_Y0 a FRANJA_Y1) y les
# pone las cajas que ya encuentra RT-DETR, para que en etiquetado/index.html solo
# haya que corregirlas (borrar las que sobran, dibujar las que faltan).
# Del video 20261014_081527A solo se usan los primeros 99 s: de ahi en adelante es
# el tramo contado a mano (08:17:07 - 08:20:27), que se deja para probar el modelo.

FRANJA_Y0, FRANJA_Y1 = 240, 470 # la calle; arriba es la calle del fondo, abajo la banqueta
W, H = 1280, 720
GRIS = 114 # mismo gris que usa ultralytics para rellenar
CONF_PRE = 0.25 # cajas propuestas; con menos salen demasiadas que hay que borrar

# (video, segundo inicial, segundo final, cada cuantos segundos)
FUENTES = [
    ("chihuahua_5min.mp4", 0, 300, 2.0),
    (r"C:\Users\marif\Downloads\20261014_081527A.mp4", 0, 99, 2.0),
]


def tapar_fuera_de_calle(frame):
    frame = frame.copy()
    frame[:FRANJA_Y0] = GRIS
    frame[FRANJA_Y1:] = GRIS
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saca y pre-etiqueta frames para entrenar el detector.")
    parser.parse_args()
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    carpeta = os.path.join(ruta_base, "etiquetado")
    os.makedirs(os.path.join(carpeta, "frames"), exist_ok=True)
    model = RTDETR(os.path.join(ruta_base, "rtdetr-x.pt"))

    datos = []
    for video, t0, t1, paso in FUENTES:
        ruta = video if os.path.isabs(video) else os.path.join(ruta_base, video)
        nombre_video = os.path.splitext(os.path.basename(ruta))[0]
        cap = cv2.VideoCapture(ruta)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # se lee frame por frame en vez de saltar con CAP_PROP_POS_FRAMES: en estos mp4
        # el salto no cae en el frame exacto y las imagenes no cuadrarian con las
        # trayectorias del contador (que lee frame por frame)
        t = t0
        indice = -1
        while t < t1:
            ret, frame = cap.read()
            if not ret:
                break
            indice += 1
            if indice < int(t * fps):
                continue
            frame = tapar_fuera_de_calle(cv2.resize(frame, (W, H)))
            nombre = f"{nombre_video}_{int(t * 10):05d}.jpg"
            cv2.imwrite(os.path.join(carpeta, "frames", nombre), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            r = model.predict(frame, classes=[2, 3, 5, 7], conf=CONF_PRE, verbose=False)[0]
            # RT-DETR no hace NMS (ver count_cars_chihuahua.py)
            quedan = torchvision.ops.nms(r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), 0.7).numpy()
            cajas = [[int(v) for v in b] + [round(float(c), 2)]
                     for b, c in zip(r.boxes.xyxy.cpu().numpy()[quedan], r.boxes.conf.cpu().numpy()[quedan])]
            datos.append({"imagen": nombre, "cajas": cajas})
            t += paso
        cap.release()
        print(f"{nombre_video}: {sum(d['imagen'].startswith(nombre_video) for d in datos)} frames")

    with open(os.path.join(carpeta, "datos.js"), "w", encoding="utf-8") as f:
        f.write("const FRANJA = " + json.dumps([FRANJA_Y0, FRANJA_Y1]) + ";\n")
        f.write("const DATOS = " + json.dumps(datos) + ";\n")
    print(f"Abre: {os.path.join(carpeta, 'index.html')}")
