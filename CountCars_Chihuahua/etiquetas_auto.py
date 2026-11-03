import csv #leer trayectorias
import json #escribir etiquetas
import os #rutas de archivos
import argparse #leer argumentos de la terminal
import cv2 #leer los fps de cada video
from collections import defaultdict

# Etiquetas automaticas para entrenar el detector, a partir de las trayectorias de
# count_cars_chihuahua.py --trayectorias. Ventaja sobre las cajas de un solo frame:
# si un auto se vio antes y despues de quedar tapado, su caja se rellena en los
# frames donde el detector no lo vio, que son justo los que hay que ensenarle.
# Se quitan trayectorias cortas (detecciones falsas sueltas), las que nunca se
# mueven (letreros) y cajas que son un pedazo de otro auto.
# Escribe etiquetado/etiquetas_auto.json con el mismo formato que descarga
# etiquetado/index.html, asi se puede revisar ahi con "Cargar".

MIN_OBS = 10 # frames en que se detecto el ID
MIN_MOVIMIENTO = 25 # px que se aleja de donde aparecio; menos = algo fijo (letrero)
MAX_HUECO = 60 # frames (2 s) maximos a rellenar entre dos detecciones del mismo ID
DENTRO_PEDAZO = 0.8 # caja mas chica >= esto dentro de otra = pedazo del mismo auto


def leer_trayectorias(ruta):
    tray = defaultdict(dict) # id -> frame -> caja
    with open(ruta, newline="") as f:
        r = csv.reader(f)
        next(r)
        for fr, i, x1, y1, x2, y2, _conf, _clase in r:
            tray[int(i)][int(fr)] = (int(x1), int(y1), int(x2), int(y2))
    return tray


def se_mueve(cajas):
    x0 = (cajas[0][0] + cajas[0][2]) / 2
    y0 = (cajas[0][1] + cajas[0][3]) / 2
    return any(abs((c[0] + c[2]) / 2 - x0) + abs((c[1] + c[3]) / 2 - y0) >= MIN_MOVIMIENTO for c in cajas)


def caja_en(frames_id, fr):
    # caja observada, o interpolada si fr cae en un hueco corto entre dos observaciones
    if fr in frames_id:
        return frames_id[fr]
    antes = [f for f in frames_id if f < fr]
    despues = [f for f in frames_id if f > fr]
    if not antes or not despues:
        return None
    a, b = max(antes), min(despues)
    if b - a > MAX_HUECO:
        return None
    t = (fr - a) / (b - a)
    return tuple(round(p + (q - p) * t) for p, q in zip(frames_id[a], frames_id[b]))


def dentro_de(a, b):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    if area_a >= (b[2] - b[0]) * (b[3] - b[1]):
        return 0.0
    ancho = min(a[2], b[2]) - max(a[0], b[0])
    alto = min(a[3], b[3]) - max(a[1], b[1])
    if ancho <= 0 or alto <= 0:
        return 0.0
    return ancho * alto / area_a


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Etiquetas automaticas desde trayectorias.")
    parser.add_argument("trayectorias", nargs="+", metavar="VIDEO=CSV",
                        help="Ruta del video y su trayectorias_*.csv, ej. chihuahua_5min.mp4=trayectorias_chihuahua_5min.csv")
    args = parser.parse_args()
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    carpeta = os.path.join(ruta_base, "etiquetado")
    with open(os.path.join(carpeta, "datos.js"), encoding="utf-8") as f:
        texto = f.read()
    franja = json.loads(texto.split("const FRANJA = ")[1].split(";")[0])
    datos = json.loads(texto.split("const DATOS = ")[1].rsplit(";", 1)[0])

    salida = {"franja": franja, "imagenes": {}}
    for par in args.trayectorias:
        ruta_video, ruta_csv = par.split("=", 1)
        if not os.path.isabs(ruta_video):
            ruta_video = os.path.join(ruta_base, ruta_video)
        video = os.path.splitext(os.path.basename(ruta_video))[0]
        # el frame de cada imagen se saco con los fps del video (preparar_etiquetado.py)
        cap = cv2.VideoCapture(ruta_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        tray = leer_trayectorias(os.path.join(ruta_base, ruta_csv))
        buenas = {i: t for i, t in tray.items()
                  if len(t) >= MIN_OBS and se_mueve([t[f] for f in sorted(t)])}
        print(f"{video}: {len(buenas)} de {len(tray)} IDs se usan")
        for d in datos:
            if not d["imagen"].startswith(video + "_"):
                continue
            t = int(d["imagen"][len(video) + 1:-4]) / 10
            fr = int(t * fps) + 1 # frame_count del contador empieza en 1
            cajas = [c for c in (caja_en(t_id, fr) for t_id in buenas.values()) if c is not None]
            cajas = [c for c in cajas if franja[0] <= (c[1] + c[3]) / 2 <= franja[1]]
            cajas = [c for c in cajas if not any(dentro_de(c, o) >= DENTRO_PEDAZO for o in cajas if o is not c)]
            salida["imagenes"][d["imagen"]] = [list(c) for c in cajas]

    ruta = os.path.join(carpeta, "etiquetas_auto.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(salida, f)
    n = sum(len(v) for v in salida["imagenes"].values())
    print(f"{len(salida['imagenes'])} imagenes, {n} cajas -> {ruta}")
