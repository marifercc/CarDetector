import os, sys, json, shutil, argparse, time #lote
import cv2 #videos
import numpy as np #cajas
import torch, torchvision #NMS
from ultralytics import RTDETR #modelo maestro
from boxmot.trackers.bbox.ocsort import OcSort #tracker para rellenar huecos
from count_cars_chihuahua import W, H
from preparar_etiquetado import tapar_fuera_de_calle, FRANJA_Y0, FRANJA_Y1
from lote_14oct import CARPETA_VIDEOS
from etiquetas_auto import MIN_OBS, MIN_CONF_MAX, se_mueve, caja_en, dentro_de, DENTRO_PEDAZO

# Etiquetas automaticas de mas videos de la misma camara para reentrenar el detector
# (mismo metodo que etiquetas_auto.py: trayectorias de OC-SORT, huecos cortos
# rellenados, solo IDs que se mueven y que alguna vez fueron claramente un auto).
# Usa videos del 14 de octubre que NO estan en la muestra del lote (lote_14oct.MUESTRA,
# que es con la que se mide el F1) y videos del 22 de octubre (tarde y noche).
# Por video toma un tramo de SEGUNDOS s y guarda un frame cada PASO s.
# Salida: etiquetado/frames/<video>_<t*10>.jpg y etiquetado/etiquetas_lote.json
# (las 200 imagenes de etiquetas_auto.json + las nuevas).

VIDEOS = ["20261014_054527A", "20261014_061527A", "20261014_064027A", "20261014_070027A", "20261014_072527A",
          "20261014_075527A", "20261014_083027A", "20261014_085527A", "20261014_093027A", "20261014_110027A",
          "20261014_120027A", "20261014_130027A", "20261014_141027A", "20261014_145027A", "20261014_151527A",
          "20261014_155527A", "20261022_173214A", "20261022_180632A", "20261022_184132A", "20261022_191632A",
          "20261022_195632A", "20261022_203132A"]
CARPETA_22 = CARPETA_VIDEOS.replace("14-10-2026", "22-10-2026")


def buscar(nombre):
    for carpeta in (CARPETA_VIDEOS, CARPETA_22):
        for raiz, _, archivos in os.walk(carpeta):
            if nombre + ".mp4" in archivos:
                return os.path.join(raiz, nombre + ".mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", type=float, default=100)
    ap.add_argument("--segundos", type=float, default=90)
    ap.add_argument("--paso", type=float, default=3.0)
    ap.add_argument("--conf", type=float, default=0.08)
    args = ap.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(base, "etiquetado", "frames")
    tmp = os.path.join(base, "lote_14oct_f1", "_videos")
    salida = os.path.join(base, "etiquetado", "etiquetas_lote.json")
    res = json.load(open(salida)) if os.path.exists(salida) else \
        json.load(open(os.path.join(base, "etiquetado", "etiquetas_auto.json")))
    model = RTDETR(os.path.join(base, "rtdetr-chihuahua.pt"))
    for nombre in VIDEOS:
        if any(k.startswith(nombre + "_") for k in res["imagenes"]):
            continue
        ruta = buscar(nombre)
        if ruta is None:
            print(nombre, "no encontrado", flush=True); continue
        local = os.path.join(tmp, nombre + ".mp4")
        # OneDrive a veces falla una descarga ("Invalid argument") y a la siguiente sale bien
        for intento in range(5):
            try:
                shutil.copyfile(ruta, local)
                break
            except OSError as e:
                print(nombre, "fallo la descarga, reintento", e, flush=True)
                time.sleep(30)
        else:
            continue
        cap = cv2.VideoCapture(local)
        fps = cap.get(cv2.CAP_PROP_FPS)
        tracker = OcSort(det_thresh=args.conf, max_age=90, min_hits=1, iou_threshold=0.3)
        img = np.zeros((H, W, 3), np.uint8)
        tray, conf_max, guardar = {}, {}, {}
        a, b = int(args.inicio * fps), int((args.inicio + args.segundos) * fps)
        cada = int(args.paso * fps)
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n < a:
                continue
            if n > b:
                break
            frame = tapar_fuera_de_calle(cv2.resize(frame, (W, H)))
            r = model.predict(frame, verbose=False, conf=args.conf, imgsz=640)[0].boxes
            d = np.concatenate([r.xyxy.cpu().numpy(), r.conf.cpu().numpy()[:, None], r.cls.cpu().numpy()[:, None]], 1)
            if len(d) > 1:
                d = d[torchvision.ops.nms(torch.from_numpy(d[:, :4]), torch.from_numpy(d[:, 4]), 0.7).numpy()]
            tracks = tracker.update(d if len(d) else np.empty((0, 6)), img)
            for bx, i, c in zip(tracks.xyxy, tracks.id, tracks.conf):
                i = int(i)
                tray.setdefault(i, {})[n] = tuple(int(v) for v in bx)
                conf_max[i] = max(conf_max.get(i, 0), float(c))
            # se guarda despues de un margen para que los IDs ya existan antes y despues
            if (n - a) % cada == 0 and a + 30 < n < b - 30:
                im = f"{nombre}_{int(n / fps * 10):05d}.jpg"
                cv2.imwrite(os.path.join(frames_dir, im), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                guardar[im] = n
        cap.release()
        os.remove(local)
        buenas = {i: t for i, t in tray.items()
                  if len(t) >= MIN_OBS and conf_max[i] >= MIN_CONF_MAX and se_mueve([t[f] for f in sorted(t)])}
        for im, fr in guardar.items():
            cajas = [c for c in (caja_en(t, fr) for t in buenas.values()) if c is not None]
            cajas = [c for c in cajas if FRANJA_Y0 <= (c[1] + c[3]) / 2 <= FRANJA_Y1]
            cajas = [c for c in cajas if not any(dentro_de(c, o) >= DENTRO_PEDAZO for o in cajas if o is not c)]
            res["imagenes"][im] = [list(c) for c in cajas]
        json.dump(res, open(salida, "w"))
        print(nombre, len(guardar), "imagenes", sum(len(res["imagenes"][k]) for k in guardar), "cajas", flush=True)
    print("total", len(res["imagenes"]), "imagenes")
