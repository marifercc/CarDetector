import os, csv, json, random, argparse #lote
import collections #trayectorias por cuadro
import cv2 #recortes
import numpy as np #mosaicos
from lote_14oct import leer_formulario, tramo, buscar_reporte
from count_cars_chihuahua import W, H, FRANJA_X_IZQ, FRANJA_X_DER, FRANJA_Y_MIN, FRANJA_Y_MAX

# Revision por muestra de los conteos del lote del 14 de octubre, para sacar F1 por
# vehiculo en videos que solo tienen el total a mano por movimiento.
# Por video y sentido toma hasta --muestra conteos al azar dentro del tramo contado a
# mano y arma mosaicos: caja roja = el conteo; verdes = otros IDs ya contados en ese
# cuadro (con su numero); grises = IDs no contados. Escribe marcas_muestra.csv con la
# columna Marca vacia para llenarla (bien / doble / no_vehiculo / otro_sentido).
# f1_lote.py calcula el F1 con esas marcas.

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta", help="Carpeta del lote (con reportes, trayectorias y _videos).")
    ap.add_argument("--muestra", type=int, default=30)
    ap.add_argument("--semilla", type=int, default=14)
    args = ap.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    carpeta = os.path.join(base, args.carpeta)
    formulario = leer_formulario(os.path.join(base, "conteos_14oct.csv"))
    salida = os.path.join(carpeta, "revision")
    os.makedirs(salida, exist_ok=True)
    filas_marcas = []
    for nombre in sorted(formulario):
        rep = buscar_reporte(carpeta, nombre)
        video = os.path.join(carpeta, "_videos", nombre + ".mp4")
        if rep is None or not os.path.exists(video):
            continue
        a, b = tramo(formulario[nombre])
        cont = []
        for r in csv.reader(open(rep)):
            if not r:
                break
            if r[0].isdigit() and a <= float(r[2]) <= b:
                cont.append((int(r[0]), int(r[1]), float(r[2]), r[3], *map(int, r[5:9])))
        num_de_id = {c[1]: c[0] for c in cont}
        rnd = random.Random(f"{args.semilla}-{nombre}")
        elegidos = []
        for s in ("izq -> der", "der -> izq"):
            g = [c for c in cont if c[3] == s]
            elegidos += sorted(rnd.sample(g, min(args.muestra, len(g))), key=lambda c: c[2])
        tray = [x for x in os.listdir(carpeta) if x.startswith(f"trayectorias_{nombre}")]
        cajas = collections.defaultdict(list)
        if tray:
            fps_v = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)
            quiero = {round(c[2] * fps_v) for c in elegidos}
            with open(os.path.join(carpeta, tray[0])) as f:
                rd = csv.reader(f); next(rd)
                for fr, i, x1, y1, x2, y2, *_ in rd:
                    fr = int(fr)
                    if fr in quiero:
                        cajas[fr].append((int(i), int(x1), int(y1), int(x2), int(y2)))
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        tiles = []
        for k, c in enumerate(elegidos):
            num, i, seg, s, x1, y1, x2, y2 = c
            fr = round(seg * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(fr - 1, 0))
            ok, f = cap.read()
            if not ok:
                continue
            f = cv2.resize(f, (W, H))
            cv2.rectangle(f, (FRANJA_X_IZQ, FRANJA_Y_MIN), (FRANJA_X_DER, FRANJA_Y_MAX), (255, 0, 0), 1)
            for j, a1, b1, a2, b2 in cajas.get(fr, []):
                if j == i:
                    continue
                if j in num_de_id:
                    cv2.rectangle(f, (a1, b1), (a2, b2), (0, 220, 0), 1)
                    cv2.putText(f, f"#{num_de_id[j]}", (a1, b1 - 3), 0, 0.45, (0, 220, 0), 1)
                else:
                    cv2.rectangle(f, (a1, b1), (a2, b2), (170, 170, 170), 1)
            cv2.rectangle(f, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ox, oy = max(0, min(W - 520, cx - 260)), max(0, min(H - 200, cy - 100))
            t = f[oy:oy + 200, ox:ox + 520].copy()
            cv2.putText(t, f"{k} {s[:3]}", (6, 20), 0, 0.6, (0, 255, 255), 2)
            tiles.append(t)
            filas_marcas.append({"video": nombre, "k": k, "Numero": num, "Segundo": seg, "Sentido": s, "Marca": ""})
        for m in range(0, len(tiles), 12):
            g = tiles[m:m + 12] + [np.zeros_like(tiles[0])] * (12 - len(tiles[m:m + 12]))
            mos = np.vstack([np.hstack(g[r:r + 3]) for r in range(0, 12, 3)])
            cv2.imwrite(os.path.join(salida, f"{nombre}_{m // 12:02d}.jpg"), mos, [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(nombre, "contados en tramo", collections.Counter(c[3] for c in cont), "muestra", len(tiles), flush=True)
    ruta = os.path.join(salida, "marcas_muestra.csv")
    if os.path.exists(ruta):
        print("ya existe", ruta, "- no se sobreescribe")
    else:
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["video", "k", "Numero", "Segundo", "Sentido", "Marca"])
            w.writeheader(); w.writerows(filas_marcas)
