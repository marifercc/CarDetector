import sys, json, os #argumentos
import cv2 #recortes
import numpy as np #mosaico
from replay_conteo import cargar, replay
from evalua_replay import evalua
from count_cars_chihuahua import W, H

# Para una configuracion, saca los conteos "desconocidos" (sin pareja en la corrida
# revisada) y arma mosaicos para revisarlos a ojo: caja roja = el conteo nuevo,
# verdes = autos ya contados en ese cuadro con su numero. Si la roja cae sobre un
# auto verde es doble; si es un auto sin caja verde, es un auto real.
#   python recortes_desconocidos.py detecciones/det_...npz '{"abajo": {...}}' carpeta_salida

if __name__ == "__main__":
    ruta, kw, carpeta = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
    fr, fps = cargar(ruta)
    cont = replay(fr, fps, **kw)
    res = evalua(cont, detalle=True)
    print({k: v for k, v in res.items() if k != "desconocidos"})
    os.makedirs(carpeta, exist_ok=True)
    cap = cv2.VideoCapture(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chihuahua_5min.mp4"))
    tiles = []
    for c in sorted(res["desconocidos"], key=lambda c: c[8]):
        num, _, seg, s, x1, y1, x2, y2, n, vecinos = c
        cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
        ok, f = cap.read()
        f = cv2.resize(f, (W, H))
        for m, (a, b, cc, d) in vecinos:
            cv2.rectangle(f, (a, b), (cc, d), (0, 220, 0), 1)
            cv2.putText(f, f"#{m}", (a, b - 3), 0, 0.45, (0, 220, 0), 1)
        cv2.rectangle(f, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        a, b = max(0, min(W - 520, cx - 260)), max(0, min(H - 200, cy - 100))
        t = cv2.resize(f[b:b + 200, a:a + 520], (780, 300))
        cv2.putText(t, f"N{num} {seg}s {s[:3]}", (8, 24), 0, 0.7, (0, 255, 255), 2)
        tiles.append(t)
    for k in range(0, len(tiles), 6):
        grupo = tiles[k:k + 6] + [np.zeros_like(tiles[0])] * (6 - len(tiles[k:k + 6]))
        mos = np.vstack([np.hstack(grupo[i:i + 2]) for i in range(0, 6, 2)])
        cv2.imwrite(os.path.join(carpeta, f"mos_{k // 6:02d}.jpg"), mos, [cv2.IMWRITE_JPEG_QUALITY, 80])
    json.dump([c[:9] for c in res["desconocidos"]], open(os.path.join(carpeta, "desconocidos.json"), "w"))
    print(len(tiles), "desconocidos")
