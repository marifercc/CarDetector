import os, csv, json, random, sys #lote
import cv2 #recortes
import numpy as np #mosaicos
from replay_conteo import cargar, replay
from lote_14oct import tramo
from count_cars_chihuahua import W, H, FRANJA_X_IZQ, FRANJA_X_DER
import evalua_lote as E

# Revision de los conteos de una configuracion nueva (modelo/tracker) que NO se emparejan
# con un conteo ya revisado de lote_14oct_f1. Los emparejados heredan su marca; de los
# sin pareja se toma una muestra por video y sentido y se arma un recorte por conteo
# en el cuadro en que se conto:
#   rojo     = el conteo a revisar, con su recorrido del ultimo segundo
#   verde    = otros IDs ya contados (mismo sentido) que estan en ese cuadro, con su numero
#   amarillo = recorrido de IDs contados en los 3 s previos que ya no estan en el cuadro;
#              si el amarillo desemboca en el rojo es el mismo auto con otro ID (doble)
#   gris     = IDs no contados en ese cuadro
# Escribe <salida>/marcas_nuevos.csv (k = posicion del recorte en los mosaicos del video)
# con Marca vacia (bien / doble / no_vehiculo / otro_sentido). f1_nuevos.py calcula el F1.
# MODELO=rtdetr-chihuahua2 python muestra_nuevos.py '<json kw>' <salida> [por_grupo]

ANCHO, ALTO = 520, 260  # recorte


def en(h, n):
    # caja de un recorrido en el cuadro n (o None)
    for f in reversed(h):
        if f[0] == n:
            return f[1:]
        if f[0] < n:
            return None


def camino(img, h, n, atras, color):
    pts = [((f[1] + f[3]) // 2, (f[2] + f[4]) // 2) for f in h if n - atras <= f[0] <= n]
    for p, q in zip(pts, pts[1:]):
        cv2.line(img, p, q, color, 2)
    return pts


def recorte(frame, c, cont, hist, fps):
    img = frame.copy()
    n, i = c[8], c[1]
    cv2.line(img, (FRANJA_X_IZQ, 0), (FRANJA_X_IZQ, H), (255, 120, 0), 1)
    cv2.line(img, (FRANJA_X_DER, 0), (FRANJA_X_DER, H), (255, 120, 0), 1)
    num = {o[1]: o[0] for o in cont}
    for j, h in hist.items():
        b = en(h, n)
        if j == i:
            continue
        if b is not None:
            color = (0, 200, 0) if j in num and cont[num[j] - 1][8] <= n else (150, 150, 150)
            cv2.rectangle(img, b[:2], b[2:], color, 1 if color[1] == 150 else 2)
            if color[1] == 200:
                cv2.putText(img, str(num[j]), (b[0], b[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        elif j in num and 0 < n - cont[num[j] - 1][8] <= 3 * fps:
            pts = camino(img, h, n, 4 * fps, (0, 220, 255))
            if pts:
                cv2.circle(img, pts[-1], 4, (0, 220, 255), -1)
                cv2.putText(img, str(num[j]), pts[-1], cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 255), 1)
    camino(img, hist[i], n, fps, (0, 0, 255))
    cv2.rectangle(img, tuple(c[4:6]), tuple(c[6:8]), (0, 0, 255), 2)
    cx, cy = (c[4] + c[6]) // 2, (c[5] + c[7]) // 2
    x0 = min(max(cx - ANCHO // 2, 0), W - ANCHO)
    y0 = min(max(cy - ALTO // 2, 0), H - ALTO)
    r = img[y0:y0 + ALTO, x0:x0 + ANCHO].copy()
    cv2.putText(r, f"#{c[0]} {c[3][:3]} t={c[2]}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
    return r


if __name__ == "__main__":
    kw = json.loads(sys.argv[1])
    salida = os.path.join(E.AQUI, sys.argv[2])
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    os.makedirs(salida, exist_ok=True)
    rnd = random.Random(17)
    base = {a: b for a, b in kw.items() if a != "abajo"}
    # un tracker por sentido, igual que replay(abajo=...): der -> izq del base, izq -> der del de abajo
    corridas = {"der -> izq": base, "izq -> der": {**base, **kw.get("abajo", {})}}
    filas = []
    for v in os.environ.get("VIDEOS", " ".join(E.FORM_LOTE)).split():
        fr, fps = cargar(os.path.join(E.AQUI, "detecciones", f"det_{v}_{E.MODELO}_img640.npz"))
        a, b = tramo(E.FORM[v])
        ref_all = E.referencia(v)
        elegidos = []
        for s, p in corridas.items():
            hist = {}
            cont = replay(fr, fps, **p, hist=hist)
            nue = [c for c in cont if c[3] == s and a <= c[2] <= b]
            ref = [r for r in ref_all if r[2] == s and a <= r[1] <= b]
            par = E.emparejar(nue, ref)
            sin = [c for j, c in enumerate(nue) if j not in par]
            mismo = [c for c in cont if c[3] == s]
            elegidos += [(c, len(sin), mismo, hist) for c in rnd.sample(sin, min(k, len(sin)))]
        elegidos.sort(key=lambda e: e[0][8])
        if not elegidos:
            continue
        cap = cv2.VideoCapture(os.path.join(E.REF_DIR, "_videos", v + ".mp4"))
        tiles = []
        for j, (c, u, mismo, hist) in enumerate(elegidos):
            filas.append(dict(video=v, k=j, Numero=c[0], Segundo=c[2], Sentido=c[3], sin_pareja=u, Marca=""))
            cap.set(cv2.CAP_PROP_POS_FRAMES, c[8] - 1)
            ok, f = cap.read()
            f = cv2.resize(f, (W, H)) if ok else np.zeros((H, W, 3), np.uint8)
            # numeracion propia del sentido para que coincida con los recorridos
            cont_s = [(q + 1, *o[1:]) for q, o in enumerate(mismo)]
            c_s = next(o for o in cont_s if o[1] == c[1] and o[8] == c[8])
            tiles.append(recorte(f, c_s, cont_s, hist, fps))
        cap.release()
        for p in range(0, len(tiles), 6):
            t = tiles[p:p + 6] + [np.zeros_like(tiles[0])] * (6 - len(tiles[p:p + 6]))
            m = np.vstack([np.hstack(t[q:q + 2]) for q in range(0, 6, 2)])
            cv2.imwrite(os.path.join(salida, f"{v}_{p // 6:02d}.jpg"), m, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(v, len(elegidos), flush=True)
    with open(os.path.join(salida, "marcas_nuevos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["video", "k", "Numero", "Segundo", "Sentido", "sin_pareja", "Marca"])
        w.writeheader(); w.writerows(filas)
    json.dump({"modelo": E.MODELO, "kw": kw}, open(os.path.join(salida, "config.json"), "w"))
