import numpy as np #matrices
import torch, torchvision #NMS igual que el contador
from boxmot.trackers.bbox.ocsort import OcSort #tracker
from count_cars_chihuahua import (W, H, Y_SENTIDOS, FRANJA_X_IZQ, FRANJA_X_DER, sin_anidados, dentro_de, lado_de)

# Repite el contador de count_cars_chihuahua.py (filtros + OC-SORT + regla de la
# franja) sobre las detecciones guardadas por cache_detecciones.py, sin volver a
# correr el detector. Sirve para probar parametros del tracker y del conteo en
# segundos. Con los parametros por default da los mismos conteos que el contador.

BASE = dict(conf=0.08, conf_arriba=0.4, iou=0.7, anidados=0.9, max_age=90, min_hits=1, iou_threshold=0.3,
            delta_t=3, inertia=0.2, use_byte=False, asso_func="iou", encimado=0.8, min_vistos=0,
            x_izq=FRANJA_X_IZQ, x_der=FRANJA_X_DER)


def cargar(ruta):
    z = np.load(ruta)
    cajas, cuadro = z["cajas"], z["cuadro"]
    corte = np.searchsorted(cuadro, np.arange(1, int(z["n"]) + 2))
    return [cajas[corte[i]:corte[i + 1]] for i in range(int(z["n"]))], float(z["fps"])


def nuevo_tracker(p):
    return OcSort(det_thresh=p["conf"], max_age=p["max_age"], min_hits=p["min_hits"],
                  iou_threshold=p["iou_threshold"], delta_t=p["delta_t"], inertia=p["inertia"],
                  use_byte=p["use_byte"], min_conf=min(0.1, p["conf"]), asso_func=p["asso_func"],
                  Q_xy_scaling=p.get("q_xy", 0.01), Q_s_scaling=p.get("q_s", 0.0001))


_MEMO = {}


def _memo(frames, fps, kw):
    clave = repr(sorted(kw.items()))
    if clave not in _MEMO:
        _MEMO[clave] = replay(frames, fps, **kw)
    return _MEMO[clave]


def replay(frames, fps, **kw):
    # Con abajo={...} corren dos trackers completos (cada uno con todas las
    # detecciones): der -> izq sale del de BASE + kw e izq -> der del que lleva
    # ademas lo de abajo. Asi se ajusta izq -> der sin mover der -> izq.
    kw = dict(kw)
    abajo = kw.pop("abajo", None)
    if abajo is not None:
        c1 = _memo(frames, fps, kw)
        c2 = replay(frames, fps, **{**kw, **abajo})
        juntos = sorted([c for c in c1 if c[3] == "der -> izq"] + [c for c in c2 if c[3] == "izq -> der"],
                        key=lambda c: c[2])
        return [(k + 1, *c[1:]) for k, c in enumerate(juntos)]
    p = {**BASE, **kw}
    tracker = nuevo_tracker(p)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    ultimo_lado, id_contado, contados, vistos = {}, {}, [], {}
    for n, d in enumerate(frames, start=1):
        d = d[d[:, 4] >= p["conf"]]
        if len(d) > 1:
            q = torchvision.ops.nms(torch.from_numpy(d[:, :4]), torch.from_numpy(d[:, 4]), p["iou"]).numpy()
            d = d[q]
        if p["conf_arriba"] > p["conf"] and len(d):
            arriba = (d[:, 1] + d[:, 3]) / 2 < Y_SENTIDOS
            d = d[~arriba | (d[:, 4] >= p["conf_arriba"])]
        if p["anidados"] > 0 and len(d) > 1:
            d = d[sin_anidados(d[:, :4], p["anidados"], Y_SENTIDOS)]
        tracks = tracker.update(d if len(d) else np.empty((0, 6)), img)
        filas = [(int(i), tuple(float(v) for v in b)) for b, i in zip(tracks.xyxy, tracks.id)]
        cajas_frame = {i: b for i, b in filas}
        for i, (x1, y1, x2, y2) in filas:
            vistos[i] = vistos.get(i, 0) + 1
            lado = lado_de(int((x1 + x2) / 2), int((y1 + y2) / 2), p["x_izq"], p["x_der"])
            if lado is None:
                continue
            a = ultimo_lado.get(i)
            if a is not None and a != lado and i not in id_contado and vistos[i] >= p["min_vistos"]:
                s = "izq -> der" if lado == "der" else "der -> izq"
                caja = (int(x1), int(y1), int(x2), int(y2))
                doble = next((id_contado[j] for j, b in cajas_frame.items()
                              if j != i and j in id_contado and contados[id_contado[j] - 1][3] == s
                              and dentro_de(caja, b) >= p["encimado"]), None)
                if doble is not None:
                    id_contado[i] = doble
                else:
                    # al final: cuadro y cajas de los ya contados en ese cuadro (para revisar)
                    vecinos = [(id_contado[j], tuple(int(v) for v in b)) for j, b in cajas_frame.items()
                               if j in id_contado]
                    contados.append((len(contados) + 1, i, round(n / fps, 1), s, *caja, n, vecinos))
                    id_contado[i] = len(contados)
            ultimo_lado[i] = lado
    return contados
