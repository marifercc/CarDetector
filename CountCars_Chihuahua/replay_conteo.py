import numpy as np #matrices
import torch, torchvision #NMS igual que el contador
from boxmot.trackers.bbox.ocsort import OcSort #tracker
from count_cars_chihuahua import (W, H, Y_SENTIDOS, FRANJA_X_IZQ, FRANJA_X_DER, FRANJA_Y_MIN, FRANJA_Y_MAX, sin_anidados, dentro_de, lado_de)

# Repite el contador de count_cars_chihuahua.py (filtros + OC-SORT + regla de la
# franja) sobre las detecciones guardadas por cache_detecciones.py, sin volver a
# correr el detector. Sirve para probar parametros del tracker y del conteo en
# segundos. Con los parametros por default da los mismos conteos que el contador.

BASE = dict(conf=0.08, conf_arriba=0.4, iou=0.7, anidados=0.9, max_age=90, min_hits=1, iou_threshold=0.3,
            delta_t=3, inertia=0.2, use_byte=False, asso_func="iou", encimado=0.8, min_vistos=0,
            x_izq=FRANJA_X_IZQ, x_der=FRANJA_X_DER, y_min=FRANJA_Y_MIN, y_max=FRANJA_Y_MAX,
            encimado_mutuo=False,  # doble tambien si la caja ya contada queda dentro de la nueva
            conf_id=0.0,
            # relevo: un ID que nace a <= relevo_dist (fraccion de su ancho) de donde se perdio un ID ya
            # contado del mismo sentido, <= relevo_gap cuadros antes, es el mismo auto (doble). 0 = apagado
            relevo_gap=0, relevo_dist=0.6,
            # carril: der -> izq solo si el centro esta arriba de y_der_max; izq -> der solo abajo de y_izq_min
            y_der_max=10000, y_izq_min=0,
            # solo se cuenta si al cruzar el centro esta a <= x_margen px de la franja (saltos de ID)
            x_margen=10000,
            # puente: un ID no contado que se pierde en un lado (tapado en la franja) y un ID nuevo que
            # aparece del otro lado <= puente_gap cuadros despues, a <= puente_dy px en y y <= puente_dx
            # px en x, se cuentan como un cruce. 0 = apagado
            puente_gap=0, puente_dy=30, puente_dx=300, puente_min=5)  # solo se cuenta un ID cuya mejor confianza llego a esto (reflejos, calle vacia)


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
    # la clave lleva el video (id de la lista de cuadros): si no, un proceso que corre
    # varios videos devolveria el conteo de otro video
    clave = (id(frames), repr(sorted(kw.items())))
    if clave not in _MEMO:
        _MEMO[clave] = replay(frames, fps, **kw)
    return _MEMO[clave]


def relevo(i, s, primero, ultimo, traza, id_contado, contados, p):
    n0, b0 = primero[i]
    c0 = ((b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2)
    lim = p["relevo_dist"] * (b0[2] - b0[0])
    for j, k in id_contado.items():
        if j == i or contados[k - 1][3] != s:
            continue
        nj, bj = ultimo[j]
        # se perdio poco antes de que naciera i (o siguieron juntos un rato: nj >= n0)
        if n0 - p["relevo_gap"] <= nj and primero[j][0] < n0:
            # si j seguia vivo cuando nacio i se compara donde estaba j en ese cuadro
            bj_n0 = bj if nj <= n0 else traza[j].get(n0, bj)
            cj = ((bj_n0[0] + bj_n0[2]) / 2, (bj_n0[1] + bj_n0[3]) / 2)
            if ((c0[0] - cj[0]) ** 2 + (c0[1] - cj[1]) ** 2) ** .5 <= lim:
                return k
    return None


def puente(i, lado, xc, yc, n, perdidos, cajas_frame, vistos, p):
    mejor, dmin = None, 1e9
    for j, (nj, lj, xj, yj) in perdidos.items():
        if j == i or j in cajas_frame or lj == lado or vistos.get(j, 0) < p["puente_min"]:
            continue
        if n - nj <= p["puente_gap"] and abs(yc - yj) <= p["puente_dy"] and abs(xc - xj) <= p["puente_dx"]:
            d = abs(xc - xj) + 3 * abs(yc - yj)
            if d < dmin:
                mejor, dmin = j, d
    return mejor


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
    hist = kw.pop("hist", None) # dict opcional: id -> lista de (cuadro, x1, y1, x2, y2) para estudiar recorridos
    p = {**BASE, **kw}
    tracker = nuevo_tracker(p)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    ultimo_lado, id_contado, contados, vistos, conf_max = {}, {}, [], {}, {}
    perdidos = {}  # id no contado -> (cuadro, lado, xc, yc) la ultima vez que tuvo lado
    primero, ultimo, traza = {}, {}, {}  # id -> (cuadro, caja) al nacer / ultima vez; id -> {cuadro: caja}
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
        for i, c in zip(tracks.id, tracks.conf):
            conf_max[int(i)] = max(conf_max.get(int(i), 0.0), float(c))
        cajas_frame = {i: b for i, b in filas}
        for i, (x1, y1, x2, y2) in filas:
            vistos[i] = vistos.get(i, 0) + 1
            primero.setdefault(i, (n, (x1, y1, x2, y2)))
            ultimo[i] = (n, (x1, y1, x2, y2))
            if p["relevo_gap"] > 0:
                traza.setdefault(i, {})[n] = (x1, y1, x2, y2)
            if hist is not None:
                hist.setdefault(i, []).append((n, int(x1), int(y1), int(x2), int(y2)))
            xc, yc = int((x1 + x2) / 2), int((y1 + y2) / 2)
            lado = None if not (p["y_min"] <= yc <= p["y_max"]) else                 "izq" if xc < p["x_izq"] else "der" if xc > p["x_der"] else None
            if lado is None:
                continue
            a = ultimo_lado.get(i)
            if a is None and p["puente_gap"] > 0 and i not in id_contado:
                j = puente(i, lado, xc, yc, n, perdidos, cajas_frame, vistos, p)
                if j is not None:
                    del perdidos[j]
                    a = "der" if lado == "izq" else "izq"  # se hereda el lado del ID perdido
                    vistos[i] = max(vistos[i], p["min_vistos"])
            if a is not None and a != lado and i not in id_contado and vistos[i] >= p["min_vistos"] and conf_max[i] >= p["conf_id"]:
                s = "izq -> der" if lado == "der" else "der -> izq"
                if (s == "der -> izq" and yc > p["y_der_max"]) or (s == "izq -> der" and yc < p["y_izq_min"])                         or not (p["x_izq"] - p["x_margen"] <= xc <= p["x_der"] + p["x_margen"]):
                    ultimo_lado[i] = lado
                    continue
                caja = (int(x1), int(y1), int(x2), int(y2))
                doble = next((id_contado[j] for j, b in cajas_frame.items()
                              if j != i and j in id_contado and contados[id_contado[j] - 1][3] == s
                              and max(dentro_de(caja, b), dentro_de(b, caja) if p["encimado_mutuo"] else 0) >= p["encimado"]), None)
                if doble is None and p["relevo_gap"] > 0:
                    doble = relevo(i, s, primero, ultimo, traza, id_contado, contados, p)
                if doble is not None:
                    id_contado[i] = doble
                else:
                    # al final: cuadro y cajas de los ya contados en ese cuadro (para revisar)
                    vecinos = [(id_contado[j], tuple(int(v) for v in b)) for j, b in cajas_frame.items()
                               if j in id_contado]
                    contados.append((len(contados) + 1, i, round(n / fps, 1), s, *caja, n, vecinos))
                    id_contado[i] = len(contados)
            ultimo_lado[i] = lado
            if p["puente_gap"] > 0 and i not in id_contado:
                perdidos[i] = (n, lado, xc, yc)
        if p["puente_gap"] > 0:
            for j in [j for j, q in perdidos.items() if n - q[0] > p["puente_gap"] or j in id_contado]:
                del perdidos[j]
    return contados
