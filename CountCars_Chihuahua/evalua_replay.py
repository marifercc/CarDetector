import csv, json, os #leer marcas y referencias
import numpy as np #emparejar
from scipy.optimize import linear_sum_assignment #emparejamiento 1 a 1

# F1 aproximado de una corrida nueva sin volver a revisar todo a mano: cada conteo
# nuevo se empareja con un conteo ya marcado de la corrida evaluada
# (reporte + marcas_chihuahua_5min_modelortdetr-chihuahua.csv) si es del mismo
# sentido, a <= DT s y su centro a <= DIST px.
#   - pareja "bien"            -> TP
#   - pareja "doble" cuyo auto original ya no se conto -> TP (es el mismo auto, una vez)
#   - pareja doble / no_vehiculo / otro_sentido -> FP
#   - sin pareja               -> desconocido (puede ser un auto que antes faltaba)
# Da F1 con desconocidos como error (piso) y como acierto (techo).

AQUI = os.path.dirname(os.path.abspath(__file__))
MANUAL = {"izq -> der": 152, "der -> izq": 153}
DT, DIST = 1.0, 60


def referencia():
    marcas = {int(r["Numero"]): r["Marca"] for r in csv.DictReader(
        open(os.path.join(AQUI, "marcas_chihuahua_5min_modelortdetr-chihuahua.csv"), encoding="utf-8-sig"))}
    ref = []
    for r in csv.reader(open(os.path.join(AQUI, "corrida_tray", "reporte_chihuahua_5min_modelortdetr-chihuahua.csv"))):
        if not r:
            break
        if r[0].isdigit():
            x1, y1, x2, y2 = map(int, r[5:9])
            ref.append((int(r[0]), float(r[2]), r[3], (x1 + x2) / 2, (y1 + y2) / 2, marcas[int(r[0])]))
    original = {int(k): v for k, v in json.load(open(os.path.join(AQUI, "dobles_original.json"))).items()}
    return ref, original


REF, ORIGINAL = referencia()


def evalua(contados, detalle=False):
    tot = {"tp": 0, "fp": 0, "unk": 0}
    por = {}
    desconocidos = []
    for s in MANUAL:
        nue = [c for c in contados if c[3] == s]
        ref = [r for r in REF if r[2] == s]
        costo = np.full((len(nue), len(ref)), 1e6)
        for a, c in enumerate(nue):
            cx, cy = (c[4] + c[6]) / 2, (c[5] + c[7]) / 2
            for b, r in enumerate(ref):
                dt = abs(c[2] - r[1]); dd = ((cx - r[3]) ** 2 + (cy - r[4]) ** 2) ** .5
                if dt <= DT and dd <= DIST:
                    costo[a, b] = dt * 30 + dd
        fa, fb = linear_sum_assignment(costo) if len(nue) and len(ref) else ([], [])
        par = {a: ref[b] for a, b in zip(fa, fb) if costo[a, b] < 1e6}
        bien_usados = {r[0] for r in par.values() if r[5] == "bien"}
        tp = fp = unk = 0
        for a, c in enumerate(nue):
            r = par.get(a)
            if r is None:
                unk += 1; desconocidos.append(c)
            elif r[5] == "bien":
                tp += 1
            elif r[5] == "doble" and ORIGINAL.get(r[0]) not in bien_usados:
                tp += 1; bien_usados.add(ORIGINAL.get(r[0]))
            else:
                fp += 1
        tp = min(tp, MANUAL[s])
        por[s] = (len(nue), tp, fp, unk)
        tot["tp"] += tp; tot["fp"] += fp; tot["unk"] += unk
    man = sum(MANUAL.values())
    tp, fp, unk = tot["tp"], tot["fp"], tot["unk"]
    piso = 2 * tp / (2 * tp + fp + unk + (man - tp))
    tp2 = min(tp + unk, man)
    techo = 2 * tp2 / (2 * tp2 + fp + (man - tp2))
    res = {"n": len(contados), "tp": tp, "fp": fp, "unk": unk, "f1_piso": round(piso, 4), "f1_techo": round(techo, 4),
           "izq": por["izq -> der"], "der": por["der -> izq"]}
    if detalle:
        res["desconocidos"] = desconocidos
    return res
