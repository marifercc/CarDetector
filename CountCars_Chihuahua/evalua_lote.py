import os, csv, json, sys, collections #lote
import numpy as np #emparejar
from scipy.optimize import linear_sum_assignment #emparejamiento 1 a 1
from multiprocessing import Pool #un video por proceso
from replay_conteo import cargar, replay
from lote_14oct import leer_formulario, tramo, buscar_reporte
from f1_lote import A_MANO, f1

# F1 aproximado en el lote del 14 de octubre para una configuracion nueva del conteo,
# sin volver a correr el detector (detecciones/det_<video>_rtdetr-chihuahua_img640.npz,
# de cache_lote.sh) ni volver a revisar a mano.
# Referencia: la corrida revisada (lote_14oct_f1: reportes + revision/marcas_muestra.csv).
# Cada conteo nuevo se empareja 1 a 1 con un conteo de la referencia (mismo sentido,
# <= DT s, centro <= DIST px). Por video y sentido:
#   - emparejados: su precision se estima con las marcas de los conteos de referencia
#     revisados que siguen emparejados ("bien" = TP; doble/no_vehiculo/otro_sentido = FP)
#   - sin pareja: autos nuevos o errores nuevos; se cuentan con precision 0 (piso),
#     1 (techo) o la de los emparejados (medio)
#   FN = a mano - TP

AQUI = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(AQUI, "lote_14oct_f1")
DT, DIST = 1.0, 60
MODELO = os.environ.get("MODELO", "rtdetr-chihuahua")  # detecciones de que modelo usar
FORM = leer_formulario(os.path.join(AQUI, "conteos_14oct.csv"))


def referencia(v):
    marcas = {}
    for r in csv.DictReader(open(os.path.join(REF_DIR, "revision", "marcas_muestra.csv"), encoding="utf-8")):
        if r["video"] == v and r["Marca"]:
            marcas[int(r["Numero"])] = r["Marca"]
    ref = []
    for r in csv.reader(open(buscar_reporte(REF_DIR, v))):
        if not r:
            break
        if r[0].isdigit():
            x1, y1, x2, y2 = map(int, r[5:9])
            ref.append((int(r[0]), float(r[2]), r[3], (x1 + x2) / 2, (y1 + y2) / 2, marcas.get(int(r[0]))))
    return ref


def emparejar(nue, ref):
    costo = np.full((len(nue), len(ref)), 1e6)
    for a, c in enumerate(nue):
        cx, cy = (c[4] + c[6]) / 2, (c[5] + c[7]) / 2
        for b, r in enumerate(ref):
            dt = abs(c[2] - r[1])
            if dt <= DT:
                dd = ((cx - r[3]) ** 2 + (cy - r[4]) ** 2) ** .5
                if dd <= DIST:
                    costo[a, b] = dt * 30 + dd
    if not len(nue) or not len(ref):
        return {}
    fa, fb = linear_sum_assignment(costo)
    return {a: ref[b] for a, b in zip(fa, fb) if costo[a, b] < 1e6}


def evalua_video(v, contados):
    a, b = tramo(FORM[v])
    ref_all = referencia(v)
    filas = {}
    for s, movs in A_MANO.items():
        manual = sum(int(FORM[v][m]) for m in movs)
        nue = [c for c in contados if c[3] == s and a <= c[2] <= b]
        ref = [r for r in ref_all if r[2] == s and a <= r[1] <= b]
        par = emparejar(nue, ref)
        marcas = [r[5] for r in par.values() if r[5]]
        p_old = [r[5] for r in ref if r[5]]
        p_m = (sum(m == "bien" for m in marcas) / len(marcas)) if marcas else \
              (sum(m == "bien" for m in p_old) / len(p_old) if p_old else 0.9)
        m, u = len(par), len(nue) - len(par)
        filas[s] = dict(n=len(nue), manual=manual, m=m, u=u, p=p_m)
    return filas


def totales(res, modo):
    t = collections.Counter()
    for v, filas in res.items():
        for s, d in filas.items():
            pu = {"piso": 0, "medio": d["p"], "techo": 1}[modo]
            tp = min(d["p"] * d["m"] + pu * d["u"], d["manual"])
            t["tp"] += tp; t["fp"] += d["n"] - tp; t["fn"] += d["manual"] - tp
            t["tp_" + s] += tp; t["fp_" + s] += d["n"] - tp; t["fn_" + s] += d["manual"] - tp
    return t


_FR = {}


def _uno(args):
    v, kw = args
    if v not in _FR:
        _FR[v] = cargar(os.path.join(AQUI, "detecciones", f"det_{v}_{MODELO}_img640.npz"))
    fr, fps = _FR[v]
    return v, evalua_video(v, replay(fr, fps, **kw))


def evalua(kw, pool=None, videos=None):
    videos = videos or sorted(FORM_LOTE)
    trabajos = [(v, kw) for v in videos]
    res = dict(pool.map(_uno, trabajos) if pool else map(_uno, trabajos))
    out = {"modelo": MODELO, "kw": kw}
    for modo in ("piso", "medio", "techo"):
        t = totales(res, modo)
        out[modo] = round(f1(t["tp"], t["fp"], t["fn"]), 4)
    t = totales(res, "medio")
    for s in A_MANO:
        out[s] = round(f1(t["tp_" + s], t["fp_" + s], t["fn_" + s]), 4)
    out["por_video"] = {v: {s: (d["n"], d["manual"], d["u"], round(d["p"], 2)) for s, d in f.items()}
                        for v, f in res.items()}
    return out


# videos con detecciones guardadas (det_20261014_053527A_rtdetr-chihuahua_img640.npz -> 20261014_053527A)
FORM_LOTE = sorted({f[4:20] for f in os.listdir(os.path.join(AQUI, "detecciones"))
                    if f.startswith("det_2026") and f.endswith(f"_{MODELO}_img640.npz")})

MODO_F1 = {"min_vistos": 15, "abajo": {"conf": 0.06, "iou_threshold": 0.5, "encimado": 0.6, "min_vistos": 15}}

if __name__ == "__main__":
    # python evalua_lote.py '<json kw o lista de kw>'   (default: la config de --modo-f1)
    combos = json.loads(sys.argv[1]) if len(sys.argv) > 1 else MODO_F1
    combos = combos if isinstance(combos, list) else [combos]
    with Pool(int(os.environ.get("PROCS", 8))) as pool, open(os.path.join(AQUI, "busqueda_lote.jsonl"), "a") as f:
        for kw in combos:
            r = evalua(kw, pool)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(r["piso"], r["medio"], r["techo"], r["izq -> der"], r["der -> izq"], json.dumps(kw), flush=True)
            if len(combos) == 1:
                for v, d in r["por_video"].items():
                    print(" ", v, d)
