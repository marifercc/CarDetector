import os, csv, json, sys, collections #lote
from multiprocessing import Pool #un video por proceso
from replay_conteo import cargar, replay
from lote_14oct import tramo
from f1_lote import A_MANO, f1
import evalua_lote as E

# Como evalua_lote.py pero con una segunda referencia revisada: los conteos de la
# configuracion base de revision_nuevos/config.json que no se emparejaron con la corrida
# revisada original, con su muestra marcada (revision_nuevos/marcas_nuevos.csv +
# marcas_claude.json). Para una configuracion nueva, por video y sentido:
#   - emparejados con la referencia original: precision de sus marcas (como antes)
#   - sin pareja: se emparejan con los conteos marcados de la muestra nueva; su precision
#     es la de esas marcas (si hay < MIN_MARCAS en el grupo, la del sentido en todo el lote)
# Asi una regla que quita dobles sube la precision de los sin pareja.
# MODELO=rtdetr-chihuahua2 python evalua_lote2.py '<json kw o lista>'

REV = os.path.join(E.AQUI, "revision_nuevos")
MIN_MARCAS = 3


def marcas_nuevas():
    m = json.load(open(os.path.join(REV, "marcas_claude.json")))
    out = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(REV, "marcas_nuevos.csv"), encoding="utf-8")):
        lista = m.get(r["video"], [])
        k = int(r["k"])
        if k < len(lista):
            out[r["video"]][(r["Sentido"], int(r["Numero"]))] = lista[k]
    return out


CONF = json.load(open(os.path.join(REV, "config.json")))
MARCAS = marcas_nuevas()


def corridas(kw):
    base = {a: b for a, b in kw.items() if a != "abajo"}
    return {"der -> izq": base, "izq -> der": {**base, **kw.get("abajo", {})}}


_FR, _REF2 = {}, {}


def ref2(v, fr, fps):
    # conteos marcados de la muestra nueva, en el formato de E.referencia
    ruta = os.path.join(REV, f"ref2_{v}.json")
    if v not in _REF2 and os.path.exists(ruta):
        _REF2[v] = [tuple(r) for r in json.load(open(ruta))]
    if v not in _REF2:
        out = []
        for s, p in corridas(CONF["kw"]).items():
            for c in replay(fr, fps, **p):
                mk = MARCAS[v].get((s, c[0]))
                if c[3] == s and mk:
                    out.append((c[0], c[2], s, (c[4] + c[6]) / 2, (c[5] + c[7]) / 2, mk))
        _REF2[v] = out
        json.dump(out, open(ruta, "w"))
    return _REF2[v]


def _uno(args):
    v, kw = args
    if v not in _FR:
        _FR[v] = cargar(os.path.join(E.AQUI, "detecciones", f"det_{v}_{E.MODELO}_img640.npz"))
    fr, fps = _FR[v]
    cont = replay(fr, fps, **kw)
    a, b = tramo(E.FORM[v])
    r1, r2 = E.referencia(v), ref2(v, fr, fps)
    filas = {}
    for s in A_MANO:
        manual = sum(int(E.FORM[v][m]) for m in A_MANO[s])
        nue = [c for c in cont if c[3] == s and a <= c[2] <= b]
        ref = [r for r in r1 if r[2] == s and a <= r[1] <= b]
        par = E.emparejar(nue, ref)
        mk = [r[5] for r in par.values() if r[5]]
        p_old = [r[5] for r in ref if r[5]]
        p_m = (sum(m == "bien" for m in mk) / len(mk)) if mk else \
              (sum(m == "bien" for m in p_old) / len(p_old) if p_old else 0.9)
        sin = [c for j, c in enumerate(nue) if j not in par]
        par2 = E.emparejar(sin, [r for r in r2 if r[2] == s])
        mk2 = [r[5] for r in par2.values()]
        filas[s] = dict(n=len(nue), manual=manual, m=len(par), u=len(sin), p=p_m,
                        bien2=sum(m == "bien" for m in mk2), marc2=len(mk2))
    return v, filas


def evalua(kw, pool):
    res = dict(pool.map(_uno, [(v, kw) for v in E.FORM_LOTE]))
    # precision de los sin pareja por sentido en todo el lote (respaldo para grupos con pocas marcas)
    pool_s = {s: (sum(f[s]["bien2"] for f in res.values()), sum(f[s]["marc2"] for f in res.values())) for s in A_MANO}
    t = collections.Counter()
    for v, filas in res.items():
        for s, d in filas.items():
            pu = d["bien2"] / d["marc2"] if d["marc2"] >= MIN_MARCAS else pool_s[s][0] / max(pool_s[s][1], 1)
            tp = min(d["p"] * d["m"] + pu * d["u"], d["manual"])
            for k in ("", "_" + s):
                t["tp" + k] += tp; t["fp" + k] += d["n"] - tp; t["fn" + k] += d["manual"] - tp
    out = {"modelo": E.MODELO, "kw": kw, "f1": round(f1(t["tp"], t["fp"], t["fn"]), 4)}
    for s in A_MANO:
        out[s] = round(f1(t["tp_" + s], t["fp_" + s], t["fn_" + s]), 4)
        out["p_sin_" + s] = round(pool_s[s][0] / max(pool_s[s][1], 1), 3)
    out["n"] = sum(d["n"] for f in res.values() for d in f.values())
    out["manual"] = sum(d["manual"] for f in res.values() for d in f.values())
    return out


if __name__ == "__main__":
    combos = json.loads(sys.argv[1]) if len(sys.argv) > 1 else CONF["kw"]
    combos = combos if isinstance(combos, list) else [combos]
    with Pool(int(os.environ.get("PROCS", 8))) as pool, open(os.path.join(E.AQUI, "busqueda_lote2.jsonl"), "a") as f:
        for kw in combos:
            r = evalua(kw, pool)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(r["f1"], r["izq -> der"], r["der -> izq"], r["p_sin_izq -> der"], r["p_sin_der -> izq"],
                  r["n"], r["manual"], json.dumps(kw), flush=True)
