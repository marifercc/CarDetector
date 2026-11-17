import os, csv, random, argparse, collections #lote
from lote_14oct import leer_formulario, tramo, buscar_reporte

# F1 por vehiculo en el lote del 14 de octubre a partir de la revision por muestra
# (revision_lote.py -> revision/marcas_muestra.csv con la columna Marca llena).
# Por video y sentido:
#   precision p = marcados "bien" / revisados (muestra al azar de los conteos)
#   TP = p * contados en el tramo, FP = contados - TP, FN = max(a mano - TP, 0)
# A mano: izq -> der = M4 + M9_4A + M5 + M7; der -> izq = M3.
# El intervalo sale de remuestrear las marcas de cada video y sentido (bootstrap).

A_MANO = {"izq -> der": ["M4", "M9_4A", "M5", "M7"], "der -> izq": ["M3"]}


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0


def contados_en_tramo(rep, a, b):
    n = collections.Counter()
    for r in csv.reader(open(rep)):
        if not r:
            break
        if r[0].isdigit() and a <= float(r[2]) <= b:
            n[r[3]] += 1
    return n


def totales(datos, marcas_por):
    tot = collections.Counter()
    filas = []
    for (v, s), d in datos.items():
        m = marcas_por.get((v, s), [])
        p = sum(x == "bien" for x in m) / len(m) if m else 0
        tp = p * d["n"]
        fp = d["n"] - tp
        fn = max(d["manual"] - tp, 0)
        tot["tp"] += tp; tot["fp"] += fp; tot["fn"] += fn
        filas.append((v, s, d["n"], d["manual"], len(m), p, tp, fp, fn))
    return tot, filas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    carpeta = os.path.join(base, args.carpeta)
    formulario = leer_formulario(os.path.join(base, "conteos_14oct.csv"))
    marcas = [r for r in csv.DictReader(open(os.path.join(carpeta, "revision", "marcas_muestra.csv"), encoding="utf-8"))
              if r["Marca"]]
    marcas_por = collections.defaultdict(list)
    for r in marcas:
        marcas_por[(r["video"], r["Sentido"])].append(r["Marca"])
    datos = {}
    for v in sorted({r["video"] for r in marcas}):
        a, b = tramo(formulario[v])
        n = contados_en_tramo(buscar_reporte(carpeta, v), a, b)
        for s, movs in A_MANO.items():
            datos[(v, s)] = {"n": n[s], "manual": sum(int(formulario[v][m]) for m in movs)}

    tot, filas = totales(datos, marcas_por)
    print(f"{'video':18} {'sentido':10} {'cont':>5} {'mano':>5} {'rev':>4} {'P':>5} {'TP':>6} {'FP':>6} {'FN':>6} {'F1':>5}")
    por_video = collections.defaultdict(collections.Counter)
    for v, s, n, man, k, p, tp, fp, fn in filas:
        print(f"{v:18} {s:10} {n:5} {man:5} {k:4} {p:5.2f} {tp:6.1f} {fp:6.1f} {fn:6.1f} {f1(tp, fp, fn):5.3f}")
        por_video[v].update({"tp": tp, "fp": fp, "fn": fn})
    print()
    for v, c in por_video.items():
        print(f"{v:18} F1 {f1(c['tp'], c['fp'], c['fn']):.3f}")
    por_sentido = collections.defaultdict(collections.Counter)
    for v, s, n, man, k, p, tp, fp, fn in filas:
        por_sentido[s].update({"tp": tp, "fp": fp, "fn": fn})
    for s, c in por_sentido.items():
        print(f"{s:18} F1 {f1(c['tp'], c['fp'], c['fn']):.3f}")

    rnd = random.Random(1)
    muestras = []
    for _ in range(args.boot):
        mb = {k: [rnd.choice(m) for _ in m] for k, m in marcas_por.items()}
        t, _ = totales(datos, mb)
        muestras.append(f1(t["tp"], t["fp"], t["fn"]))
    muestras.sort()
    lo, hi = muestras[int(0.025 * args.boot)], muestras[int(0.975 * args.boot)]
    print(f"\nTOTAL  TP {tot['tp']:.0f}  FP {tot['fp']:.0f}  FN {tot['fn']:.0f}  "
          f"F1 {f1(tot['tp'], tot['fp'], tot['fn']):.3f}  (IC 95% {lo:.3f} - {hi:.3f})  revisados {len(marcas)}")
