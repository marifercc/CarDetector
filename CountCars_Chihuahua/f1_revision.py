import csv #leer las marcas
import argparse #leer argumentos de la terminal

# F1 por vehiculo a partir de la hoja de revision (marcas_<nombre>.csv descargado
# desde revision_<nombre>/index.html) y del conteo a mano por sentido.
#   TP = conteos marcados "bien" (cada uno es un auto real contado una vez, en su sentido)
#   FP = conteos marcados doble / no_vehiculo / otro_sentido
#   FN = autos del conteo a mano que el programa no conto = manual - TP (por sentido)
# Un conteo "otro_sentido" es un auto real pero en el sentido equivocado: cuenta como
# FP en el sentido donde se conto y como FN en el otro (ya queda en manual - TP).

MANUAL_5MIN = {"izq -> der": 152, "der -> izq": 153}


def leer_marcas(ruta):
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return p, r, (2 * p * r / (p + r) if p + r else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 por vehiculo desde las marcas de la hoja de revision.")
    parser.add_argument("marcas", help="CSV descargado de la hoja de revision (marcas_<nombre>.csv).")
    parser.add_argument("--manual-izq-der", type=int, default=MANUAL_5MIN["izq -> der"])
    parser.add_argument("--manual-der-izq", type=int, default=MANUAL_5MIN["der -> izq"])
    args = parser.parse_args()
    manual = {"izq -> der": args.manual_izq_der, "der -> izq": args.manual_der_izq}

    marcas = leer_marcas(args.marcas)
    total = {"tp": 0, "fp": 0, "fn": 0}
    print(f"{'sentido':12} {'contados':>8} {'manual':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}")
    for sentido in manual:
        filas = [m for m in marcas if m["Sentido"] == sentido]
        tp = sum(m["Marca"] == "bien" for m in filas)
        fp = len(filas) - tp
        fn = max(manual[sentido] - tp, 0)
        if tp > manual[sentido]:
            print(f"  ojo: {sentido} tiene {tp} marcados bien pero el conteo a mano es {manual[sentido]}")
        p, r, f = f1(tp, fp, fn)
        print(f"{sentido:12} {len(filas):8} {manual[sentido]:6} {tp:4} {fp:4} {fn:4} {p:6.3f} {r:6.3f} {f:6.3f}")
        total["tp"] += tp; total["fp"] += fp; total["fn"] += fn
    p, r, f = f1(total["tp"], total["fp"], total["fn"])
    print(f"{'total':12} {len(marcas):8} {sum(manual.values()):6} {total['tp']:4} {total['fp']:4} {total['fn']:4} "
          f"{p:6.3f} {r:6.3f} {f:6.3f}")
    errores = {}
    for m in marcas:
        if m["Marca"] != "bien":
            errores[m["Marca"]] = errores.get(m["Marca"], 0) + 1
    print("errores:", errores or "ninguno")
