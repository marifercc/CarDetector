import json, os, sys #etiquetas

# Limpia etiquetas automaticas (etiquetas_lote.py / etiquetas_auto.py) antes de entrenar.
# Las trayectorias con conf baja dejan cajas que no son autos y que el modelo
# aprenderia como "vehiculo":
#  - postes y letreros (sobre todo el de la orilla izquierda): cajas mas altas que anchas
#  - el camellon amarillo y sombras: cajas muy planas
#  - cajas diminutas
#  - dos cajas casi iguales sobre el mismo auto
# Vistos de lado, los autos miden de alto ~0.3 a ~0.9 de su ancho (motos ~1).
# python limpiar_etiquetas.py etiquetado/etiquetas_lote.json -> etiquetado/etiquetas_lote_limpias.json

MAX_ALTO_ANCHO = 1.2
MIN_ALTO_ANCHO = 0.25
MIN_ALTO = 10
IOU_DUPLICADO = 0.6


def iou(a, b):
    ancho = min(a[2], b[2]) - max(a[0], b[0])
    alto = min(a[3], b[3]) - max(a[1], b[1])
    if ancho <= 0 or alto <= 0:
        return 0.0
    inter = ancho * alto
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def limpiar(cajas):
    buenas = []
    for c in cajas:
        w, h = c[2] - c[0], c[3] - c[1]
        if w <= 0 or h < MIN_ALTO or not (MIN_ALTO_ANCHO <= h / w <= MAX_ALTO_ANCHO):
            continue
        # poste con letrero de la orilla izquierda (x < ~130); un auto cortado ahi esta
        # lejos de la franja de conteo y no importa perderlo
        if c[2] < 130 and h / w > 0.8:
            continue
        buenas.append(c)
    # de dos casi iguales se queda la mas grande
    buenas.sort(key=lambda c: -(c[2] - c[0]) * (c[3] - c[1]))
    quedan = []
    for c in buenas:
        if all(iou(c, q) < IOU_DUPLICADO for q in quedan):
            quedan.append(c)
    return quedan


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else "etiquetado/etiquetas_lote.json"
    d = json.load(open(entrada))
    antes = sum(len(v) for v in d["imagenes"].values())
    d["imagenes"] = {k: limpiar(v) for k, v in d["imagenes"].items()}
    despues = sum(len(v) for v in d["imagenes"].values())
    salida = entrada.replace(".json", "_limpias.json")
    json.dump(d, open(salida, "w"))
    print(f"{len(d['imagenes'])} imagenes, {antes} -> {despues} cajas -> {salida}")
