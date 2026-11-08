import csv #leer el formulario y los reportes
import os #rutas de archivos
import sys #saber con que python correr el contador
import time #medir cuanto tarda cada video
import argparse #leer argumentos de la terminal
import subprocess #correr el contador por cada video
import shutil #copiar el video de OneDrive a una carpeta local
from datetime import datetime

# Corre count_cars_chihuahua.py en videos del 14 de octubre que tienen conteo a mano
# (conteos_14oct.csv, del formulario) y compara en el mismo tramo de tiempo.
# Los reportes y trayectorias quedan en lote_14oct/; no se guarda el video de salida.
# Los videos estan en OneDrive (solo en la nube): cada uno se copia a lote_14oct/_videos
# mientras se procesa (necesita la app de OneDrive abierta) y luego se borra.

CARPETA_VIDEOS = (r"C:\Users\marif\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey"
                  r"\Shortcuts\IMPLAN - Datos\Videos\14-10-2026\Teofilo Borunda y Pedro de Zulouaga")
MOVIMIENTOS = ["M4", "M3", "M2", "M1", "M9_4B", "M9_4A", "M9_3", "M9_2", "M6", "M5", "M8", "M7"]

# muestra repartida en el dia: madrugada (oscuro), hora pico, mediodia y tarde
MUESTRA = ["053527A", "060527A", "063027A", "065027A", "071027A", "074527A", "081027A", "084527A",
           "091527A", "100027A", "104027A", "113027A", "122527A", "135527A", "143527A", "154527A"]


def leer_formulario(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return {fila["video"]: fila for fila in csv.DictReader(f)}


def tramo(fila):
    # segundos del video que se contaron a mano; el video empieza a la hora de su nombre
    inicio_video = datetime.strptime(fila["video"][9:15], "%H%M%S")
    a = (datetime.strptime(fila["inicio"], "%H:%M:%S") - inicio_video).total_seconds()
    b = (datetime.strptime(fila["fin"], "%H:%M:%S") - inicio_video).total_seconds()
    return max(a, 0), b


def contar_en_tramo(ruta_reporte, a, b):
    conteo = {"izq -> der": 0, "der -> izq": 0}
    with open(ruta_reporte, newline="") as f:
        for fila in csv.reader(f):
            if not fila:
                break
            if fila[0] != "Numero" and a <= float(fila[2]) <= b:
                conteo[fila[3]] += 1
    return conteo


def buscar_video(nombre):
    for raiz, _, archivos in os.walk(CARPETA_VIDEOS):
        if nombre + ".mp4" in archivos:
            return os.path.join(raiz, nombre + ".mp4")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corre el contador en videos del 14 de octubre y compara con el formulario.")
    parser.add_argument("--todos", action="store_true", help="Todos los videos del formulario, no solo la muestra.")
    parser.add_argument("--solo-tabla", action="store_true", help="No correr nada, solo armar la tabla con los reportes que ya hay.")
    parser.add_argument("extra", nargs="*", help="Opciones extra para el contador, ej. --modelo rtdetr-chihuahua (despues de --).")
    args = parser.parse_args()
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    salida = os.path.join(ruta_base, "lote_14oct")
    formulario = leer_formulario(os.path.join(ruta_base, "conteos_14oct.csv"))
    videos = sorted(formulario) if args.todos else [f"20261014_{m}" for m in MUESTRA]

    # ya corridos: el lote se puede interrumpir (Ctrl+C) y seguir despues donde se quedo
    pendientes = [n for n in videos if not os.path.exists(os.path.join(salida, f"reporte_{n}.csv"))]
    print(f"{len(videos)} videos, {len(videos) - len(pendientes)} ya corridos, {len(pendientes)} por correr", flush=True)
    if args.solo_tabla:
        pendientes = []
    duraciones = []
    for k, nombre in enumerate(pendientes, 1):
        ruta = buscar_video(nombre)
        if ruta is None:
            print(f"[{k}/{len(pendientes)}] {nombre}: no se encontro el video", flush=True)
            continue
        t0 = time.time()
        faltan = ""
        if duraciones:
            minutos = sum(duraciones) / len(duraciones) * (len(pendientes) - k + 1) / 60
            faltan = f" | faltan ~{minutos // 60:.0f} h {minutos % 60:.0f} min"
        print(f"[{k}/{len(pendientes)}] {nombre}: bajando de OneDrive...{faltan}", end="", flush=True)
        # se copia a una carpeta local: OpenCV no dispara la descarga de un archivo que
        # esta solo en la nube, y asi el contador lee un archivo completo
        local = os.path.join(salida, "_videos", nombre + ".mp4")
        os.makedirs(os.path.dirname(local), exist_ok=True)
        # OneDrive a veces falla una descarga ("Invalid argument") y a la siguiente sale bien
        for intento in range(1, 4):
            try:
                shutil.copyfile(ruta, local)
                break
            except OSError as e:
                if intento == 3:
                    print(f"\n\nNo se pudo bajar {nombre} de OneDrive: {e}")
                    print("Revisa que la app de OneDrive este abierta y con sesion iniciada, y vuelve a correr el lote.")
                    sys.exit(1)
                print(f" (fallo la descarga, reintento {intento}/2 en 30 s)", end="", flush=True)
                time.sleep(30)
        proceso = subprocess.Popen([sys.executable, "-u", os.path.join(ruta_base, "count_cars_chihuahua.py"), local,
                                    "--sin-ventana", "--sin-video", "--trayectorias", "--carpeta-salida", salida,
                                    *args.extra],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        for linea in proceso.stdout:
            # el contador imprime "  120 s procesados, 45 contados" cada 10 s de video
            if "procesados" in linea:
                print(f"\r[{k}/{len(pendientes)}] {nombre}: {linea.strip()} de ~300 s{faltan}      ", end="", flush=True)
        proceso.wait()
        os.remove(local)
        duraciones.append(time.time() - t0)
        estado = "listo" if proceso.returncode == 0 else f"ERROR (codigo {proceso.returncode})"
        print(f"\r[{k}/{len(pendientes)}] {nombre}: {estado} en {duraciones[-1] / 60:.1f} min{faltan}" + " " * 30,
              flush=True)

    filas = []
    for nombre in videos:
        ruta_reporte = os.path.join(salida, f"reporte_{nombre}.csv")
        if not os.path.exists(ruta_reporte):
            continue
        f = formulario[nombre]
        a, b = tramo(f)
        prog = contar_en_tramo(ruta_reporte, a, b)
        filas.append({"video": nombre, "tramo_s": f"{a:.0f}-{b:.0f}", "prog_izq_der": prog["izq -> der"],
                      "prog_der_izq": prog["der -> izq"], **{m: f[m] for m in MOVIMIENTOS}, "nota": f["nota"]})
    if not filas:
        print("Todavia no hay reportes para armar la tabla")
        sys.exit()
    ruta_tabla = os.path.join(salida, "tabla.csv")
    with open(ruta_tabla, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"{len(filas)} videos en {ruta_tabla}")
