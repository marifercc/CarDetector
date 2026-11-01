import cv2 #abrir video
import numpy as np #matrices y operaciones matemáticas
from ultralytics import YOLO #importar modelo de yolo
from boxmot.trackers.bbox.ocsort import OcSort #mismo tracker que count_cars(E1) del TEC
import csv #usar csv para reporte final
import os #abrir rutas de archivos
import argparse #leer argumentos de la terminal

# --- FRANJA DE CONTEO ---
# En este video (camara de lado, a traves de la reja) los autos cruzan todo el
# cuadro horizontalmente, asi que en vez de un poligono como en el TEC se usa una
# franja vertical: se cuenta cuando un vehiculo se ve de un lado de la franja y
# despues del otro. El sentido sale solo (izq->der o der->izq).
# Ubicacion elegida con analizar_video.py (analisis_chihuahua/recorridos.jpg):
#  - a la derecha de x~650: hasta ahi llega la fila de autos detenidos, donde el
#    tracker se confunde porque los autos se tapan entre si
#  - a la izquierda de x~900: en x~940-960 la columna de piedra y el pino tapan
#    a los autos y casi todos los recorridos se cortan ahi
#  - a la derecha del tronco (x~280-340), que tambien corta algunos recorridos
FRANJA_X_IZQ = 800
FRANJA_X_DER = 860
FRANJA_Y_MIN = 280 # arriba de esto es la calle del fondo (autos muy chicos)
FRANJA_Y_MAX = 430 # abajo de esto ya es banqueta

W, H = 1280, 720 # mismo tamaño que en el TEC: las coordenadas estan en este espacio


def lado_de(xc, yc, x_izq, x_der):
    # "izq"/"der" si esta fuera de la franja (y dentro de la calle), None si no
    if not (FRANJA_Y_MIN <= yc <= FRANJA_Y_MAX):
        return None
    if xc < x_izq:
        return "izq"
    if xc > x_der:
        return "der"
    return None


def detector(video_path, mostrar, x_izq=FRANJA_X_IZQ, x_der=FRANJA_X_DER):
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    video_full_path = video_path if os.path.isabs(video_path) else os.path.join(ruta_base, video_path)
    nombre = os.path.splitext(os.path.basename(video_full_path))[0]
    # si se prueba otra franja, las salidas llevan la franja en el nombre para no
    # sobreescribir los resultados de la franja de default
    if (x_izq, x_der) != (FRANJA_X_IZQ, FRANJA_X_DER):
        nombre += f"_x{x_izq}-{x_der}"
    print(f"Abriendo video: {video_full_path}")
    # el modelo vive en la carpeta del TEC (114 MB, para no tenerlo duplicado)
    model = YOLO(os.path.join(ruta_base, "..", "CountCars_Tec", "yolo11x.pt"))
    cap = cv2.VideoCapture(video_full_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_full_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # salidas con el nombre del video para no sobreescribir resultados de otros videos
    ruta_video_salida = os.path.join(ruta_base, f"resultado_{nombre}.mp4")
    ruta_csv = os.path.join(ruta_base, f"reporte_{nombre}.csv")
    video_writer = cv2.VideoWriter(ruta_video_salida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    # mismos parametros que en el TEC
    oc_tracker = OcSort(det_thresh=0.3, max_age=30, min_hits=1, iou_threshold=0.3)

    ultimo_lado = {} # id del tracker -> ultimo lado ("izq"/"der") donde se le vio fuera de la franja
    contados = [] # (numero, id, segundo, sentido, clase)
    id_contado = {} # id del tracker -> numero de conteo (cada ID cuenta una sola vez)
    conteo_sentido = {"izq -> der": 0, "der -> izq": 0}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = cv2.resize(frame, (W, H))

        # mismos parametros de deteccion que en el TEC (ver comentarios en count_cars(E1))
        results = model.predict(
            frame, classes=[2, 3, 5, 7], verbose=False,
            conf=0.4, agnostic_nms=True, imgsz=640
        )
        det_boxes = results[0].boxes.xyxy.cpu().numpy()
        det_conf = results[0].boxes.conf.cpu().numpy()
        det_cls = results[0].boxes.cls.cpu().numpy()
        if len(det_boxes) > 0:
            dets = np.concatenate([det_boxes, det_conf.reshape(-1, 1), det_cls.reshape(-1, 1)], axis=1)
        else:
            dets = np.empty((0, 6))

        tracks = oc_tracker.update(dets, frame)  # hay que llamarlo en CADA frame, incluso sin detecciones

        cv2.rectangle(frame, (x_izq, FRANJA_Y_MIN), (x_der, FRANJA_Y_MAX), (255, 0, 0), 2)

        if len(tracks) > 0:
            nombres = results[0].names
            for box, id_obj, cls in zip(tracks.xyxy, tracks.id, tracks.cls):
                id_obj = int(id_obj)
                xmin, ymin, xmax, ymax = box
                xc, yc = int((xmin + xmax) / 2), int((ymin + ymax) / 2)

                lado = lado_de(xc, yc, x_izq, x_der)
                if lado is not None:
                    anterior = ultimo_lado.get(id_obj)
                    if anterior is not None and anterior != lado and id_obj not in id_contado:
                        sentido = "izq -> der" if lado == "der" else "der -> izq"
                        numero = len(contados) + 1
                        id_contado[id_obj] = numero
                        conteo_sentido[sentido] += 1
                        contados.append((numero, id_obj, round(frame_count / fps, 1), sentido, nombres[int(cls)]))
                    ultimo_lado[id_obj] = lado

                color = (0, 255, 0) if id_obj in id_contado else (200, 200, 200)
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                etiqueta = f"#{id_contado[id_obj]}" if id_obj in id_contado else f"id {id_obj}"
                cv2.putText(frame, etiqueta, (int(xmin), int(ymin) - 8), 0, 0.55, color, 2)

        # --- HUD ---
        segundos_transcurridos = int(frame_count / fps) if fps > 0 else 0
        tiempo_texto = f"{segundos_transcurridos // 60:02d}:{segundos_transcurridos % 60:02d}"
        cv2.rectangle(frame, (10, 40), (230, 130), (255, 0, 0), -1)
        cv2.putText(frame, f"Total: {len(contados)}", (20, 62), 0, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"izq -> der: {conteo_sentido['izq -> der']}", (20, 84), 0, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"der -> izq: {conteo_sentido['der -> izq']}", (20, 104), 0, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Tiempo: {tiempo_texto}", (20, 124), 0, 0.5, (255, 255, 255), 1)

        if mostrar:
            cv2.imshow("Chihuahua Traffic Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_writer.write(frame)
        if frame_count % int(fps * 10) == 0:
            print(f"  {frame_count / fps:.0f} s procesados, {len(contados)} contados")

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # --- RESUMEN FINAL ---
    print(f"\n=== RESUMEN FINAL ===")
    print(f"Total de vehículos: {len(contados)}")
    for sentido, cantidad in conteo_sentido.items():
        print(f"  {sentido}: {cantidad}")

    # lista de cada vehiculo contado con su segundo, para poder compararlo
    # contra un conteo a mano viendo el video
    with open(ruta_csv, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Numero", "ID tracker", "Segundo", "Sentido", "Clase"])
        csv_writer.writerows(contados)
        csv_writer.writerow([])
        csv_writer.writerow(["RESUMEN"])
        for sentido, cantidad in conteo_sentido.items():
            csv_writer.writerow([sentido, cantidad])
        csv_writer.writerow(["Total", len(contados)])

    print(f"Resultados guardados en: {ruta_csv}")
    print(f"Video procesado guardado en: {ruta_video_salida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cuenta vehiculos que cruzan una franja vertical, por sentido.")
    parser.add_argument("video_path", nargs="?", default="chihuahua.mp4",
                        help="Ruta del video a procesar. Si no se especifica, usa chihuahua.mp4 en la carpeta del script.")
    parser.add_argument("--sin-ventana", action="store_true", help="No mostrar la ventana mientras procesa (mas rapido).")
    parser.add_argument("--franja", type=int, nargs=2, metavar=("X_IZQ", "X_DER"),
                        default=[FRANJA_X_IZQ, FRANJA_X_DER],
                        help=f"Bordes x de la franja de conteo (default {FRANJA_X_IZQ} {FRANJA_X_DER}).")
    args = parser.parse_args()
    detector(args.video_path, not args.sin_ventana, *args.franja)
