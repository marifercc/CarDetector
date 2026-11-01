import cv2 #abrir video
import numpy as np #matrices y operaciones matemáticas
from ultralytics import YOLO, RTDETR #importar modelo de yolo (o RT-DETR)
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


def sin_anidados(boxes, umbral=0.9, y_max=None):
    # RT-DETR a veces pone un cuadro extra sobre una parte del mismo auto (ej. el
    # techo o la mitad de una camioneta). Si un cuadro queda >= umbral dentro de
    # otro mas grande, es un duplicado y se quita. Devuelve que cuadros se quedan.
    # Con y_max solo se filtran cuadros con centro arriba de y_max: en los carriles
    # cercanos (izq -> der) un cuadro adentro de otro suele ser un auto tapado real.
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    quedan = np.ones(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if y_max is not None and (boxes[i, 1] + boxes[i, 3]) / 2 > y_max:
            continue
        for j in range(len(boxes)):
            if i == j or areas[j] <= areas[i]:
                continue
            ancho = min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0])
            alto = min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1])
            if ancho > 0 and alto > 0 and ancho * alto >= umbral * areas[i]:
                quedan[i] = False
                break
    return quedan


def detector(video_path, mostrar, x_izq=FRANJA_X_IZQ, x_der=FRANJA_X_DER, max_age=30, imgsz=640, iou=0.7, conf=0.4, modelo="yolo11x", anidados=None, anidados_y=None):
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    video_full_path = video_path if os.path.isabs(video_path) else os.path.join(ruta_base, video_path)
    nombre = os.path.splitext(os.path.basename(video_full_path))[0]
    # si se prueba otra configuracion, las salidas llevan lo que cambio en el nombre
    # para no sobreescribir los resultados de la configuracion de default
    if (x_izq, x_der) != (FRANJA_X_IZQ, FRANJA_X_DER):
        nombre += f"_x{x_izq}-{x_der}"
    if max_age != 30:
        nombre += f"_age{max_age}"
    if imgsz != 640:
        nombre += f"_img{imgsz}"
    if iou != 0.7:
        nombre += f"_iou{iou}"
    if conf != 0.4:
        nombre += f"_conf{conf}"
    if modelo != "yolo11x":
        nombre += f"_{modelo}"
    if anidados is not None:
        nombre += f"_anid{anidados}"
        if anidados_y is not None:
            nombre += f"_y{anidados_y}"
    print(f"Abriendo video: {video_full_path}")
    if modelo == "yolo11x":
        # el modelo vive en la carpeta del TEC (114 MB, para no tenerlo duplicado)
        model = YOLO(os.path.join(ruta_base, "..", "CountCars_Tec", "yolo11x.pt"))
    elif modelo.startswith("rtdetr"):
        # RT-DETR (transformer): detecta mas autos medio tapados en la fila del
        # semaforo; se descarga solo la primera vez a esta carpeta
        model = RTDETR(os.path.join(ruta_base, f"{modelo}.pt"))
    else:
        model = YOLO(os.path.join(ruta_base, f"{modelo}.pt"))
    cap = cv2.VideoCapture(video_full_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_full_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # salidas con el nombre del video para no sobreescribir resultados de otros videos
    ruta_video_salida = os.path.join(ruta_base, f"resultado_{nombre}.mp4")
    ruta_csv = os.path.join(ruta_base, f"reporte_{nombre}.csv")
    video_writer = cv2.VideoWriter(ruta_video_salida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    # mismos parametros que en el TEC, salvo max_age: aqui los autos se detienen en
    # el semaforo y se tapan entre si, y con 30 frames (1 s) pierden el ID
    oc_tracker = OcSort(det_thresh=0.3, max_age=max_age, min_hits=1, iou_threshold=0.3)

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

        # mismos parametros de deteccion que en el TEC (ver comentarios en count_cars(E1));
        # imgsz e iou se pueden subir para no perder autos chicos o encimados;
        # conf se puede bajar para aceptar autos medio tapados (con poca confianza)
        results = model.predict(
            frame, classes=[2, 3, 5, 7], verbose=False,
            conf=conf, agnostic_nms=True, imgsz=imgsz, iou=iou
        )
        det_boxes = results[0].boxes.xyxy.cpu().numpy()
        det_conf = results[0].boxes.conf.cpu().numpy()
        det_cls = results[0].boxes.cls.cpu().numpy()
        if anidados is not None and len(det_boxes) > 1:
            quedan = sin_anidados(det_boxes, anidados, anidados_y)
            det_boxes, det_conf, det_cls = det_boxes[quedan], det_conf[quedan], det_cls[quedan]
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
    parser.add_argument("--max-age", type=int, default=30, help="Frames que el tracker aguanta sin ver un auto (default 30).")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen para YOLO (default 640).")
    parser.add_argument("--iou", type=float, default=0.7, help="Umbral IoU del NMS de YOLO (default 0.7).")
    parser.add_argument("--conf", type=float, default=0.4, help="Confianza minima de YOLO (default 0.4).")
    parser.add_argument("--modelo", default="yolo11x", help="Modelo de deteccion: yolo11x (default), rtdetr-x, rtdetr-l, yolo12x...")
    parser.add_argument("--anidados", type=float, default=None,
                        help="Quitar cuadros que quedan al menos esta fraccion dentro de otro mas grande (ej. 0.9). Default: no filtrar.")
    parser.add_argument("--anidados-y", type=int, default=None,
                        help="Aplicar --anidados solo a cuadros con centro arriba de esta y (ej. 335: carril der -> izq).")
    args = parser.parse_args()
    detector(args.video_path, not args.sin_ventana, *args.franja, args.max_age, args.imgsz, args.iou, args.conf,
             args.modelo, args.anidados, args.anidados_y)
