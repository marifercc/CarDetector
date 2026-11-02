import cv2 #abrir video
import numpy as np #matrices y operaciones matemáticas
from ultralytics import YOLO, RTDETR #importar modelo de yolo (o RT-DETR)
from boxmot.trackers.bbox.ocsort import OcSort #mismo tracker que count_cars(E1) del TEC
import torch #saber si hay GPU para el modelo de apariencia (ReID) de los otros trackers
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

# --- CONFIGURACION DE DETECCION ---
# Calibrada con chihuahua_5min.mp4 contra un conteo a mano (izq->der 152,
# der->izq 153, total 305). Con estos valores da 140 / 165 / 305: ~8% de error
# en cada sentido. Con YOLO11x y los parametros del TEC daba 110 / 150 / 260
# (-28% en izq->der). Todas las pruebas estan en los reporte_chihuahua_5min_*.csv.
#  - izq -> der va por los carriles cercanos y se detiene en el semaforo en 2-3
#    filas: los autos de adentro quedan medio tapados por los de afuera. RT-DETR
#    los detecta mejor que YOLO, y con conf 0.3 y max_age 90 (3 s) se recuperan mas.
#  - der -> izq va por el carril del fondo (centro y < Y_SENTIDOS) y ya contaba
#    bien; ahi RT-DETR pone cuadros duplicados sobre partes del mismo auto, asi que
#    se quitan los cuadros anidados y se pide conf 0.4.
Y_SENTIDOS = 335 # arriba: carril der -> izq (y ~300-330); abajo: izq -> der (y ~345-405)
DEFAULTS = {
    "max_age": 90,
    "imgsz": 640,
    "iou": 0.7,
    "conf": 0.3, # carriles de abajo (izq -> der)
    "conf_arriba": 0.4, # carril de arriba (der -> izq); 0 = misma conf que abajo
    "modelo": "rtdetr-x",
    "anidados": 0.9, # 0 = no quitar cuadros anidados
    "tracker": "ocsort",
}


def crear_tracker(nombre, max_age):
    # OC-SORT (default) solo usa posicion y movimiento. BoT-SORT, DeepOC-SORT y
    # StrongSORT ademas comparan la apariencia del auto (modelo ReID), para no
    # cambiarle el ID cuando queda tapado un rato en la fila del semaforo.
    # Los umbrales van en 0.3 (igual que conf) porque los autos tapados salen con
    # poca confianza y con los defaults de boxmot (0.45-0.6) no abririan track.
    comunes = dict(det_thresh=0.3, max_age=max_age, min_hits=1, iou_threshold=0.3)
    if nombre == "ocsort":
        return OcSort(**comunes)
    if nombre == "bytetrack":
        from boxmot.trackers.bbox.bytetrack import ByteTrack
        return ByteTrack(track_thresh=0.3, track_buffer=max_age, **comunes)
    # el modelo ReID (osnet, ~1 MB) se descarga solo la primera vez
    from boxmot.reid.core.reid import ReID
    reid = ReID(weights="osnet_x0_25_msmt17.pt", device="cuda:0" if torch.cuda.is_available() else "cpu").model
    if nombre == "botsort":
        from boxmot.trackers.bbox.botsort import BotSort
        return BotSort(reid_model=reid, track_high_thresh=0.3, new_track_thresh=0.3,
                       track_buffer=max_age, **comunes)
    if nombre == "deepocsort":
        from boxmot.trackers.bbox.deepocsort import DeepOcSort
        return DeepOcSort(reid_model=reid, **comunes)
    if nombre == "strongsort":
        from boxmot.trackers.bbox.strongsort import StrongSort
        return StrongSort(reid_model=reid, n_init=1, **comunes)
    raise ValueError(f"Tracker desconocido: {nombre}")


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


def detector(video_path, mostrar, x_izq=FRANJA_X_IZQ, x_der=FRANJA_X_DER, max_age=DEFAULTS["max_age"],
             imgsz=DEFAULTS["imgsz"], iou=DEFAULTS["iou"], conf=DEFAULTS["conf"], modelo=DEFAULTS["modelo"],
             anidados=DEFAULTS["anidados"], conf_arriba=DEFAULTS["conf_arriba"], tracker=DEFAULTS["tracker"]):
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    video_full_path = video_path if os.path.isabs(video_path) else os.path.join(ruta_base, video_path)
    nombre = os.path.splitext(os.path.basename(video_full_path))[0]
    # si se prueba otra configuracion, las salidas llevan lo que cambio en el nombre
    # para no sobreescribir los resultados de la configuracion de default
    if (x_izq, x_der) != (FRANJA_X_IZQ, FRANJA_X_DER):
        nombre += f"_x{x_izq}-{x_der}"
    usados = {"max_age": max_age, "imgsz": imgsz, "iou": iou, "conf": conf,
              "conf_arriba": conf_arriba, "modelo": modelo, "anidados": anidados, "tracker": tracker}
    for clave, valor in usados.items():
        if valor != DEFAULTS[clave]:
            nombre += f"_{clave.replace('_', '')}{valor}"
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
    oc_tracker = crear_tracker(tracker, max_age)

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

        # se predice con conf (la mas baja) y abajo se le pide conf_arriba al carril de
        # arriba; ver CONFIGURACION DE DETECCION para por que cada valor
        results = model.predict(
            frame, classes=[2, 3, 5, 7], verbose=False,
            conf=conf, agnostic_nms=True, imgsz=imgsz, iou=iou
        )
        det_boxes = results[0].boxes.xyxy.cpu().numpy()
        det_conf = results[0].boxes.conf.cpu().numpy()
        det_cls = results[0].boxes.cls.cpu().numpy()
        if conf_arriba > conf:
            # confianza distinta por carril: abajo (izq -> der, autos tapados) basta con
            # conf; arriba de Y_SENTIDOS (der -> izq) se pide conf_arriba para no
            # meter detecciones dudosas en el carril que ya cuenta bien
            arriba = (det_boxes[:, 1] + det_boxes[:, 3]) / 2 < Y_SENTIDOS
            quedan = ~arriba | (det_conf >= conf_arriba)
            det_boxes, det_conf, det_cls = det_boxes[quedan], det_conf[quedan], det_cls[quedan]
        if anidados > 0 and len(det_boxes) > 1:
            quedan = sin_anidados(det_boxes, anidados, Y_SENTIDOS)
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
    # los defaults son la configuracion calibrada (ver DEFAULTS arriba); las opciones
    # sirven para probar otras sin editar el codigo
    parser.add_argument("--max-age", type=int, default=DEFAULTS["max_age"],
                        help=f"Frames que el tracker aguanta sin ver un auto (default {DEFAULTS['max_age']}).")
    parser.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"],
                        help=f"Tamaño de imagen para el detector (default {DEFAULTS['imgsz']}).")
    parser.add_argument("--iou", type=float, default=DEFAULTS["iou"],
                        help=f"Umbral IoU del NMS (default {DEFAULTS['iou']}).")
    parser.add_argument("--conf", type=float, default=DEFAULTS["conf"],
                        help=f"Confianza minima en los carriles de abajo, izq -> der (default {DEFAULTS['conf']}).")
    parser.add_argument("--conf-arriba", type=float, default=DEFAULTS["conf_arriba"],
                        help=f"Confianza minima en el carril de arriba, der -> izq (default {DEFAULTS['conf_arriba']}; 0 = igual que --conf).")
    parser.add_argument("--modelo", default=DEFAULTS["modelo"],
                        help="Modelo de deteccion: rtdetr-x (default), rtdetr-l, yolo11x, yolo12x...")
    parser.add_argument("--anidados", type=float, default=DEFAULTS["anidados"],
                        help=f"Quitar en el carril de arriba cuadros que quedan al menos esta fraccion dentro de otro (default {DEFAULTS['anidados']}; 0 = no filtrar).")
    parser.add_argument("--tracker", default=DEFAULTS["tracker"],
                        choices=["ocsort", "bytetrack", "botsort", "deepocsort", "strongsort"],
                        help=f"Tracker de boxmot (default {DEFAULTS['tracker']}).")
    args = parser.parse_args()
    detector(args.video_path, not args.sin_ventana, *args.franja, args.max_age, args.imgsz, args.iou, args.conf,
             args.modelo, args.anidados, args.conf_arriba, args.tracker)
