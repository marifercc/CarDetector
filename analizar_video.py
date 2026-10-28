import cv2 #abrir video
import numpy as np #matrices y operaciones matemáticas
from ultralytics import YOLO #importar modelo de yolo
from boxmot.trackers.bbox.ocsort import OcSort #mismo tracker que count_cars(E1)
import argparse #leer argumentos de la terminal
import os #abrir rutas de archivos

# Script de ANALISIS (no cuenta nada): sirve para revisar un video nuevo antes de
# definir las zonas de conteo. Genera una carpeta analisis_<video>/ con imagenes
# que se pueden revisar (o pasarle a Claude) para decidir donde poner las zonas.
# Usa la misma deteccion y tracker que count_cars(E1) para que lo que se vea aqui
# sea lo mismo que va a "ver" el contador.

W, H = 1280, 720 # mismo tamaño que count_cars(E1): las coordenadas salen en este espacio
CLASES = [2, 3, 5, 7] # car, motorcycle, bus, truck
COLORES = {2: (0, 255, 0), 3: (0, 0, 255), 5: (255, 0, 255), 7: (255, 255, 0)}


def dibujar_cuadricula(img):
    # lineas cada 50 px, numeradas cada 100 px, para poder leer coordenadas a ojo
    for x in range(0, W, 50):
        grosor = 2 if x % 100 == 0 else 1
        cv2.line(img, (x, 0), (x, H), (255, 255, 255), grosor)
        if x % 100 == 0:
            cv2.putText(img, str(x), (x + 2, 15), 0, 0.45, (0, 255, 255), 1)
    for y in range(0, H, 50):
        grosor = 2 if y % 100 == 0 else 1
        cv2.line(img, (0, y), (W, y), (255, 255, 255), grosor)
        if y % 100 == 0:
            cv2.putText(img, str(y), (2, y - 3), 0, 0.45, (0, 255, 255), 1)
    return img


def analizar(video_path, cada_seg, max_seg):
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    video_full_path = video_path if os.path.isabs(video_path) else os.path.join(ruta_base, video_path)
    nombre = os.path.splitext(os.path.basename(video_full_path))[0]
    carpeta = os.path.join(ruta_base, f"analisis_{nombre}")
    os.makedirs(carpeta, exist_ok=True)

    cap = cv2.VideoCapture(video_full_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_full_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limite = min(total_frames, int(max_seg * fps))
    print(f"Video: {w_orig}x{h_orig}, {fps:.1f} fps, {total_frames / fps:.1f} s -> analizando {limite / fps:.1f} s")

    # el modelo vive en la carpeta del TEC (114 MB, para no tenerlo duplicado)
    model = YOLO(os.path.join(ruta_base, "..", "CountCars_Tec", "yolo11x.pt"))
    oc_tracker = OcSort(det_thresh=0.3, max_age=30, min_hits=1, iou_threshold=0.3)

    recorridos = {} # id -> lista de (frame, xc, yc, cls)
    det_bajas = [] # detecciones entre 0.25 y 0.4 (las que count_cars tira); dice si la reja baja la confianza
    fondo = None
    frames_muestra = []
    frame_count = 0

    while frame_count < limite:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (W, H))
        if fondo is None:
            fondo = frame.copy()

        # conf=0.25 para ver tambien lo que count_cars(E1) descarta con conf=0.4
        results = model.predict(
            frame, classes=CLASES, verbose=False,
            conf=0.25, agnostic_nms=True, imgsz=640
        )
        det_boxes = results[0].boxes.xyxy.cpu().numpy()
        det_conf = results[0].boxes.conf.cpu().numpy()
        det_cls = results[0].boxes.cls.cpu().numpy()

        altas = det_conf >= 0.4
        for box in det_boxes[~altas]:
            det_bajas.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))

        if altas.any():
            dets = np.concatenate([det_boxes[altas], det_conf[altas].reshape(-1, 1), det_cls[altas].reshape(-1, 1)], axis=1)
        else:
            dets = np.empty((0, 6))
        tracks = oc_tracker.update(dets, frame)

        if len(tracks) > 0:
            for box, id_obj, cls in zip(tracks.xyxy, tracks.id, tracks.cls):
                xc, yc = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                recorridos.setdefault(int(id_obj), []).append((frame_count, xc, yc, int(cls)))

        if frame_count % int(cada_seg * fps) == 0:
            muestra = frame.copy()
            for box, conf, cls in zip(det_boxes, det_conf, det_cls):
                color = COLORES.get(int(cls), (255, 255, 255)) if conf >= 0.4 else (128, 128, 128)
                cv2.rectangle(muestra, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(muestra, f"{conf:.2f}", (int(box[0]), int(box[1]) - 4), 0, 0.5, color, 1)
            seg = frame_count / fps
            cv2.putText(muestra, f"t={seg:.0f}s", (W - 120, H - 15), 0, 0.8, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(carpeta, f"frame_{int(seg):03d}s.jpg"), dibujar_cuadricula(muestra))
            frames_muestra.append(muestra)

        frame_count += 1
        if frame_count % int(fps * 5) == 0:
            print(f"  {frame_count / fps:.0f} s procesados, {len(recorridos)} IDs hasta ahora")

    cap.release()

    # --- recorridos: por donde pasan los vehiculos y donde el tracker los pierde ---
    # verde/rojo/morado/cyan = clase, circulo blanco = donde empieza el ID,
    # X roja = donde termina. Si hay muchas X y circulos juntos a media calle
    # (ej. junto al tronco) es que ahi el tracker pierde al vehiculo y le da ID nuevo.
    img_rec = dibujar_cuadricula((fondo * 0.6).astype(np.uint8))
    cortos = 0
    for id_obj, pts in recorridos.items():
        if len(pts) < 5: # IDs que duran menos de 5 frames: ruido, se cuentan aparte
            cortos += 1
            continue
        cls = max(set(p[3] for p in pts), key=[p[3] for p in pts].count)
        color = COLORES.get(cls, (255, 255, 255))
        linea = np.array([(p[1], p[2]) for p in pts], np.int32)
        cv2.polylines(img_rec, [linea], False, color, 2)
        cv2.circle(img_rec, tuple(linea[0]), 6, (255, 255, 255), -1)
        cv2.drawMarker(img_rec, tuple(linea[-1]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(img_rec, str(id_obj), tuple(linea[0] + 6), 0, 0.4, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(carpeta, "recorridos.jpg"), img_rec)

    # --- detecciones descartadas por confianza baja (0.25-0.4) ---
    img_bajas = dibujar_cuadricula((fondo * 0.6).astype(np.uint8))
    for xc, yc in det_bajas:
        cv2.circle(img_bajas, (xc, yc), 3, (0, 165, 255), -1)
    cv2.imwrite(os.path.join(carpeta, "detecciones_baja_confianza.jpg"), img_bajas)

    # --- mosaico de todos los frames muestra, para ver el minuto de un vistazo ---
    if frames_muestra:
        chicos = [cv2.resize(f, (W // 4, H // 4)) for f in frames_muestra]
        while len(chicos) % 4:
            chicos.append(np.zeros_like(chicos[0]))
        filas = [np.hstack(chicos[i:i + 4]) for i in range(0, len(chicos), 4)]
        cv2.imwrite(os.path.join(carpeta, "mosaico.jpg"), np.vstack(filas))

    largos = {k: v for k, v in recorridos.items() if len(v) >= 5}
    nombres = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    por_clase = {}
    for pts in largos.values():
        cls = max(set(p[3] for p in pts), key=[p[3] for p in pts].count)
        por_clase[nombres.get(cls, cls)] = por_clase.get(nombres.get(cls, cls), 0) + 1
    with open(os.path.join(carpeta, "resumen.txt"), "w", encoding="utf-8") as f:
        f.write(f"Video: {video_full_path}\n")
        f.write(f"Original: {w_orig}x{h_orig}, {fps:.1f} fps, {total_frames / fps:.1f} s\n")
        f.write(f"Analizado: {frame_count / fps:.1f} s (redimensionado a {W}x{H})\n")
        f.write(f"IDs con >=5 frames: {len(largos)} {por_clase}\n")
        f.write(f"IDs con <5 frames (ruido): {cortos}\n")
        f.write(f"Detecciones con conf 0.25-0.4 (descartadas por count_cars): {len(det_bajas)}\n")
        f.write("\nID, clase, frame_inicio, frame_fin, (x,y) inicio, (x,y) fin\n")
        for id_obj, pts in sorted(largos.items()):
            cls = max(set(p[3] for p in pts), key=[p[3] for p in pts].count)
            f.write(f"{id_obj}, {nombres.get(cls, cls)}, {pts[0][0]}, {pts[-1][0]}, "
                    f"({pts[0][1]},{pts[0][2]}), ({pts[-1][1]},{pts[-1][2]})\n")

    print(f"Listo. Resultados en: {carpeta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analiza un video nuevo para decidir donde poner las zonas de conteo.")
    parser.add_argument("video_path", help="Ruta del video a analizar.")
    parser.add_argument("--cada", type=float, default=5, help="Cada cuantos segundos guardar un frame con cuadricula (default 5).")
    parser.add_argument("--max", type=float, default=60, help="Segundos maximos a analizar (default 60).")
    args = parser.parse_args()
    analizar(args.video_path, args.cada, args.max)
