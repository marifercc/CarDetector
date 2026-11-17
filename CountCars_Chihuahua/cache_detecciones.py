import os #rutas
import argparse #leer argumentos de la terminal
import cv2 #abrir video
import numpy as np #guardar las detecciones
from ultralytics import RTDETR #modelo de deteccion
from count_cars_chihuahua import W, H #mismo tamaño de cuadro que el contador
from preparar_etiquetado import tapar_fuera_de_calle #mismo recorte con que se entreno el modelo propio

# Guarda las detecciones crudas de cada cuadro (antes del NMS y de los filtros por
# carril) para probar trackers y reglas de conteo sin volver a pasar el video por el
# detector, que es lo lento. Las usa replay_conteo.py.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guarda las detecciones de cada cuadro en un .npz.")
    parser.add_argument("video_path", nargs="?", default="chihuahua_5min.mp4")
    parser.add_argument("--modelo", default="rtdetr-chihuahua")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05, help="Confianza minima que se guarda.")
    args = parser.parse_args()

    ruta_base = os.path.dirname(os.path.abspath(__file__))
    video = args.video_path if os.path.isabs(args.video_path) else os.path.join(ruta_base, args.video_path)
    model = RTDETR(os.path.join(ruta_base, f"{args.modelo}.pt"))
    propio = "vehiculo" in model.names.values()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cajas, cuadro = [], []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        frame = cv2.resize(frame, (W, H))
        r = model.predict(tapar_fuera_de_calle(frame) if propio else frame, classes=None if propio else [2, 3, 5, 7],
                          verbose=False, conf=args.conf, imgsz=args.imgsz)[0].boxes
        d = np.concatenate([r.xyxy.cpu().numpy(), r.conf.cpu().numpy()[:, None], r.cls.cpu().numpy()[:, None]], axis=1)
        cajas.append(d.astype(np.float32))
        cuadro.append(np.full(len(d), n, dtype=np.int32))
        if n % 1000 == 0:
            print(n, "cuadros", flush=True)
    nombre = os.path.splitext(os.path.basename(video))[0]
    salida = os.path.join(ruta_base, "detecciones", f"det_{nombre}_{args.modelo}_img{args.imgsz}.npz")
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    np.savez_compressed(salida, cajas=np.concatenate(cajas), cuadro=np.concatenate(cuadro), fps=fps, n=n)
    print("guardado", salida)
