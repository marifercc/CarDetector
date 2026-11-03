import json #leer etiquetas
import os #rutas de archivos
import shutil #copiar imagenes y el modelo final
import argparse #leer argumentos de la terminal
from ultralytics import RTDETR #modelo a reentrenar

# Reentrena RT-DETR-x con frames de esta camara (etiquetado/frames + un
# etiquetas*.json de etiquetado/index.html o de etiquetas_auto.py), con una sola
# clase "vehiculo". El modelo queda en rtdetr-chihuahua.pt y se usa con
# count_cars_chihuahua.py --modelo rtdetr-chihuahua.
# Validacion: el ultimo 15% (en tiempo) de los frames de cada video. No se mezclan
# al azar porque frames a 2 s de distancia son casi iguales y la validacion saldria
# demasiado optimista.

W, H = 1280, 720
PARTE_VAL = 0.15


def armar_dataset(etiquetas, carpeta_frames, carpeta_dataset):
    por_video = {}
    for imagen in sorted(etiquetas["imagenes"]):
        por_video.setdefault(imagen.rsplit("_", 1)[0], []).append(imagen)
    for parte in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(carpeta_dataset, sub, parte), exist_ok=True)
    cuantas = {"train": 0, "val": 0}
    for imagenes in por_video.values():
        corte = int(len(imagenes) * (1 - PARTE_VAL))
        for k, imagen in enumerate(imagenes):
            parte = "train" if k < corte else "val"
            shutil.copy(os.path.join(carpeta_frames, imagen), os.path.join(carpeta_dataset, "images", parte, imagen))
            # formato YOLO: clase x_centro y_centro ancho alto, normalizados
            with open(os.path.join(carpeta_dataset, "labels", parte, imagen[:-4] + ".txt"), "w") as f:
                for x1, y1, x2, y2 in etiquetas["imagenes"][imagen]:
                    f.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
            cuantas[parte] += 1
    ruta_yaml = os.path.join(carpeta_dataset, "data.yaml")
    with open(ruta_yaml, "w") as f:
        f.write(f"path: {carpeta_dataset}\ntrain: images/train\nval: images/val\nnames:\n  0: vehiculo\n")
    print(f"dataset: {cuantas['train']} train, {cuantas['val']} val")
    return ruta_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reentrena RT-DETR con frames de esta camara.")
    parser.add_argument("etiquetas", nargs="?", default="etiquetado/etiquetas.json",
                        help="JSON de etiquetas (default etiquetado/etiquetas.json).")
    parser.add_argument("--epocas", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_etiquetas = args.etiquetas if os.path.isabs(args.etiquetas) else os.path.join(ruta_base, args.etiquetas)
    with open(ruta_etiquetas, encoding="utf-8") as f:
        etiquetas = json.load(f)

    carpeta_dataset = os.path.join(ruta_base, "dataset")
    shutil.rmtree(carpeta_dataset, ignore_errors=True)
    ruta_yaml = armar_dataset(etiquetas, os.path.join(ruta_base, "etiquetado", "frames"), carpeta_dataset)

    model = RTDETR(os.path.join(ruta_base, "rtdetr-x.pt"))
    # workers=0: en Windows los procesos extra del dataloader dan problemas
    model.train(data=ruta_yaml, epochs=args.epocas, imgsz=640, batch=args.batch, device=0, workers=0,
                patience=20, project=os.path.join(ruta_base, "entrenamiento"), name="rtdetr-chihuahua",
                exist_ok=True)
    mejor = os.path.join(ruta_base, "entrenamiento", "rtdetr-chihuahua", "weights", "best.pt")
    shutil.copy(mejor, os.path.join(ruta_base, "rtdetr-chihuahua.pt"))
    print(f"Modelo guardado en: {os.path.join(ruta_base, 'rtdetr-chihuahua.pt')}")
