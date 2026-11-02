import cv2 #abrir video
import csv #leer el reporte
import os #rutas de archivos
import html #escapar texto en la pagina
import argparse #leer argumentos de la terminal

# Hoja para revisar a mano cada vehiculo contado: toma el reporte_*.csv y el
# resultado_*.mp4 de count_cars_chihuahua.py (que ya trae dibujado "#N" sobre cada
# auto contado), recorta la zona de la franja en el segundo en que se conto, y
# arma una pagina (revision_<nombre>/index.html) donde se marca cada recorte como
# bien contado, doble (mismo auto con otro ID), no es vehiculo u otro sentido.
# Las marcas se descargan como CSV desde la misma pagina.

# zona del recorte (coordenadas del video de 1280x720): la franja esta en x 800-860
# y los autos recien contados suelen quedar justo a un lado. Si el reporte trae la
# caja del auto, el recorte se centra en ella (un auto que el tracker perdio detras
# de la columna puede contarse lejos de la franja) y la caja se marca en rojo.
RECORTE_X = (520, 1140)
RECORTE_Y = (230, 450)
ESCALA = 1.5
# der -> izq va por un solo carril: dos conteos ahi casi al mismo tiempo no pueden ser
# dos autos seguidos, asi que probablemente es el mismo auto con dos IDs. En izq -> der
# hay 2-3 carriles y es normal que crucen juntos, ahi no se marca.
SEG_DOBLE = 0.3
FRAMES_DESPUES = 3 # la etiqueta "#N" aparece desde el frame en que se conto


def leer_reporte(ruta_csv):
    contados = []
    with open(ruta_csv, newline="") as f:
        for fila in csv.reader(f):
            if not fila:
                break # despues de la lista viene el RESUMEN
            if fila[0] == "Numero":
                continue
            c = {"numero": int(fila[0]), "id": int(fila[1]), "segundo": float(fila[2]),
                 "sentido": fila[3], "clase": fila[4], "caja": None}
            if len(fila) >= 9: # reportes viejos no traen la caja
                c["caja"] = tuple(int(v) for v in fila[5:9])
            contados.append(c)
    return contados


def marcar_posibles_dobles(contados):
    anterior = None
    for c in contados:
        c["posible_doble"] = False
        if c["sentido"] != "der -> izq":
            continue
        if anterior is not None and c["segundo"] - anterior["segundo"] <= SEG_DOBLE:
            c["posible_doble"] = anterior["posible_doble"] = True
        anterior = c


def guardar_recortes(ruta_video, contados, carpeta):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {ruta_video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    # el reporte guarda frame / fps redondeado a 0.1 s; se toman unos frames despues
    # para que la etiqueta ya este dibujada
    por_frame = {}
    for c in contados:
        por_frame.setdefault(round(c["segundo"] * fps) + FRAMES_DESPUES, []).append(c)
    ultimo_frame = max(por_frame)
    frame_count = 0
    while frame_count < ultimo_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        for c in por_frame.get(frame_count, []):
            x0, x1 = RECORTE_X
            imagen = frame
            if c["caja"] is not None:
                bx1, by1, bx2, by2 = c["caja"]
                ancho = x1 - x0
                x0 = min(max((bx1 + bx2) // 2 - ancho // 2, 0), frame.shape[1] - ancho)
                x1 = x0 + ancho
                imagen = frame.copy()
                cv2.rectangle(imagen, (bx1 - 3, by1 - 3), (bx2 + 3, by2 + 3), (0, 0, 255), 2)
            recorte = imagen[RECORTE_Y[0]:RECORTE_Y[1], x0:x1]
            recorte = cv2.resize(recorte, None, fx=ESCALA, fy=ESCALA)
            c["imagen"] = f"{c['numero']:03d}.jpg"
            cv2.imwrite(os.path.join(carpeta, c["imagen"]), recorte, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cap.release()


def escribir_pagina(contados, carpeta, nombre):
    tarjetas = []
    for c in contados:
        if "imagen" not in c:
            continue
        aviso = '<span class="aviso">posible doble</span>' if c["posible_doble"] else ""
        sentido = "der-izq" if c["sentido"] == "der -> izq" else "izq-der"
        opciones = "".join(
            f'<label><input type="radio" name="m{c["numero"]}" value="{v}"{" checked" if v == "bien" else ""}> {t}</label>'
            for v, t in [("bien", "Bien"), ("doble", "Doble"), ("no_vehiculo", "No es vehículo"), ("otro_sentido", "Otro sentido")])
        tarjetas.append(f'''
<div class="tarjeta" data-sentido="{sentido}" data-numero="{c["numero"]}" data-id="{c["id"]}" data-segundo="{c["segundo"]}">
  <img src="{c["imagen"]}" loading="lazy" alt="#{c["numero"]}">
  <div class="datos"><b>#{c["numero"]}</b> · {c["segundo"]} s · {html.escape(c["sentido"])} · {html.escape(c["clase"])} · id {c["id"]} {aviso}</div>
  <div class="opciones">{opciones}</div>
</div>''')
    pagina = f'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Revisión {html.escape(nombre)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #f4f4f4; color: #222; }}
header {{ position: sticky; top: 0; background: #f4f4f4; padding: 8px 0; z-index: 1; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 12px; }}
.tarjeta {{ background: #fff; border: 2px solid #ddd; border-radius: 6px; padding: 6px; }}
.tarjeta.mal {{ border-color: #d33; background: #fff0f0; }}
.tarjeta img {{ width: 100%; display: block; }}
.datos {{ margin: 4px 0; font-size: 14px; }}
.opciones {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 14px; }}
.aviso {{ background: #f5c542; padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
.oculta {{ display: none; }}
</style></head><body>
<header>
  <b>{html.escape(nombre)}</b> —
  Mostrar: <select id="filtro"><option value="der-izq">der → izq</option><option value="izq-der">izq → der</option><option value="todos">todos</option></select>
  <label><input type="checkbox" id="solo_dobles"> solo posibles dobles</label>
  · <span id="cuenta"></span>
  <button id="descargar">Descargar marcas (CSV)</button>
  <div style="font-size:13px">Busca la etiqueta verde <b>#N</b> en cada recorte. Doble = ese auto ya se contó con otro número.</div>
</header>
<div class="grid">{"".join(tarjetas)}</div>
<script>
const tarjetas = [...document.querySelectorAll('.tarjeta')];
function marca(t) {{ return t.querySelector('input:checked').value; }}
function actualizar() {{
  const f = document.getElementById('filtro').value, dobles = document.getElementById('solo_dobles').checked;
  let n = 0, mal = 0;
  for (const t of tarjetas) {{
    const ver = (f === 'todos' || t.dataset.sentido === f) && (!dobles || t.querySelector('.aviso'));
    t.classList.toggle('oculta', !ver);
    t.classList.toggle('mal', marca(t) !== 'bien');
    if (f === 'todos' || t.dataset.sentido === f) {{ n++; if (marca(t) !== 'bien') mal++; }}
  }}
  document.getElementById('cuenta').textContent = `${{n}} contados, ${{mal}} marcados mal, ${{n - mal}} bien`;
}}
document.addEventListener('change', actualizar);
document.getElementById('descargar').onclick = () => {{
  const filas = ['Numero,ID tracker,Segundo,Sentido,Marca'];
  for (const t of tarjetas) filas.push([t.dataset.numero, t.dataset.id, t.dataset.segundo, t.dataset.sentido, marca(t)].join(','));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([filas.join('\\n')], {{type: 'text/csv'}}));
  a.download = 'marcas_{html.escape(nombre)}.csv';
  a.click();
}};
actualizar();
</script></body></html>'''
    with open(os.path.join(carpeta, "index.html"), "w", encoding="utf-8") as f:
        f.write(pagina)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hoja para revisar a mano los vehiculos contados.")
    parser.add_argument("nombre", nargs="?", default="chihuahua_5min",
                        help="Nombre de la corrida: usa reporte_<nombre>.csv y resultado_<nombre>.mp4 (default chihuahua_5min).")
    args = parser.parse_args()
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    contados = leer_reporte(os.path.join(ruta_base, f"reporte_{args.nombre}.csv"))
    marcar_posibles_dobles(contados)
    carpeta = os.path.join(ruta_base, f"revision_{args.nombre}")
    os.makedirs(carpeta, exist_ok=True)
    guardar_recortes(os.path.join(ruta_base, f"resultado_{args.nombre}.mp4"), contados, carpeta)
    escribir_pagina(contados, carpeta, args.nombre)
    print(f"{len(contados)} vehiculos, {sum(c['posible_doble'] for c in contados)} posibles dobles")
    print(f"Abre: {os.path.join(carpeta, 'index.html')}")
