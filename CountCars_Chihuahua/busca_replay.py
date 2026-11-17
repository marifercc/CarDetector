import sys, json, itertools, os #argumentos y resultados
from multiprocessing import Pool #varias configuraciones a la vez
from replay_conteo import cargar, replay
from evalua_replay import evalua

# Prueba una rejilla de parametros del tracker/conteo sobre detecciones guardadas
# y escribe el F1 aproximado de cada una (evalua_replay.py) en busqueda_<det>.jsonl.
#   python busca_replay.py detecciones/det_....npz '{"iou_threshold":[0.2,0.3],"delta_t":[1,3]}'

FR = FPS = None


def init(ruta):
    global FR, FPS
    FR, FPS = cargar(ruta)


def una(kw):
    try:
        return kw, evalua(replay(FR, FPS, **kw))
    except Exception as e:
        return kw, {"error": repr(e)}


if __name__ == "__main__":
    ruta = sys.argv[1]
    rejilla = json.load(open(sys.argv[2][1:])) if sys.argv[2].startswith("@") else json.loads(sys.argv[2])
    claves = list(rejilla) if isinstance(rejilla, dict) else []
    combos = rejilla if isinstance(rejilla, list) else [dict(zip(claves, v)) for v in itertools.product(*rejilla.values())]
    salida = os.path.join(os.path.dirname(ruta), "busqueda_" + os.path.basename(ruta).replace(".npz", ".jsonl"))
    with Pool(int(os.environ.get("PROCS", 6)), initializer=init, initargs=(ruta,)) as pool, open(salida, "a") as f:
        for kw, res in pool.imap_unordered(una, combos):
            f.write(json.dumps({"kw": kw, **res}) + "\n"); f.flush()
            print(res.get("f1_piso"), res.get("f1_techo"), res.get("n"), res.get("izq"), res.get("der"), kw, flush=True)
