# guarda las detecciones de los 16 videos del lote (para replay_conteo.py)
M=${MODELO:-rtdetr-chihuahua}
for v in lote_14oct_f1/_videos/*.mp4; do
  n=$(basename "$v" .mp4)
  [ -f "detecciones/det_${n}_${M}_img640.npz" ] && continue
  echo "== $n $(date +%H:%M)"
  python -u cache_detecciones.py "$v" --modelo $M --conf 0.05 | tail -1
done
echo LISTO
