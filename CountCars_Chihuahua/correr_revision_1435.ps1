$Host.UI.RawUI.WindowTitle = "Revision 14:35"
$src = Get-ChildItem -Recurse -Filter 20261014_143527A.mp4 "C:\Users\marif\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Shortcuts\IMPLAN - Datos\Videos\14-10-2026\Teofilo Borunda y Pedro de Zulouaga" | Select-Object -First 1
New-Item -ItemType Directory -Force lote_14oct\_videos | Out-Null
Write-Host "[1/3] Bajando video de OneDrive..."
Copy-Item $src.FullName lote_14oct\_videos\20261014_143527A.mp4
Write-Host "[2/3] Contando (avanza de 10 en 10 s hasta ~300 s)..."
python count_cars_chihuahua.py lote_14oct\_videos\20261014_143527A.mp4 --sin-ventana
Write-Host "[3/3] Armando hoja de revision..."
python hoja_revision.py 20261014_143527A
Remove-Item lote_14oct\_videos\20261014_143527A.mp4
Write-Host "Listo. Ya puedes cerrar esta ventana."
