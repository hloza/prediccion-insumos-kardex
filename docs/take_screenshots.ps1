# Takes the report screenshots with headless Edge. The server must be running on port 8000.
$browser = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
$out = Join-Path $PSScriptRoot 'capturas'
New-Item -ItemType Directory -Force $out | Out-Null
$cases = @{
  '01_caso_valido_leche.png'   = 'insumo=Leche%20entera&hasta=2026-09-26&stock=20000&horizonte=7&entrega=2&run=1'
  '02_caso_dificil_harina.png' = 'insumo=Harina&hasta=2026-09-26&stock=0&horizonte=14&entrega=3&run=1'
  '03_entrada_invalida.png'    = 'insumo=Leche%20entera&historial=2026-09-01,4200|2026-09-02,3900|2026-09-04,4100&stock=500&run=1'
}
foreach ($name in $cases.Keys) {
  $url = "http://127.0.0.1:8000/?" + $cases[$name]
  & $browser --headless=new --disable-gpu --hide-scrollbars --window-size=1280,1000 `
    --virtual-time-budget=15000 --screenshot="$out\$name" $url 2>$null
  Start-Sleep -Seconds 4
}
Get-ChildItem $out
