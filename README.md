# Predicción del consumo diario de insumos

Proyecto Capstone · Maestría · Hugo Loza Guzmán

El sistema predice cuánto se va a consumir de cada insumo de una cafetería en los próximos días, a partir del kardex. Con ese pronóstico calcula cuántos días alcanza el stock y cuánto conviene reponer, para no quedarse sin insumo ni comprar de más y que se venza.

| Elemento | Definición |
|---|---|
| Unidad de observación | Un insumo en una fecha |
| Variable objetivo | Cantidad consumida en el día (salidas del kardex por producción y venta, menos devoluciones) |
| Línea base | Lo mismo que el mismo día de la semana anterior (naive estacional, lag 7) |
| Modelo | LightGBM global para los 14 insumos con consumo pasado, stock disponible y calendario |
| Métrica principal | MASE (menor a 1 indica que el modelo supera al naive estacional) |
| Validación | División por fecha: entrenamiento, validación y prueba |

## Resultado principal (conjunto de prueba, ene–oct 2026)

| Métrica (promedio de 14 insumos) | LightGBM | Línea base |
|---|---|---|
| MASE | **0,934** | 1,323 |
| MASE mediana | **0,796** | 1,153 |
| WAPE diario | **44,5 %** | 63,6 % |
| WAPE semanal | **18,9 %** | 20,9 %|
| Sesgo | −9,6 % | −1,0 % |

El modelo tiene menor MASE que la línea base en los 14 insumos. El detalle está en `results/metrics_by_item.csv` y el análisis en el informe (`docs/informe_tecnico.pdf`).

## Estructura

```
data/       extract_from_dump.py, build_dataset.py, daily_consumption.csv, catalog.csv
model/      features.py, train.py, artifacts/model.txt, artifacts/metadata.json
backend/    app.py (FastAPI), predictor.py
frontend/   index.html (servido por el backend)
tests/      test_api.py
results/    métricas, predicciones de prueba y gráficos
docs/       informe técnico, presentación, capturas
```

## Instalación

Requiere Python 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

Los datos son reales: kardex de una cafetería de Tarija entre el 1-oct-2021 y el 3-oct-2026, tomados de un respaldo PostgreSQL (`backup.sql`) del sistema de gestión del negocio. El respaldo **no** se sube al repositorio porque contiene clientes, usuarios y ventas. Solo se comparte el dataset agregado `data/daily_consumption.csv`, que no tiene datos personales.

Si se tiene el respaldo, el dataset se regenera así:

```bash
python data/extract_from_dump.py ruta/a/backup.sql   # solo lee tablas de productos, lotes, existencias y movimientos
python data/build_dataset.py                          # genera data/daily_consumption.csv
```

`daily_consumption.csv`: 25 606 filas (14 insumos × 1 829 días). Columnas: `date`, `item`, `unit`, `consumption`, `opened`, `waste`, `stock_close`.

## Entrenamiento y evaluación

```bash
python model/train.py
```

Entrena con tres objetivos de LightGBM (L2, L1, Tweedie), elige el de menor MASE en validación, y recién al final evalúa una sola vez en prueba. Genera `model/artifacts/` y `results/`. Semilla 42.

## Ejecución del sistema

```bash
uvicorn backend.app:app --port 8000
```

Abrir http://127.0.0.1:8000. En la interfaz:

1. Elegir un insumo y presionar **Cargar historial del kardex** (o subir un CSV `fecha,consumo`).
2. Ajustar el stock actual, el horizonte (1–14 días) y los días de entrega del proveedor.
3. Presionar **Predecir y calcular reposición**.

API:

| Método | Ruta | Uso |
|---|---|---|
| GET | `/api/salud` | Estado del modelo |
| GET | `/api/insumos` | Insumos, unidad y MASE en prueba |
| GET | `/api/historial?insumo=Harina&dias=56&hasta=2026-09-26` | Historial de ejemplo desde el dataset |
| POST | `/api/predecir` | Pronóstico y reposición |

Ejemplo de cuerpo para `/api/predecir`:

```json
{"insumo": "Leche entera", "stock_actual": 20000, "horizonte": 7, "dias_entrega": 2,
 "historial": [{"fecha": "2026-08-30", "consumo": 4200}, "... al menos 28 días consecutivos ..."]}
```

## Pruebas

```bash
python -m pytest -q tests
```

Incluye un caso válido (leche), un caso difícil (harina, consumo intermitente y stock en cero), entradas inválidas (insumo desconocido, historial corto, cantidades negativas, horizonte fuera de rango, días faltantes) y el flujo completo página → insumos → historial → predicción. Las capturas se regeneran con `docs/take_screenshots.ps1` con el servidor encendido.

## Limitaciones

- El modelo se entrenó para predecir el día siguiente. Para 2 a 14 días el pronóstico es recursivo y el error aumenta con el horizonte.
- El stock usado es el saldo de envases abiertos según el kardex, no el inventario de almacén.
- No hay variables de promociones, eventos ni clima.
- Insumos de producción por lotes (harina, huevo) tienen la mitad de los días en cero; el acierto diario es bajo y conviene mirar el total semanal.
- El modelo subestima en promedio un 9,6 %; el stock de seguridad lo compensa en parte.
