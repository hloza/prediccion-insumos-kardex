"""Build the technical report (docs/informe_tecnico.pdf) from the saved results."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SHOTS = ROOT / "docs" / "capturas"

m = json.loads((RES / "metrics.json").read_text(encoding="utf-8"))
by_item = pd.read_csv(RES / "metrics_by_item.csv")

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=10.2,
                      leading=14.2, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14,
                    spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1d2731"))
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
                    spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1f6f8b"))
cap = ParagraphStyle("c", parent=body, fontSize=8.8, leading=11, textColor=colors.HexColor("#5f6b76"),
                     alignment=1, spaceAfter=10)
title = ParagraphStyle("t", parent=h1, fontSize=18, leading=22, alignment=1, spaceAfter=4)
sub = ParagraphStyle("s", parent=body, alignment=1, textColor=colors.HexColor("#5f6b76"))


def P(t, s=body):
    return Paragraph(t, s)


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def table(rows, widths, header=True):
    t = Table(rows, colWidths=widths, hAlign="CENTER")
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.8),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d0d6")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ALIGN", (1, 0), (-1, -1), "CENTER"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef1")),
                  ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.8)]
    t.setStyle(TableStyle(style))
    return t


def img(path, width_cm):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


tm, tb = m["test_model"], m["test_baseline"]
story = [
    P("Predicción del consumo diario de insumos para la reposición de inventario", title),
    P("Proyecto Capstone · Sistema integrado con Machine Learning", sub),
    P("Nida Hugo Loza Guzmán · Octubre de 2026", sub),
    Spacer(1, 10),

    P("1. Problema y objetivo", h1),
    P("En una cafetería los insumos (leche, café en grano, harina, queso, etc.) se reponen casi siempre "
      "a ojo. Cuando se compra poco hay faltantes y se dejan de vender productos; cuando se compra de más, "
      "los perecibles se vencen. El kardex del sistema de gestión ya registra cada salida de insumo, pero "
      "esa información no se usa para anticipar la compra."),
    P("El objetivo del proyecto es predecir el consumo diario de cada insumo y, con ese pronóstico, decir "
      "cuántos días alcanza el stock y cuánto conviene reponer. La unidad de observación es un insumo en una "
      "fecha y la variable a predecir es la cantidad consumida ese día (salidas del kardex)."),
    P("Se revisó la coherencia de lo registrado en el Excel de proyectos y se mantuvo: la línea base "
      "(mismo día de la semana anterior) es justamente el denominador natural de la MASE con periodo 7, "
      "así que una MASE menor a 1 se lee directamente como “mejor que la línea base”. Se hicieron dos "
      "ajustes con justificación técnica: (a) un solo modelo LightGBM global para los 14 insumos, con las "
      "cantidades normalizadas por la media de entrenamiento de cada insumo, porque por separado varios "
      "insumos tienen pocos datos; y (b) el consumo se toma neto de devoluciones de venta."),

    P("2. Dataset y preparación", h1),
    P("<b>Origen.</b> Datos reales del sistema de gestión de una cafetería de Tarija (PostgreSQL 16), "
      "obtenidos de un respaldo completo de la base de datos. El periodo va del 1 de octubre de 2021 al "
      "3 de octubre de 2026. El respaldo tiene 374 404 movimientos de kardex sobre 19 543 existencias "
      "(envases o unidades abiertas de un lote)."),
    P("<b>Privacidad.</b> El respaldo incluye clientes, personas, usuarios y ventas. El script de "
      "extracción solo lee las tablas de productos, unidades, lotes, existencias y movimientos, y no "
      "exporta identificadores de usuario. El respaldo no se sube al repositorio; se comparte únicamente "
      "el dataset agregado por día."),
    P("<b>Preparación.</b> (1) Cada movimiento se asigna a su insumo base: las presentaciones de compra "
      "(por ejemplo “Harina bolsa 1 kg”) se suman a su insumo padre (“Harina”, en gramos). (2) Se toman los "
      "14 insumos de tipo Insumo. (3) Consumo del día = consumo por producción + consumo por venta − "
      "devoluciones. Las mermas y vencimientos se guardan aparte y no se cuentan como consumo. (4) Se "
      "completa el calendario: un día sin movimientos es consumo cero (en feriados como Navidad o "
      "Carnaval la cafetería no abre). (5) El stock de cierre se reconstruye acumulando entradas "
      "(aperturas, ajustes positivos, devoluciones) menos salidas."),
]

story.append(table([
    ["Dataset final", "Valor"],
    ["Filas (insumo × día)", "25 606"],
    ["Insumos", "14 (g, ml o unidades)"],
    ["Días", "1 829 (01/10/2021 – 03/10/2026)"],
    ["Variables de origen", "date, item, unit, consumption, opened, waste, stock_close"],
    ["Días con consumo cero", "2 % (leche, café) a 50 % (harina, huevo, uso por lotes)"],
], [5.5 * cm, 9.5 * cm]))
story.append(Spacer(1, 6))
story.append(P("<b>Variables del modelo.</b> Para predecir el día t solo se usa información disponible al "
               "cierre de t−1: rezagos del consumo (1, 2, 3, 7, 14, 21 y 28 días), medias móviles de 7 y 28 "
               "días, desviación de 7 días, proporción de días en cero en las últimas 4 semanas, stock de "
               "cierre del día anterior y días de cobertura (stock / media de 7 días). Del calendario: día "
               "de la semana, mes, día del mes, fin de semana y feriado (feriados nacionales de Bolivia, "
               "Carnaval, Viernes Santo, Corpus Christi y el 15 de abril de Tarija). Además el código del "
               "insumo como variable categórica."))

story += [
    P("3. Modelo y línea base", h1),
    P("<b>Línea base.</b> Naive estacional: el consumo de mañana es igual al del mismo día de la semana "
      "anterior (lag 7). No se entrena y se evalúa sobre las mismas filas de prueba que el modelo."),
    P("<b>Modelo.</b> LightGBM 4.7.0 entrenado desde cero (no hay partes preentrenadas). La salida se "
      "entrena sobre el consumo dividido por la media de entrenamiento del insumo y luego se vuelve a "
      "escalar. Parámetros: learning_rate 0,03, num_leaves 31, min_data_in_leaf 40, feature_fraction 0,8, "
      "bagging_fraction 0,8, lambda_l2 1, semilla 42, parada temprana de 100 rondas sobre validación. Se "
      f"probaron tres funciones de pérdida y se eligió la de menor MASE en validación: "
      f"L2 {num(m['selection_on_validation']['regression']['val_mase'])}, "
      f"L1 {num(m['selection_on_validation']['regression_l1']['val_mase'])}, "
      f"Tweedie {num(m['selection_on_validation']['tweedie']['val_mase'])} "
      f"({m['selection_on_validation']['tweedie']['iterations']} árboles). Tweedie tiene sentido aquí "
      "porque el consumo es no negativo y tiene muchos ceros en los insumos de producción."),
    P("<b>Artefactos.</b> model/artifacts/model.txt (modelo LightGBM) y metadata.json (códigos de insumo, "
      "escalas, desviación de residuos de validación y parámetros). El backend carga ambos al iniciar."),

    P("4. Métricas y validación", h1),
    P("<b>Métrica principal: MASE.</b> Error absoluto medio del modelo dividido por el error absoluto "
      "medio del naive estacional (lag 7) calculado en el periodo de entrenamiento de cada insumo. Es "
      "comparable entre insumos con unidades distintas (ml, g, unidades), no se indefine con días en cero "
      "como el MAPE, y menor a 1 indica que el modelo supera a la línea base. Se reporta el promedio de "
      "los 14 insumos."),
    P("<b>Métricas complementarias.</b> WAPE (error absoluto total sobre consumo total) diario y semanal, "
      "porque la compra se decide por semana; y sesgo (error con signo sobre consumo total), porque "
      "subestimar lleva a faltantes y sobreestimar a vencimientos."),
    P("<b>Validación.</b> División por fecha, sin mezclar: entrenamiento 29/10/2021–30/06/2025 "
      f"({str(m['split']['rows']['train'])[:-3]} {str(m['split']['rows']['train'])[-3:]} filas), validación 01/07/2025–31/12/2025 "
      f"({str(m['split']['rows']['validation'])[:-3]} {str(m['split']['rows']['validation'])[-3:]}) y prueba 01/01/2026–03/10/2026 "
      f"({str(m['split']['rows']['test'])[:-3]} {str(m['split']['rows']['test'])[-3:]}). Los primeros 28 días se descartan porque no tienen rezagos "
      "completos. La validación se usó para elegir la pérdida y el número de árboles. La prueba se usó "
      "una sola vez al final. Las escalas de normalización y el denominador de la MASE salen solo de "
      "entrenamiento. Todas las variables están desplazadas al menos un día, así que el consumo y el stock "
      "del propio día nunca entran como predictores."),
]

story.append(P("5. Resultados", h1))
story.append(table([
    ["Conjunto de prueba (promedio de 14 insumos)", "LightGBM", "Línea base (lag 7)"],
    ["MASE (principal)", num(tm["mase_mean"]), num(tb["mase_mean"])],
    ["MASE mediana", num(tm["mase_median"]), num(tb["mase_median"])],
    ["WAPE diario", f"{num(tm['wape_mean'] * 100, 1)} %", f"{num(tb['wape_mean'] * 100, 1)} %"],
    ["WAPE semanal", f"{num(tm['weekly_wape_mean'] * 100, 1)} %", f"{num(tb['weekly_wape_mean'] * 100, 1)} %"],
    ["Sesgo", f"{num(tm['bias_mean'] * 100, 1)} %", f"{num(tb['bias_mean'] * 100, 1)} %"],
    ["Insumos donde gana", f"{m['items_where_model_wins']} de {m['items_total']}", "–"],
], [7.5 * cm, 3.5 * cm, 4 * cm]))
story.append(P("Tabla 1. Resultados en prueba (01/01/2026 – 03/10/2026).", cap))

rows = [["Insumo", "Unidad", "MASE modelo", "MASE base", "WAPE modelo", "WAPE base", "Sesgo modelo"]]
for r in by_item.sort_values("model_mase").itertuples():
    rows.append([r.item, r.unit, num(r.model_mase), num(r.baseline_mase),
                 f"{num(r.model_wape * 100, 1)} %", f"{num(r.baseline_wape * 100, 1)} %",
                 f"{num(r.model_bias * 100, 1)} %"])
story.append(table(rows, [3.6 * cm, 1.4 * cm, 2 * cm, 2 * cm, 2.1 * cm, 2 * cm, 2.2 * cm]))
story.append(P("Tabla 2. Resultados por insumo en prueba.", cap))
story.append(img(RES / "mase_por_insumo.png", 14))
story.append(P("Figura 1. MASE por insumo. La línea roja marca MASE = 1.", cap))

story += [
    P("<b>Análisis de errores.</b>", h2),
    P("• El modelo reduce el error de la línea base en todos los insumos, entre 25 % y 34 %. La mejora "
      "viene sobre todo de no copiar picos aislados: la línea base repite cualquier día atípico de la "
      "semana anterior y el modelo lo suaviza con las medias móviles (Figura 2)."),
    P("• Crema de leche y tomate tienen MASE mayor a 1 en ambos métodos. No es que el modelo sea peor ahí: "
      "esos insumos recién se usan desde 2024 (sándwich napolitano), así que el periodo de entrenamiento "
      "tiene más de dos años en cero y el denominador de la MASE queda muy chico. Su WAPE (46 %) es "
      "parecido al del resto."),
    P("• Harina y huevo son el caso difícil: se consumen por lotes de producción y la mitad de los días "
      "son cero. El WAPE diario es alto (84 %) y el sesgo es negativo (−31 % y −28 %), es decir, el modelo "
      "reparte el consumo entre varios días en vez de acertar el día de producción. Para reponer, lo que "
      "importa es el total de la semana, y ahí el error baja mucho."),
    P("• El modelo subestima en promedio 9,6 % (la línea base casi no tiene sesgo). Por eso la reposición "
      "suma un stock de seguridad con el error de validación de cada insumo."),
    P(f"• Por día de la semana el error más alto está en sábado; en feriados el modelo mantiene el error "
      f"y la línea base empeora (error escalado {num(m['error_holiday_scaled']['abs_err_model']['1'], 2)} "
      f"contra {num(m['error_holiday_scaled']['abs_err_base']['1'], 2)}), porque la base copia el día "
      "cerrado de la semana anterior o no ve el cierre."),
    img(RES / "serie_leche.png", 15),
    P("Figura 2. Leche entera, agosto a octubre de 2026 (prueba).", cap),
    img(RES / "serie_harina.png", 15),
    P("Figura 3. Harina: consumo por lotes, el caso difícil.", cap),
    img(RES / "importancia_variables.png", 11),
    P("Figura 4. Importancia de variables (gain). Pesan más las medias móviles, los rezagos de 7 días y el "
      "día de la semana.", cap),

    P("6. Arquitectura e interfaz", h1),
    P("El sistema tiene tres partes. <b>Datos y modelo:</b> scripts de extracción, armado del dataset y "
      "entrenamiento en Python (pandas, LightGBM). <b>Backend:</b> API en FastAPI que valida la entrada con "
      "Pydantic, arma las variables con el mismo código usado en el entrenamiento (model/features.py) y "
      "ejecuta el modelo cargado de model.txt. Para horizontes de más de un día predice de forma recursiva: "
      "la predicción de mañana entra como rezago para pasado mañana y el stock se descuenta con el consumo "
      "predicho. Luego calcula días de cobertura, stock de seguridad (z = 1,65 por la desviación de "
      "residuos de validación por la raíz de los días de entrega) y cantidad sugerida. <b>Frontend:</b> "
      "página HTML con JavaScript que el mismo backend sirve; permite cargar el historial desde el "
      "kardex o subir un CSV, muestra estado de carga, errores y los límites de la solución."),
]
story.append(table([
    ["Endpoint", "Descripción"],
    ["GET /api/salud", "Estado y versión del modelo"],
    ["GET /api/insumos", "Lista de insumos con unidad y MASE de prueba"],
    ["GET /api/historial", "Historial de un insumo hasta una fecha (para la demostración)"],
    ["POST /api/predecir", "Recibe insumo, historial ≥ 28 días, stock, horizonte y días de entrega"],
], [4.5 * cm, 10.5 * cm]))
story.append(Spacer(1, 8))
story.append(img(SHOTS / "01_caso_valido_leche.png", 15.5))
story.append(P("Figura 5. Caso válido: leche entera, stock 20 000 ml, 7 días, entrega en 2 días.", cap))

story += [
    P("7. Pruebas realizadas", h1),
    P("Las pruebas automáticas están en tests/test_api.py (pytest, 10 pruebas, todas pasan). Cubren:"),
    P("• <b>Caso válido</b>: leche entera con 56 días de historial; responde 7 días con valores no "
      "negativos y en el rango real del insumo."),
    P("• <b>El resultado sale del modelo</b>: dos historiales distintos dan pronósticos distintos (no hay "
      "respuestas fijas)."),
    P("• <b>Caso difícil</b>: harina con consumo intermitente y stock en cero a 14 días; el sistema avisa que "
      "hay que pedir hoy y sugiere una cantidad positiva."),
    P("• <b>Entradas inválidas</b>: insumo desconocido, menos de 28 días, consumo negativo, horizonte mayor a "
      "14 y días faltantes en el historial. Todas devuelven 422 con un mensaje en español."),
    P("• <b>Flujo completo</b>: página principal → lista de insumos → historial → predicción, por el mismo "
      "camino que usa la interfaz."),
    P("Las pruebas se corrieron en un entorno virtual con las versiones fijadas en requirements.txt, siguiendo los pasos del README."),
    img(SHOTS / "02_caso_dificil_harina.png", 15.5),
    P("Figura 6. Caso difícil: harina, stock 0, horizonte 14 días.", cap),
    img(SHOTS / "03_entrada_invalida.png", 15.5),
    P("Figura 7. Entrada inválida: historial de solo 3 días. El backend rechaza la petición y explica el motivo.", cap),

    P("8. Conclusiones", h1),
    P(f"El modelo LightGBM supera a la línea base en los 14 insumos, con MASE promedio de "
      f"{num(tm['mase_mean'])} frente a {num(tb['mase_mean'])} y mediana de {num(tm['mase_median'])}. La "
      "comparación se hizo sobre nueve meses de 2026 que el modelo no vio, con división temporal y sin "
      "fugas. El sistema funciona de punta a punta: se carga el historial, el backend ejecuta el modelo real "
      "y la interfaz muestra cuánto se va a consumir, cuántos días alcanza el stock y cuánto pedir."),
    P("La mejora es clara en el día a día, pero en el total semanal la diferencia es menor (18,9 % contra "
      "20,9 % de WAPE). Para la decisión de compra, que se hace por semana, la ganancia práctica está más en "
      "evitar sobrerreacciones a días atípicos que en un acierto diario fino."),

    P("9. Limitaciones", h1),
    P("• Se entrenó a un día. Para 2 a 14 días el pronóstico es recursivo y el error crece; no se evaluó "
      "por separado el error a 7 días."),
    P("• El stock disponible es el saldo de envases abiertos según el kardex, no el inventario de almacén; "
      "por eso en la interfaz el stock lo ingresa el usuario."),
    P("• No hay datos de promociones, eventos, clima ni cortes de proveedor."),
    P("• Subestima en promedio; en insumos de producción por lotes el sesgo llega a −31 %."),
    P("• Es una sola cafetería; no se probó en otros negocios."),

    P("10. Referencias", h1),
    P("Hyndman, R. J., &amp; Koehler, A. B. (2006). Another look at measures of forecast accuracy. "
      "<i>International Journal of Forecasting, 22</i>(4), 679–688."),
    P("Hyndman, R. J., &amp; Athanasopoulos, G. (2021). <i>Forecasting: Principles and Practice</i> (3.ª ed.). "
      "OTexts. https://otexts.com/fpp3/"),
    P("Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., &amp; Liu, T.-Y. (2017). LightGBM: "
      "A highly efficient gradient boosting decision tree. <i>Advances in Neural Information Processing "
      "Systems, 30</i>."),
    P("Makridakis, S., Spiliotis, E., &amp; Assimakopoulos, V. (2022). M5 accuracy competition: Results, "
      "findings, and conclusions. <i>International Journal of Forecasting, 38</i>(4), 1346–1364."),
    P("Silver, E. A., Pyke, D. F., &amp; Thomas, D. J. (2017). <i>Inventory and Production Management in "
      "Supply Chains</i> (4.ª ed.). CRC Press."),
]


def footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#7a8590"))
    canvas.drawString(2 * cm, 1.2 * cm, "Predicción del consumo de insumos · Proyecto Capstone")
    canvas.drawRightString(19 * cm, 1.2 * cm, str(doc.page))


doc = SimpleDocTemplate(str(ROOT / "docs" / "informe_tecnico.pdf"), pagesize=A4,
                        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Predicción del consumo diario de insumos", author="Nida Hugo Loza Guzmán")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("ok")
