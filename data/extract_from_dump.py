"""Extract kardex movements of supplies (insumos) from the PostgreSQL dump.

Only non-personal tables are read: product catalog, units, lots, stock
containers and kardex movements. Customer, person, user, sales and cash
tables are skipped on purpose, and user ids are not exported.

Usage:
    python data/extract_from_dump.py path/to/backup.sql
Outputs (in data/):
    catalog.csv        product catalog (id, name, type, unit, parent)
    raw_movements.csv  one row per kardex movement
"""
import csv
import sys
from pathlib import Path

TABLES = {
    "producto", "unidad", "tipo", "tipo_movimiento", "conversiones",
    "detalle_lote", "existencias", "movimientos",
}


def read_copy_blocks(dump_path):
    """Yield (table, row_dict) for every COPY row of the wanted tables."""
    table, cols = None, None
    with open(dump_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if table is None:
                if line.startswith("COPY public."):
                    name = line.split()[1].split(".")[1]
                    if name in TABLES:
                        table = name
                        cols = line[line.index("(") + 1: line.index(")")].split(", ")
                continue
            if line == "\\.":
                table, cols = None, None
                continue
            values = [None if v == "\\N" else v for v in line.split("\t")]
            yield table, dict(zip(cols, values))


def main(dump_path):
    data = {t: [] for t in TABLES}
    for table, row in read_copy_blocks(dump_path):
        data[table].append(row)

    products = {p["id"]: p for p in data["producto"]}
    units = {u["id"]: u["abreviatura"] for u in data["unidad"]}
    types = {t["id"]: t["nombre"] for t in data["tipo"]}
    mov_types = {t["id"]: t["nombre"] for t in data["tipo_movimiento"]}
    lots = {d["id"]: d for d in data["detalle_lote"]}
    stocks = {e["id"]: e for e in data["existencias"]}

    out_dir = Path(__file__).resolve().parent
    with open(out_dir / "catalog.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["product_id", "name", "type", "unit", "parent_id", "stock", "stock_min"])
        for p in products.values():
            w.writerow([p["id"], p["nombre"], types.get(p["id_tipo"]),
                        units[p["unidad_actual"]], p["id_producto_padre"],
                        p["stock"], p["stock_minimo"]])

    with open(out_dir / "conversions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["from_unit", "to_unit", "factor"])
        for c in data["conversiones"]:
            w.writerow([units[c["id_unidad_origen"]], units[c["id_unidad_destino"]],
                        c["factor_conversion"]])

    with open(out_dir / "raw_movements.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "base_product_id", "lot_product_id", "movement_type",
                    "quantity", "stock_unit", "stock_initial", "deleted"])
        for m in data["movimientos"]:
            st = stocks.get(m["id_existencia"])
            if st is None:
                continue
            prod = products[lots[st["id_detalle_lote"]]["id_producto"]]
            base_id = prod["id_producto_padre"] or prod["id"]
            w.writerow([m["fecha_movimiento"], base_id, prod["id"],
                        mov_types[m["id_tipo_movimiento"]], m["cantidad"],
                        units[st["unidad"]], st["contenido_inicial"], m["borrado"]])
    print("movements:", len(data["movimientos"]), "stocks:", len(stocks))


if __name__ == "__main__":
    main(sys.argv[1])
