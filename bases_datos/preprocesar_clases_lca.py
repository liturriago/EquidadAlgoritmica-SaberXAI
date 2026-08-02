import polars as pl
from pathlib import Path

INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "saber11_lca_clases.parquet"
OUTPUT_DIR = INPUT_PATH.parent

# --- Mapeos ordinales ---
map_estrato = {
    "Sin Estrato": 0,
    "Estrato 1": 1, "Estrato 2": 2, "Estrato 3": 3,
    "Estrato 4": 4, "Estrato 5": 5, "Estrato 6": 6,
}

map_cuartos = {
    "Uno": 1, "Dos": 2, "Tres": 3, "Cuatro": 4,
    "Cinco": 5, "Seis O Más": 6,
}

map_personas = {
    "1 A 2": 1, "3 A 4": 3, "5 A 6": 5,
    "7 A 8": 7, "9 O Más": 9,
}

cols_binarias = [
    "fami_tieneautomovil", "fami_tienecomputador", "fami_tieneinternet",
    "fami_tienelavadora", "fami_tieneserviciotv", "cole_bilingue", "cole_sede_principal",
]

cols_categoricas = [
    "cole_area_ubicacion", "cole_calendario", "cole_caracter",
    "cole_depto_ubicacion", "cole_genero", "cole_jornada",
    "cole_naturaleza", "estu_genero",
]

cols_a_eliminar = ["clase_lca", "prob_max_lca"]


def transformar(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.drop(cols_a_eliminar)
        # Binarias: Si/No -> 1/0
        .with_columns([
            pl.when(pl.col(c).str.to_lowercase() == "si")
              .then(1).otherwise(0).cast(pl.Int8).alias(c)
            for c in cols_binarias
        ])
        # Ordinales
        .with_columns([
            pl.col("fami_estratovivienda").replace_strict(map_estrato, default=None).cast(pl.Int8),
            pl.col("fami_cuartoshogar").replace_strict(map_cuartos, default=None).cast(pl.Int8),
            pl.col("fami_personashogar").replace_strict(map_personas, default=None).cast(pl.Int8),
        ])
        # Categoricals
        .with_columns([
            pl.col(c).cast(pl.Categorical) for c in cols_categoricas
        ])
    )


def main():
    print(f"Cargando {INPUT_PATH}...")
    df = pl.read_parquet(INPUT_PATH)
    print(f"Total: {len(df):,} filas")

    clases = sorted(df["clase_lca"].unique().to_list())
    print(f"Clases LCA detectadas: {clases}")

    for clase in clases:
        print(f"\nProcesando clase {clase}...")
        df_clase = df.filter(pl.col("clase_lca") == clase)
        df_procesado = transformar(df_clase)

        out_path = OUTPUT_DIR / f"clase_{clase}.parquet"
        df_procesado.write_parquet(out_path, compression="zstd")
        print(f"  -> {len(df_procesado):,} filas guardadas en {out_path.name}")

    print("\nProceso completado.")


if __name__ == "__main__":
    main()
