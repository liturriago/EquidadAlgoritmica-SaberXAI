from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


COLUMNAS_COMUNES: List[str] = [
    "cole_area_ubicacion",
    "cole_bilingue",
    "cole_calendario",
    "cole_caracter",
    "cole_cod_dane_establecimiento",
    "cole_cod_dane_sede",
    "cole_cod_depto_ubicacion",
    "cole_cod_mcpio_ubicacion",
    "cole_codigo_icfes",
    "cole_depto_ubicacion",
    "cole_genero",
    "cole_jornada",
    "cole_mcpio_ubicacion",
    "cole_naturaleza",
    "cole_nombre_establecimiento",
    "cole_nombre_sede",
    "cole_sede_principal",
    "desemp_ingles",
    "estu_agregado",
    "estu_cod_depto_presentacion",
    "estu_cod_mcpio_presentacion",
    "estu_cod_reside_depto",
    "estu_cod_reside_mcpio",
    "estu_consecutivo",
    "estu_depto_presentacion",
    "estu_depto_reside",
    "estu_estudiante",
    "estu_fechanacimiento",
    "estu_genero",
    "estu_grado",
    "estu_inse_individual",
    "estu_mcpio_presentacion",
    "estu_mcpio_reside",
    "estu_nacionalidad",
    "estu_nse_individual",
    "estu_pais_reside",
    "estu_privado_libertad",
    "estu_tipodocumento",
    "fami_cuartoshogar",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "fami_estratovivienda",
    "fami_personashogar",
    "fami_tieneautomovil",
    "fami_tienecomputador",
    "fami_tieneinternet",
    "fami_tienelavadora",
    "fami_tieneserviciotv",
    "periodo",
    "punt_c_naturales",
    "punt_global",
    "punt_ingles",
    "punt_lectura_critica",
    "punt_matematicas",
    "punt_sociales_ciudadanas",
]


def listar_archivos_datos(carpeta_datos: Path) -> Iterable[Path]:
    archivos = sorted(carpeta_datos.glob("Examen_Saber_11_*.txt"))
    # Excluye 2014-1 si el archivo existe.
    return [a for a in archivos if "20141" not in a.stem]


def leer_y_filtrar(archivo: Path, columnas_objetivo: List[str]) -> pd.DataFrame:
    df = pd.read_csv(archivo, sep=";", dtype=str, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]

    faltantes = [c for c in columnas_objetivo if c not in df.columns]
    for col in faltantes:
        df[col] = pd.NA

    return df[columnas_objetivo]


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    carpeta_datos = base_dir / "datos_crudos"
    salida_general = (
        base_dir / "bases_datos" / "saber11_columnas_comunes_20142_20242.csv"
    )
    salida_urbano = (
        base_dir / "bases_datos" / "saber11_columnas_comunes_urbano_20142_20242.csv"
    )
    salida_rural = (
        base_dir / "bases_datos" / "saber11_columnas_comunes_rural_20142_20242.csv"
    )

    if not carpeta_datos.exists():
        raise FileNotFoundError(f"No existe la carpeta de datos: {carpeta_datos}")

    archivos = list(listar_archivos_datos(carpeta_datos))
    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos Examen_Saber_11_*.txt en {carpeta_datos}"
        )

    frames = []
    for archivo in archivos:
        print(f"Leyendo: {archivo.name}")
        frames.append(leer_y_filtrar(archivo, COLUMNAS_COMUNES))

    df_unificado = pd.concat(frames, ignore_index=True)
    filas_antes_dropna = len(df_unificado)
    df_unificado = df_unificado.dropna(how="any").reset_index(drop=True)
    filas_despues_dropna = len(df_unificado)
    filas_eliminadas = filas_antes_dropna - filas_despues_dropna

    salida_general.parent.mkdir(parents=True, exist_ok=True)
    df_unificado.to_csv(salida_general, sep=";", index=False, encoding="utf-8")

    # Version separada por zona segun area del colegio.
    area = df_unificado["cole_area_ubicacion"].fillna("").str.strip().str.upper()
    df_urbano = df_unificado[area == "URBANO"].copy()
    df_rural = df_unificado[area == "RURAL"].copy()
    df_urbano.to_csv(salida_urbano, sep=";", index=False, encoding="utf-8")
    df_rural.to_csv(salida_rural, sep=";", index=False, encoding="utf-8")

    print(
        f"\nMerge finalizado. Filas: {len(df_unificado):,} | "
        f"Columnas: {len(df_unificado.columns)}"
    )
    print(
        f"Filas eliminadas por NaN: {filas_eliminadas:,} "
        f"(de {filas_antes_dropna:,} a {filas_despues_dropna:,})"
    )
    print(f"Archivo general: {salida_general}")
    print(f"Archivo urbano: {salida_urbano} | Filas: {len(df_urbano):,}")
    print(f"Archivo rural: {salida_rural} | Filas: {len(df_rural):,}")


if __name__ == "__main__":
    main()
