# Decision de exclusion del periodo 2014-1

## Contexto

Se realizo una comparacion de encabezados (columnas) entre los archivos de `datos_crudos/` correspondientes a los periodos del examen Saber 11.

## Decision

Se excluye el archivo `Examen_Saber_11_20141.txt` del conjunto principal de analisis.

## Razon tecnica

El periodo `2014-1` tiene una estructura de columnas significativamente distinta frente al resto de periodos:

- `2014-1` tiene 138 columnas.
- Los demas periodos tienen entre 57 y 85 columnas.
- Al incluir `2014-1`, la interseccion comun entre todos los archivos baja a 46 columnas.
- Al excluir `2014-1`, la interseccion comun sube a 55 columnas.

Excluir `2014-1` mejora la consistencia del esquema longitudinal y reduce perdida de informacion util en variables comparables entre periodos.

## Limpieza adicional por valores nulos

Sobre el consolidado `bases_datos/v1_todos_los_datos.csv` se aplico exclusion de filas con al menos un valor nulo (`dropna` por fila).

- Filas originales: `7,239,505`
- Filas conservadas sin nulos: `4,821,037`
- Filas excluidas por nulos: `2,418,468`
- Porcentaje conservado: `66.5935%` (aprox. `66.59%`)

Esta regla tambien quedo implementada en `bases_datos/merge_columnas_comunes.py`, para que el consolidado se exporte ya depurado de nulos.

## Columnas comunes conservadas (55)

1. `cole_area_ubicacion`
2. `cole_bilingue`
3. `cole_calendario`
4. `cole_caracter`
5. `cole_cod_dane_establecimiento`
6. `cole_cod_dane_sede`
7. `cole_cod_depto_ubicacion`
8. `cole_cod_mcpio_ubicacion`
9. `cole_codigo_icfes`
10. `cole_depto_ubicacion`
11. `cole_genero`
12. `cole_jornada`
13. `cole_mcpio_ubicacion`
14. `cole_naturaleza`
15. `cole_nombre_establecimiento`
16. `cole_nombre_sede`
17. `cole_sede_principal`
18. `desemp_ingles`
19. `estu_agregado`
20. `estu_cod_depto_presentacion`
21. `estu_cod_mcpio_presentacion`
22. `estu_cod_reside_depto`
23. `estu_cod_reside_mcpio`
24. `estu_consecutivo`
25. `estu_depto_presentacion`
26. `estu_depto_reside`
27. `estu_estudiante`
28. `estu_fechanacimiento`
29. `estu_genero`
30. `estu_grado`
31. `estu_inse_individual`
32. `estu_mcpio_presentacion`
33. `estu_mcpio_reside`
34. `estu_nacionalidad`
35. `estu_nse_individual`
36. `estu_pais_reside`
37. `estu_privado_libertad`
38. `estu_tipodocumento`
39. `fami_cuartoshogar`
40. `fami_educacionmadre`
41. `fami_educacionpadre`
42. `fami_estratovivienda`
43. `fami_personashogar`
44. `fami_tieneautomovil`
45. `fami_tienecomputador`
46. `fami_tieneinternet`
47. `fami_tienelavadora`
48. `fami_tieneserviciotv`
49. `periodo`
50. `punt_c_naturales`
51. `punt_global`
52. `punt_ingles`
53. `punt_lectura_critica`
54. `punt_matematicas`
55. `punt_sociales_ciudadanas`
