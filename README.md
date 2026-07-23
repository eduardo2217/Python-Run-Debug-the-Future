# Reto EPAM — Análisis de Reseñas FlowApp

## Descripción

Este proyecto procesa un dataset de reseñas de usuarios de FlowApp (`resenas_flowapp.csv`) para generar:
- Un resumen estadístico de los ratings
- Las palabras más frecuentes por nivel de rating
- Un dataset de texto limpio y sin duplicados

## Stack

- Python + [Polars](https://pola.rs/) (procesamiento de datos vectorizado)
- `requests` (descarga del dataset desde un gist remoto)

## Estructura del pipeline

1. **Carga de datos**: lectura del CSV directamente desde una URL remota vía `requests` + `StringIO`.
2. **Validación y limpieza de `rating`**:
   - La columna se lee como texto (`Utf8`) para evitar errores de parseo.
   - Se castea a entero con `strict=False`, convirtiendo valores no numéricos en `null`.
   - Se valida el rango esperado (1-5); valores fuera de rango también se marcan como `null`.
3. **Limpieza de texto (`texto`)**:
   - Conversión a minúsculas.
   - Eliminación de caracteres especiales y emojis (se conservan letras, tildes, ñ/ü, números y espacios).
   - Normalización de espacios múltiples.
   - Eliminación de reseñas duplicadas (texto idéntico tras la limpieza).
4. **Resumen estadístico**: promedio, mediana, desviación estándar, mínimo y máximo, calculados **solo sobre ratings válidos**.
5. **Distribución de ratings**: conteo y porcentaje por cada nivel de rating.
6. **Palabras más frecuentes por rating**: se excluyen stopwords en español; se obtiene el top 10 de palabras por nivel de rating usando `explode` + `group_by` + `rank`.



## Hallazgos de calidad de datos

Durante el desarrollo se identificaron varios problemas en el dataset original, todos manejados explícitamente en el pipeline:

| Problema encontrado | Ejemplo | Tratamiento aplicado |
|---|---|---|
| Ratings no numéricos | `?`, `N/A` | Casteo con `strict=False` → `null` |
| Ratings fuera de rango válido (1-5) | `-1`, `0`, `6`, `7` | Validación de rango explícita → `null` |
| Texto con espacios extra al inicio/fin | `"  Muy recomendada...  "` | `str.strip_chars()` |
| Mayúsculas inconsistentes | `"MUY RECOMENDADA"` vs `"Muy recomendada"` | `str.to_lowercase()` |
| Emojis y signos de puntuación | `👏`, `😐`, `!`, `,` | Reemplazo por espacio vía regex |
| Reseñas duplicadas entre filas | — | `.unique(subset=["texto_clean"])` |


### Cantidad de registros afectados
- Registros con rating inválido/faltante (no numérico o fuera de rango): *(completar con el número real que te dio tu script)*
- Reseñas duplicadas eliminadas: *(completar)*
- **Reseñas con texto vacío/nulo**: X
- **Filas con posible duplicado de envío (mismo usuario + fecha)**: X

## Cómo ejecutar

```bash
uv add polars requests
uv run reto_1.py

O

uv sync
uv run reto_1.py
```

"# Python-Run-Debug-the-Future" 
