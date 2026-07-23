import requests
import polars as pl
from io import StringIO

url = "https://gist.githubusercontent.com/epam-latam/b22217692beb4ae0b407d750b95de600/raw/081847e6eb68375262708b2d3e6a277c7d2868d7/resenas_flowapp.csv"
response = requests.get(url)

#carga de datos con polars, forzando rating a string para limpiar después
df = pl.read_csv(
    StringIO(response.text),
    schema_overrides={"rating": pl.Utf8},
)

print("Valores únicos en rating (antes de limpiar):")
print(df["rating"].unique().sort())

# Castear rating a numérico, forzando inválidos a null
df_og = df.with_columns(
    pl.col("rating").str
    .strip_chars()
    .cast(pl.Int64, strict=False).alias("rating")
)

df = df_og.with_columns(
    pl.when(pl.col("rating").is_between(1, 5))
    .then(pl.col("rating"))
    .otherwise(None)
    .alias("rating")
)
n_nulos = df.filter(pl.col("rating").is_null()).height
print("\n=============================================================")
print(f"Registros con rating inválido/faltante: {n_nulos} de {df.height}")
print("============================================================= \n")
# ============================================================
# 1. Limpie el texto (minúsculas, sin caracteres especiales, sin duplicados)
# ============================================================


df_clean = (
    df
    .with_columns(
        pl.col("texto")
        .str.to_lowercase() # convertir a minúsculas todo el texto que se encuentra en la columna "texto"
        .str.replace_all(r"[^a-záéíóúñü0-9\s]", " ") # reemplazar todos los caracteres que no sean letras, números o espacios por un espacio
        .str.replace_all(r"\s+", " ") # quita los dobles espacios que genere la limpieza anterior
        .str.strip_chars()# quita los espacios al inicio y al final del texto
        .alias("texto_clean") # cambia el nombre de la columna a "texto_clean"
    )
    .unique(subset=["texto_clean"]) # elimina los registros duplicados basándose en la columna "texto_clean"
)

n_texto_vacio = df.filter(
    pl.col("texto").is_null() | (pl.col("texto").str.strip_chars() == "")
).height

# 1. Reportar y limpiar texto vacío/nulo
print("=============================================================")
print(f"Reseñas con texto vacío/nulo: {n_texto_vacio}")
print("============================================================= \n")

# 2. Detectar posibles duplicados de ENVÍO (mismo usuario + misma fecha)
posibles_duplicados = df.filter(pl.struct(["usuario","fecha"]).is_duplicated())

print("=============================================================")
print(f"Filas con mismo usuario+fecha (posible duplicado de envío): {posibles_duplicados.height}")
print("============================================================= \n")

# 3. Reportar cuántos duplicados exactos de texto se eliminaron
antes = df.height
df_clean = df_clean.unique(subset=["texto_clean"])
despues = df_clean.height
print("=============================================================")
print(f"Duplicados exactos de texto eliminados: {antes - despues}")
print("============================================================= \n")
# ============================================================
# 2. Genere un resumen estadístico (promedio, mediana, distribución de ratings)
# ============================================================
df_validos = df_clean.filter(pl.col("rating").is_not_null()) 

resumen = df_validos.select(
    pl.col("rating").mean().alias("promedio"), #Se calcula el promedio de la columna "rating" y se le asigna el alias "promedio"
    pl.col("rating").median().alias("mediana"), #Se calcula la mediana de la columna "rating" y se le asigna el alias "mediana"
    pl.col("rating").std().alias("desv_std"), #Se calcula la desviación estándar de la columna "rating" y se le asigna el alias "desv_std"
    pl.col("rating").min().alias("min"),#Se calcula el valor mínimo de la columna "rating" y se le asigna el alias "min"
    pl.col("rating").max().alias("max"), #Se calcula el valor máximo de la columna "rating" y se le asigna el alias "max"
)

distribucion = (
    df_validos
    .group_by("rating")
    .agg(pl.len().alias("conteo"))
    .sort("rating")
    .with_columns(
        (pl.col("conteo") / pl.col("conteo").sum() * 100).round(2).alias("porcentaje")
    )
)

# ============================================================
# 3. Calcule las palabras más frecuentes por nivel de rating
# ============================================================
STOPWORDS = {"el","la","de","que","y","a","en","un","es","se","no","los","con","por","las",
             "lo","le","del","al","su","mi","me","muy","para","como","más","pero"}

palabras_por_rating = (
    df_validos
    .select(["rating", "texto_clean"])
    .with_columns(pl.col("texto_clean").str.split(" ").alias("palabras"))
    .explode("palabras")
    .filter((pl.col("palabras") != "") & (~pl.col("palabras").is_in(STOPWORDS))) # filtra palabras vacías y stopwords (No cuenta los conectores y los espacios)
    .group_by(["rating", "palabras"])
    .agg(pl.len().alias("frecuencia"))
    .sort(["rating", "frecuencia"], descending=[False, True])
)

top_n = 10
top_palabras = (
    palabras_por_rating
    .with_columns(
        pl.col("frecuencia").rank(method="ordinal", descending=True).over("rating").alias("rank")
    )
    .filter(pl.col("rank") <= top_n)
    .drop("rank")
)
print("==============================Resumen estadístico===============================")
print(resumen)
print("================================================================================\n")

print("==============================Distribución de ratings===============================")
print(distribucion)
print("================================================================================\n")
print("==============================Palabras más frecuentes por rating===============================")
print(top_palabras)
print("===============================================================================================\n")