#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFM: Predicción de la Empleabilidad de Personas con Discapacidad
Script / Notebook Interactivo: 02_feature_engineering.py
----------------------------------------------------------------
Fase 2: Preprocesamiento e Ingeniería de Características
- Carga del dataset depurado de la EDAD 2020.
- Aislamiento de variables de target leakage y metadatos.
- Construcción de nuevas características (Feature Engineering).
- Partición estratificada Train / Test (80/20).
- Configuración y ajuste de Scikit-Learn ColumnTransformer.
- Exportación de datasets listos para modelado y serialización del preprocesador.
"""

# %% 1. Importación de Librerías y Configuración
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

warnings.filterwarnings('ignore')

# Rutas adaptables según entorno de ejecución
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if "__file__" in locals() else os.getcwd()
INPUT_DATA_PATH = os.path.join(BASE_DIR, "Data", "processed", "dataset_empleabilidad_edad2020.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "processed", "train_test")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 75)
print("FASE 2: PREPROCESAMIENTO Y FEATURE ENGINEERING")
print("=" * 75)

# %% 2. Carga del Dataset
if not os.path.exists(INPUT_DATA_PATH):
    # Alternativa si se ejecuta desde raíz
    INPUT_DATA_PATH = os.path.join("Data", "processed", "dataset_empleabilidad_edad2020.csv")

print(f"Cargando dataset base desde: {INPUT_DATA_PATH}")
df = pd.read_csv(INPUT_DATA_PATH, sep=";", decimal=",", encoding="utf-8-sig")
print(f"Dataset cargado correctamente: {df.shape[0]:,} filas x {df.shape[1]} columnas.\n")

# %% 3. Feature Engineering (Creación de Nuevas Variables)
print("Construyendo variables engineered basadas en el dominio...")

df_fe = df.copy()

# 1. Brecha Digital / Conectividad (Índice 0 a 2)
# 0: Sin TIC, 1: Conectividad parcial (solo PC o solo Internet), 2: Conectividad plena
df_fe['indice_conectividad'] = df_fe['tiene_internet'].astype(int) + df_fe['tiene_ordenador'].astype(int)

# 2. Severidad Funcional y Multidiversidad
# Pluridiscapacidad grave (2 o más limitaciones graves acumuladas)
df_fe['pluridiscapacidad_grave'] = (df_fe['num_limitaciones_graves'] >= 2).astype(int)

# Limitación sensorial combinada (visión o audición)
df_fe['lim_sensorial'] = ((df_fe['lim_vision'] == 1) | (df_fe['lim_audicion'] == 1)).astype(int)

# Limitación grave de autonomía básica / dependencia (movilidad o autocuidado)
df_fe['lim_autonomia_diaria'] = ((df_fe['lim_movilidad'] == 1) | (df_fe['lim_autocuidado'] == 1)).astype(int)

# Limitación cognitivo-comunicativa (aprendizaje o comunicación)
df_fe['lim_cognitivo_relacional'] = ((df_fe['lim_aprendizaje'] == 1) | (df_fe['lim_comunicacion'] == 1)).astype(int)

# 3. Interacciones Demográficas y de Entorno del Hogar
# Interacción edad sénior (>=50 años) con limitaciones graves acumuladas
df_fe['edad_senior_con_limitacion'] = ((df_fe['edad'] >= 50) & (df_fe['num_limitaciones_graves'] >= 1)).astype(int)

# Hogar unipersonal / vive solo (vulnerabilidad o autonomía de convivencia)
df_fe['vive_solo'] = ((df_fe['tamano_hogar'] == 1) | (df_fe['tipo_hogar'] == 'Hogar unipersonal')).astype(int)

print(f"Nuevas variables creadas: 'indice_conectividad', 'pluridiscapacidad_grave', "
      f"'lim_sensorial', 'lim_autonomia_diaria', 'lim_cognitivo_relacional', "
      f"'edad_senior_con_limitacion', 'vive_solo'.")

# %% 4. Separación de Variables: Predictores (X), Target (y) y Metadatos
# Se descartan identificadores y variables que generan target leakage
cols_excluir = [
    'IDENTHOGAR',               # Identificador de hogar
    'NORDEN',                   # Identificador de individuo
    'empleado',                 # Target
    'situacion_laboral',        # Target leakage directa (empleado deriva de ella)
    'factor_elevacion',         # Ponderador muestral INE (se conserva para evaluación, no como feature)
    'nivel_estudios',           # Se usa nivel_estudios_agrupado (más limpio y ordinal)
    'tramo_edad',               # Se usa edad continua + edad_senior_con_limitacion
    'grado_discapacidad_pct'    # Gran porcentaje de nulos; capturado por tramo_grado_discapacidad
]

features = [c for c in df_fe.columns if c not in cols_excluir]
X = df_fe[features].copy()
y = df_fe['empleado'].copy()
metadata_eval = df_fe[['IDENTHOGAR', 'NORDEN', 'situacion_laboral', 'factor_elevacion']].copy()

print(f"\nMatriz de características X: {X.shape[1]} variables predictoras seleccionadas.")
print(f"Variable objetivo y: {y.name} (Distribución: {y.value_counts(normalize=True).to_dict()})")

# %% 5. Partición Estratificada Train / Test (80% / 20%)
print("\nRealizando partición estratificada Train / Test (80 / 20)...")
X_train_raw, X_test_raw, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, metadata_eval,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"  - Conjunto de Entrenamiento (Train): {X_train_raw.shape[0]:,} muestras ({X_train_raw.shape[0]/len(df)*100:.1f}%)")
print(f"  - Conjunto de Prueba (Test):         {X_test_raw.shape[0]:,} muestras ({X_test_raw.shape[0]/len(df)*100:.1f}%)")
print(f"  - Tasa de positivos en Train:        {y_train.mean()*100:.2f}%")
print(f"  - Tasa de positivos en Test:         {y_test.mean()*100:.2f}%")

# %% 6. Definición del Pipeline de Preprocesamiento (ColumnTransformer)
# Identificación de tipos de variables
vars_ordinales = [
    'nivel_estudios_agrupado',
    'tramo_grado_discapacidad',
    'tamano_municipio',
    'ingresos_hogar'
]

categorias_ordinales = [
    # nivel_estudios_agrupado
    ['Sin estudios / Primaria', 'Secundaria / Bachillerato', 'Formación Profesional', 'Universidad / Posgrado'],
    # tramo_grado_discapacidad
    ['Sin grado reconocido / No consta', 'Menos del 33%', 'De 33% a 64% (Moderado)', 'De 65% a 74% (Severo)', '75% o más (Muy severo / Gran invalidez)'],
    # tamano_municipio
    ['Menos de 10.000 hab', '10.000 a 19.999 hab', '20.000 a 49.999 hab', '50.000 a 99.999 hab', '100.000 a 499.999 hab', 'Más de 500.000 hab'],
    # ingresos_hogar
    ['No consta', 'Menos de 500 €', '500 a 999 €', '1.000 a 1.499 €', '1.500 a 1.999 €', '2.000 a 2.499 €', '2.500 a 2.999 €', '3.000 a 4.999 €', '5.000 € o más']
]

vars_nominales = [
    'sexo',
    'ccaa',
    'estado_civil',
    'tipo_hogar'
]

vars_numericas = [
    'edad',
    'tamano_hogar',
    'num_limitaciones_graves'
]

vars_binarias = [
    'certificado_oficial',
    'lim_vision',
    'lim_audicion',
    'lim_comunicacion',
    'lim_aprendizaje',
    'lim_movilidad',
    'lim_autocuidado',
    'tiene_internet',
    'tiene_ordenador',
    'indice_conectividad',
    'pluridiscapacidad_grave',
    'lim_sensorial',
    'lim_autonomia_diaria',
    'lim_cognitivo_relacional',
    'edad_senior_con_limitacion',
    'vive_solo'
]

# Construcción de transformadores
ordinal_transformer = OrdinalEncoder(
    categories=categorias_ordinales,
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

nominal_transformer = OneHotEncoder(
    drop='first',
    sparse_output=False,
    handle_unknown='ignore'
)

numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, vars_numericas),
        ('ord', ordinal_transformer, vars_ordinales),
        ('nom', nominal_transformer, vars_nominales),
        ('bin', 'passthrough', vars_binarias)
    ],
    remainder='drop'
)

# %% 7. Ajuste del Preprocesador (EXCLUSIVAMENTE con datos de Train)
print("\nAjustando preprocesador únicamente sobre X_train (prevención de fuga de datos)...")
preprocessor.fit(X_train_raw)

# Obtención de nombres de características transformadas
def get_column_names_from_preprocessor(ct):
    output_cols = []
    for name, trans, cols in ct.transformers_:
        if name == 'remainder' and trans == 'drop':
            continue
        if hasattr(trans, 'get_feature_names_out'):
            try:
                names = trans.get_feature_names_out(cols)
                output_cols.extend(names)
                continue
            except Exception:
                pass
        if name == 'bin' or trans == 'passthrough':
            output_cols.extend(cols)
        elif name == 'num':
            output_cols.extend([f"num__{c}" for c in cols])
        elif name == 'ord':
            output_cols.extend([f"ord__{c}" for c in cols])
        elif name == 'nom':
            try:
                names = trans.get_feature_names_out(cols)
                output_cols.extend(names)
            except Exception:
                output_cols.extend([f"nom__{c}" for c in cols])
    return output_cols

feature_names = get_column_names_from_preprocessor(preprocessor)

# Transformación de datos
X_train_proc = pd.DataFrame(
    preprocessor.transform(X_train_raw),
    columns=feature_names,
    index=X_train_raw.index
)

X_test_proc = pd.DataFrame(
    preprocessor.transform(X_test_raw),
    columns=feature_names,
    index=X_test_raw.index
)

print(f"Transformación completada con éxito.")
print(f"Dimensiones de X_train_proc: {X_train_proc.shape[0]:,} filas x {X_train_proc.shape[1]} columnas.")
print(f"Dimensiones de X_test_proc:  {X_test_proc.shape[0]:,} filas x {X_test_proc.shape[1]} columnas.")

# Verificación de valores nulos
nulos_train = X_train_proc.isnull().sum().sum()
nulos_test = X_test_proc.isnull().sum().sum()
print(f"Valores nulos en X_train procesado: {nulos_train}")
print(f"Valores nulos en X_test procesado:  {nulos_test}")
assert nulos_train == 0, "Error: Existen valores nulos en el conjunto de entrenamiento procesado."
assert nulos_test == 0, "Error: Existen valores nulos en el conjunto de prueba procesado."

# %% 8. Exportación de Artefactos y Conjuntos de Datos
print("\nExportando artefactos a disco...")

# 1. Guardar el objeto preprocesador ajustado
preprocessor_file = os.path.join(MODELS_DIR, "preprocessor.joblib")
joblib.dump(preprocessor, preprocessor_file)
print(f"  [+] Preprocesador serializado guardado en: {preprocessor_file}")

# 2. Guardar datasets procesados (para modelos lineales, SVM, redes neuronales, etc.)
X_train_proc.to_csv(os.path.join(OUTPUT_DIR, "X_train_proc.csv"), index=False, sep=";", decimal=",", encoding="utf-8-sig")
X_test_proc.to_csv(os.path.join(OUTPUT_DIR, "X_test_proc.csv"), index=False, sep=";", decimal=",", encoding="utf-8-sig")

# 3. Guardar datasets con variables categóricas originales (para modelos de árboles tipo LightGBM / CatBoost)
X_train_raw.to_csv(os.path.join(OUTPUT_DIR, "X_train_raw.csv"), index=False, sep=";", decimal=",", encoding="utf-8-sig")
X_test_raw.to_csv(os.path.join(OUTPUT_DIR, "X_test_raw.csv"), index=False, sep=";", decimal=",", encoding="utf-8-sig")

# 4. Guardar target (y)
y_train.to_frame().to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False, sep=";", decimal=",", encoding="utf-8-sig")
y_test.to_frame().to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False, sep=";", decimal=",", encoding="utf-8-sig")

# 5. Guardar metadatos de evaluación (para análisis de subgrupos, fairness y factores de elevación)
meta_train.to_csv(os.path.join(OUTPUT_DIR, "meta_train.csv"), index=False, sep=";", decimal=",", encoding="utf-8-sig")
meta_test.to_csv(os.path.join(OUTPUT_DIR, "meta_test.csv"), index=False, sep=";", decimal=",", encoding="utf-8-sig")

# 6. Guardar resumen de metadatos en JSON
metadata_resumen = {
    'total_filas_original': len(df),
    'train_filas': int(X_train_proc.shape[0]),
    'test_filas': int(X_test_proc.shape[1]),
    'num_features_procesadas': int(X_train_proc.shape[1]),
    'feature_names': feature_names,
    'tasa_empleo_train_pct': float(y_train.mean() * 100),
    'tasa_empleo_test_pct': float(y_test.mean() * 100),
    'vars_numericas': vars_numericas,
    'vars_ordinales': vars_ordinales,
    'vars_nominales': vars_nominales,
    'vars_binarias': vars_binarias
}

with open(os.path.join(OUTPUT_DIR, "features_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata_resumen, f, indent=4, ensure_ascii=False)

print(f"  [+] Datasets particionados exportados a: {OUTPUT_DIR}")
print(f"  [+] Metadatos guardados en: {os.path.join(OUTPUT_DIR, 'features_metadata.json')}")

print("\n" + "=" * 75)
print("¡FASE 2 COMPLETADA CON ÉXITO! DATASETS LISTOS PARA MODELADO (FASE 3).")
print("=" * 75)
