#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFM: Predicción de la Empleabilidad de Personas con Discapacidad
Script / Notebook Interactivo: 01_eda_empleabilidad.py
----------------------------------------------------------------
Este script contiene el Análisis Exploratorio de Datos (EDA) completo.
Puede ejecutarse celda a celda en VS Code (modo interactivo) o directamente
desde terminal con: python notebooks/01_eda_empleabilidad.py
"""

# %% [markdown]
# # TFM: Predicción de la Empleabilidad de Personas con Discapacidad
# ## Notebook 01: Análisis Exploratorio de Datos (EDA)
#
# **Autor:** Álvaro Ruiz  
# **Fuente de Datos:** Microdatos de la Encuesta EDAD 2020 (INE).  
# **Objetivo:** Explorar las características de la población con discapacidad en edad laboral,
# analizar la tasa de empleo y examinar los factores explicativos clave.

# %% 1. Configuración del Entorno y Carga de Datos
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

# Rutas adaptables según el directorio de ejecución
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if "__file__" in locals() else os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "Data", "processed", "dataset_empleabilidad_edad2020.csv")

if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join("Data", "processed", "dataset_empleabilidad_edad2020.csv")

print(f"Cargando dataset desde: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH, sep=";", decimal=",", encoding="utf-8-sig")
print(f"Dataset cargado con éxito: {df.shape[0]:,} filas y {df.shape[1]} columnas.\n")

# %% 2. Inspección Inicial del Dataset
print("Primeros 5 registros:")
print(df.head())
print("\nInformación del Dataset:")
df.info()

print("\nComprobación de valores nulos:")
nulos = df.isnull().sum()
nulos_pct = (nulos / len(df)) * 100
df_nulos = pd.DataFrame({'Valores Nulos': nulos, 'Porcentaje (%)': nulos_pct})
print(df_nulos[df_nulos['Valores Nulos'] > 0])

# %% 3. Análisis de la Variable Objetivo (empleado)
conteo_target = df['empleado'].value_counts()
pct_target = df['empleado'].value_counts(normalize=True) * 100

total_pond = df['factor_elevacion'].sum()
empleados_pond = df[df['empleado'] == 1]['factor_elevacion'].sum()
tasa_empleo_pond = (empleados_pond / total_pond) * 100

print("=" * 65)
print("DISTRIBUCIÓN DE LA VARIABLE OBJETIVO ('empleado')")
print("=" * 65)
print(f"  - No Empleados (0): {conteo_target[0]:,} ({pct_target[0]:.2f}% muestral)")
print(f"  - Empleados    (1): {conteo_target[1]:,} ({pct_target[1]:.2f}% muestral)")
print(f"  - Tasa de empleo ponderada a nivel poblacional: {tasa_empleo_pond:.2f}%")
print("=" * 65)

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Target Binario
colors_target = ['#e74c3c', '#2ecc71']
sns.barplot(x=pct_target.index.map({0: 'No Empleado (0)', 1: 'Empleado (1)'}),
            y=pct_target.values, palette=colors_target, ax=axes[0])
axes[0].set_title('Distribución de la Variable Objetivo (empleado)')
axes[0].set_ylabel('Porcentaje (%)')
for i, v in enumerate(pct_target.values):
    axes[0].text(i, v + 1, f"{v:.1f}%\n({conteo_target[i]:,} pers.)", ha='center', fontweight='bold')
axes[0].set_ylim(0, 100)

# Situación Laboral
sit_counts = df['situacion_laboral'].value_counts(normalize=True) * 100
sns.barplot(x=sit_counts.index, y=sit_counts.values, palette='Blues_r', ax=axes[1])
axes[1].set_title('Desglose por Situación Laboral')
axes[1].set_ylabel('Porcentaje (%)')
for i, v in enumerate(sit_counts.values):
    axes[1].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
axes[1].set_ylim(0, 100)

plt.tight_layout()
plt.show()

# %% 4. Educación y Empleabilidad
orden_estudios = [
    'Sin estudios / Primaria',
    'Secundaria / Bachillerato',
    'Formación Profesional',
    'Universidad / Posgrado'
]

resumen_estudios = df.groupby('nivel_estudios_agrupado').agg(
    total_personas=('empleado', 'count'),
    personas_empleadas=('empleado', 'sum'),
    tasa_empleo_pct=('empleado', lambda x: x.mean() * 100)
).reindex(orden_estudios)

print("\nTasa de Empleo por Nivel de Estudios:")
print(resumen_estudios)

plt.figure(figsize=(10, 5))
ax = sns.barplot(x=resumen_estudios.index, y=resumen_estudios['tasa_empleo_pct'], palette='viridis')
plt.title('Tasa de Empleo según Nivel Educativo')
plt.ylabel('Tasa de Empleo (%)')
plt.xlabel('Nivel de Estudios')
plt.ylim(0, max(resumen_estudios['tasa_empleo_pct']) + 15)

for i, v in enumerate(resumen_estudios['tasa_empleo_pct']):
    tot = resumen_estudios['total_personas'].iloc[i]
    ax.text(i, v + 1.5, f"{v:.1f}%\n(N={tot})", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# %% 5. Tipos de Limitaciones Funcionales y Severidad
limitaciones = {
    'lim_vision': 'Visual',
    'lim_audicion': 'Auditiva',
    'lim_comunicacion': 'Comunicación',
    'lim_aprendizaje': 'Aprendizaje/Cognitiva',
    'lim_movilidad': 'Movilidad',
    'lim_autocuidado': 'Autocuidado'
}

datos_lim = []
for col, nombre in limitaciones.items():
    tasa_con_lim = df[df[col] == 1]['empleado'].mean() * 100
    n_con_lim = (df[col] == 1).sum()
    tasa_sin_lim = df[df[col] == 0]['empleado'].mean() * 100
    datos_lim.append({
        'Tipo de Limitación': nombre,
        'Tasa Empleo con Limitación (%)': tasa_con_lim,
        'N Personas': n_con_lim,
        'Tasa Empleo sin Limitación (%)': tasa_sin_lim,
        'Brecha (p.p.)': tasa_sin_lim - tasa_con_lim
    })

df_lim_comp = pd.DataFrame(datos_lim).sort_values(by='Tasa Empleo con Limitación (%)', ascending=True)
print("\nTasa de Empleo por Tipo de Limitación Funcional Grave:")
print(df_lim_comp)

plt.figure(figsize=(11, 5))
sns.barplot(data=df_lim_comp, x='Tasa Empleo con Limitación (%)', y='Tipo de Limitación', palette='Reds_r')
plt.title('Tasa de Empleo según Tipo de Limitación Funcional Grave')
plt.xlabel('Tasa de Empleo (%)')
plt.xlim(0, 45)

for i, v in enumerate(df_lim_comp['Tasa Empleo con Limitación (%)']):
    n = df_lim_comp['N Personas'].iloc[i]
    plt.text(v + 0.8, i, f"{v:.1f}% (N={n})", va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Multidiversidad funcional
tasa_num_lim = df.groupby('num_limitaciones_graves').agg(
    total=('empleado', 'count'),
    tasa_empleo_pct=('empleado', lambda x: x.mean() * 100)
).reset_index()

plt.figure(figsize=(9, 5))
sns.lineplot(data=tasa_num_lim, x='num_limitaciones_graves', y='tasa_empleo_pct', marker='o', color='#c0392b', linewidth=2.5, markersize=8)
plt.title('Efecto del Número de Limitaciones Graves Acumuladas en la Empleabilidad')
plt.xlabel('Número de Limitaciones Graves')
plt.ylabel('Tasa de Empleo (%)')
plt.ylim(0, max(tasa_num_lim['tasa_empleo_pct']) + 10)

for _, row in tasa_num_lim.iterrows():
    plt.text(row['num_limitaciones_graves'], row['tasa_empleo_pct'] + 1.5, f"{row['tasa_empleo_pct']:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# %% 6. Factores Demográficos y Territoriales
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sexo
tasa_sexo = df.groupby('sexo')['empleado'].agg(['count', lambda x: x.mean() * 100]).rename(columns={'<lambda_0>': 'tasa_pct'})
sns.barplot(x=tasa_sexo.index, y=tasa_sexo['tasa_pct'], palette=['#3498db', '#e84393'], ax=axes[0])
axes[0].set_title('Tasa de Empleo por Sexo (Brecha de Género)')
axes[0].set_ylabel('Tasa de Empleo (%)')
axes[0].set_ylim(0, 45)
for i, v in enumerate(tasa_sexo['tasa_pct']):
    axes[0].text(i, v + 1, f"{v:.1f}%\n(N={tasa_sexo['count'].iloc[i]})", ha='center', fontweight='bold')

# Edad
tasa_edad = df.groupby('tramo_edad')['empleado'].agg(['count', lambda x: x.mean() * 100]).rename(columns={'<lambda_0>': 'tasa_pct'})
sns.barplot(x=tasa_edad.index, y=tasa_edad['tasa_pct'], palette='Blues_r', ax=axes[1])
axes[1].set_title('Tasa de Empleo por Tramo de Edad')
axes[1].set_ylabel('Tasa de Empleo (%)')
axes[1].set_ylim(0, 45)
axes[1].tick_params(axis='x', rotation=25)
for i, v in enumerate(tasa_edad['tasa_pct']):
    axes[1].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Comunidad Autónoma
tasa_ccaa = df.groupby('ccaa')['empleado'].agg(
    total=('count'),
    tasa_pct=lambda x: x.mean() * 100
).sort_values(by='tasa_pct', ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(x=tasa_ccaa['tasa_pct'], y=tasa_ccaa.index, palette='crest')
media_nacional = df['empleado'].mean() * 100
plt.axvline(media_nacional, color='red', linestyle='--', label=f'Media Muestral ({media_nacional:.1f}%)')
plt.title('Tasa de Empleo por Comunidad Autónoma')
plt.xlabel('Tasa de Empleo (%)')
plt.legend()

for i, v in enumerate(tasa_ccaa['tasa_pct']):
    plt.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=9)

plt.tight_layout()
plt.show()

# %% 7. Entorno del Hogar y Brecha Digital
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Internet
tasa_net = df.groupby('tiene_internet')['empleado'].mean() * 100
sns.barplot(x=['Sin Internet (0)', 'Con Internet (1)'], y=tasa_net.values, palette=['#e74c3c', '#27ae60'], ax=axes[0])
axes[0].set_title('Tasa de Empleo según Acceso a Internet')
axes[0].set_ylabel('Tasa de Empleo (%)')
axes[0].set_ylim(0, 45)
for i, v in enumerate(tasa_net.values):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

# Ordenador
tasa_pc = df.groupby('tiene_ordenador')['empleado'].mean() * 100
sns.barplot(x=['Sin Ordenador (0)', 'Con Ordenador (1)'], y=tasa_pc.values, palette=['#e74c3c', '#2980b9'], ax=axes[1])
axes[1].set_title('Tasa de Empleo según Disponibilidad de Ordenador')
axes[1].set_ylabel('Tasa de Empleo (%)')
axes[1].set_ylim(0, 45)
for i, v in enumerate(tasa_pc.values):
    axes[1].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# %% 8. Matriz de Correlaciones con Empleo
cols_corr = [
    'empleado', 'edad', 'tamano_hogar',
    'lim_vision', 'lim_audicion', 'lim_comunicacion',
    'lim_aprendizaje', 'lim_movilidad', 'lim_autocuidado',
    'num_limitaciones_graves',
    'tiene_internet', 'tiene_ordenador'
]

corr_matrix = df[cols_corr].corr()

plt.figure(figsize=(11, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, vmin=-0.4, vmax=0.4, cbar_kws={'label': 'Coeficiente de Correlación'})
plt.title('Matriz de Correlación de Variables Clave con Empleo')
plt.tight_layout()
plt.show()

ranking_corr = corr_matrix['empleado'].drop('empleado').sort_values(ascending=False)
print("=" * 65)
print("RANKING DE CORRELACIÓN CON LA EMPLEABILIDAD ('empleado'):")
print("=" * 65)
for var, val in ranking_corr.items():
    sentido = "(+) Favorece empleo" if val > 0 else "(-) Frena empleo"
    print(f"  {var:25s}: {val:+.4f}  {sentido}")
print("=" * 65)

print("\n¡EDA completado con éxito! Listo para la fase de Feature Engineering y Modelado.")
