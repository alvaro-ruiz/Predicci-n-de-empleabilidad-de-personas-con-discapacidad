#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFM: Predicción de la Empleabilidad de Personas con Discapacidad
Script / Notebook Interactivo: 04_explicabilidad_shap.py
----------------------------------------------------------------
Fase 4: Explicabilidad (XAI) con SHAP y Auditoría de Equidad (Fairness)
- Carga del mejor modelo entrenado en la Fase 3 ('models/best_model.joblib').
- Cálculo de valores Shapley (SHAP TreeExplainer) sobre el conjunto de test.
- Explicabilidad Global:
  * Ranking cuantitativo de importancia (SHAP Bar Plot).
  * Impacto y dirección de cada factor (SHAP Beeswarm Plot).
  * Gráficos de dependencia e interacciones (Edad, Conectividad TIC, Discapacidad).
- Explicabilidad Local (Estudios de Caso):
  * SHAP Waterfall Plots para perfiles de alta empleabilidad y de alta vulnerabilidad.
  * Análisis de factores palanca (qué variables pueden cambiar la decisión).
- Auditoría Ética y de Equidad (Fairness):
  * Análisis de sesgo de género (Hombre vs Mujer) y edad (Jóvenes vs Séniors).
  * Tasa de selección positiva, Disparate Impact Ratio (Regla de los 4/5) e Igualdad de Oportunidades.
"""

# %% 1. Importación de Librerías y Configuración
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.inspection import permutation_importance

# Importar SHAP con fallback en caso de no estar disponible
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[!] Advertencia: La librería 'shap' no está instalada en el entorno. Se utilizará interpretabilidad basada en árboles y permutation importance.")

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (11, 6)
plt.rcParams['font.size'] = 11

# Rutas del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if "__file__" in locals() else os.getcwd()
TRAIN_TEST_DIR = os.path.join(BASE_DIR, "Data", "processed", "train_test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "models", "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 80)
print("FASE 4: EXPLICABILIDAD (XAI) CON SHAP Y AUDITORÍA DE EQUIDAD (FAIRNESS)")
print("=" * 80)

# %% 2. Carga de Modelo, Metadatos y Conjunto de Test
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "best_model_metadata.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}. Ejecuta primero la Fase 3 (03_modelado_empleabilidad.py).")

print(f"Cargando modelo serializado desde: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print(f"Modelo cargado: {type(model).__name__}")

with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    metadata_modelo = json.load(f)

umbral_optimo = metadata_modelo.get('umbral_optimo', 0.43)
print(f"Umbral óptimo cargado: {umbral_optimo:.2f}")

# Cargar datos de test procesados y metadatos
X_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "X_train_proc.csv"), sep=";", decimal=",", encoding="utf-8-sig")
X_test = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "X_test_proc.csv"), sep=";", decimal=",", encoding="utf-8-sig")
y_test = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "y_test.csv"), sep=";", decimal=",", encoding="utf-8-sig").iloc[:, 0].values
meta_test = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "meta_test.csv"), sep=";", decimal=",", encoding="utf-8-sig")

print(f"Datos de test cargados: {X_test.shape[0]:,} muestras x {X_test.shape[1]} características.")

# %% 3. Cálculo de Valores SHAP (TreeExplainer)
print("\n" + "=" * 80)
print("CÁLCULO DE VALORES SHAPLEY (EXPLICABILIDAD GLOBAL Y LOCAL)")
print("=" * 80)

if HAS_SHAP:
    print("Inicializando shap.TreeExplainer con el modelo de Random Forest...")
    explainer = shap.TreeExplainer(model)
    
    # Calcular valores SHAP para el conjunto de test
    # shap_values puede ser un Explanation object o un array/lista según versión
    shap_exp = explainer(X_test)
    
    # Manejar formato dimensional: si es multiclase (2 clases), extraemos la clase positiva (empleado = 1)
    if len(shap_exp.shape) == 3 and shap_exp.shape[2] == 2:
        shap_values_pos = shap_exp[:, :, 1]
    else:
        shap_values_pos = shap_exp

    base_value = shap_values_pos.base_values[0] if isinstance(shap_values_pos.base_values, np.ndarray) else shap_values_pos.base_values
    print(f"Valores SHAP calculados con éxito para {len(X_test)} individuos.")
    print(f"Valor base global E[f(x)] (log-odds / prob base): {base_value:.4f}")
else:
    shap_values_pos = None
    print("[i] SHAP no disponible; calculando Feature Importance nativa de Scikit-Learn...")

# %% 4. Explicabilidad Global: Ranking y Beeswarm Plot
print("\nGenerando visualizaciones de Explicabilidad Global...")

if HAS_SHAP:
    # 1. SHAP Beeswarm Plot (Muestra impacto y dirección: rojo=alto valor, azul=bajo valor)
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values_pos, max_display=20, show=False)
    plt.title("Impacto y Dirección de las 20 Variables Clave en la Empleabilidad (SHAP Beeswarm)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    beeswarm_path = os.path.join(REPORTS_DIR, "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  [+] Gráfico Beeswarm guardado en: {beeswarm_path}")

    # 2. SHAP Bar Plot (Importancia absoluta promedio)
    plt.figure(figsize=(11, 7))
    shap.plots.bar(shap_values_pos, max_display=20, show=False)
    plt.title("Ranking de Importancia Global de Factores de Empleabilidad (SHAP Absoluto)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    bar_path = os.path.join(REPORTS_DIR, "shap_importance_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  [+] Gráfico de barras SHAP guardado en: {bar_path}")
else:
    # Fallback con Feature Importance nativa
    importancias = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False).head(20)
    plt.figure(figsize=(11, 7))
    sns.barplot(x=importancias.values, y=importancias.index, palette='viridis')
    plt.title("Top 20 Variables más Importantes (Gini Importance del Modelo)", fontsize=13, fontweight='bold')
    plt.xlabel("Importancia Relativa")
    plt.tight_layout()
    bar_path = os.path.join(REPORTS_DIR, "feature_importance_bar.png")
    plt.savefig(bar_path, dpi=300)
    plt.show()

# %% 5. Efectos de Dependencia e Interacciones Clave
print("\nGenerando gráficos de dependencia e interacción de variables críticas...")

if HAS_SHAP:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # 1. Dependencia de Edad
    shap.plots.scatter(shap_values_pos[:, "edad"], color=shap_values_pos[:, "num_limitaciones_graves"], ax=axes[0, 0], show=False)
    axes[0, 0].set_title("Efecto de la Edad en la Empleabilidad (Color = Nº Limitaciones)")
    
    # 2. Dependencia de Conectividad Digital
    shap.plots.scatter(shap_values_pos[:, "indice_conectividad"], color=shap_values_pos[:, "nivel_estudios_agrupado"], ax=axes[0, 1], show=False)
    axes[0, 1].set_title("Efecto del Acceso Digital (Color = Nivel Educativo)")
    
    # 3. Dependencia del Nivel Educativo
    shap.plots.scatter(shap_values_pos[:, "nivel_estudios_agrupado"], color=shap_values_pos[:, "edad"], ax=axes[1, 0], show=False)
    axes[1, 0].set_title("Efecto del Nivel Educativo (Color = Edad)")
    
    # 4. Dependencia del Tramo de Grado de Discapacidad
    shap.plots.scatter(shap_values_pos[:, "tramo_grado_discapacidad"], color=shap_values_pos[:, "pluridiscapacidad_grave"], ax=axes[1, 1], show=False)
    axes[1, 1].set_title("Efecto del Grado Reconocido (Color = Pluridiscapacidad)")
    
    plt.tight_layout()
    dep_path = os.path.join(REPORTS_DIR, "shap_dependence_key_factors.png")
    plt.savefig(dep_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  [+] Gráficos de dependencia guardados en: {dep_path}")

# %% 6. Explicabilidad Local (Estudios de Caso Individuales)
print("\n" + "=" * 80)
print("EXPLICABILIDAD LOCAL: ANÁLISIS DE PERFILES INDIVIDUALES (WATERFALL PLOTS)")
print("=" * 80)

# Obtener predicciones y probabilidades sobre test
probas_test = model.predict_proba(X_test)[:, 1]

# Identificar perfiles representativos en test
# Caso 1: Alta empleabilidad predicha (alta confianza positiva) y efectivamente empleado
candidatos_exito = np.where((probas_test >= 0.70) & (y_test == 1))[0]
idx_exito = candidatos_exito[0] if len(candidatos_exito) > 0 else np.argmax(probas_test)

# Caso 2: Altas barreras predichas (baja probabilidad) y no empleado
candidatos_barreras = np.where((probas_test <= 0.15) & (y_test == 0))[0]
idx_barreras = candidatos_barreras[0] if len(candidatos_barreras) > 0 else np.argmin(probas_test)

# Caso 3: Caso Límite / Umbral (probabilidad cercana al umbral óptimo 0.43)
candidatos_limite = np.where((probas_test >= 0.38) & (probas_test <= 0.48))[0]
idx_limite = candidatos_limite[0] if len(candidatos_limite) > 0 else 0

casos = [
    (idx_exito, "caso_exito", f"Perfil con Alta Probabilidad de Empleo (p = {probas_test[idx_exito]:.2f})"),
    (idx_barreras, "caso_barreras", f"Perfil con Barreras Estructurales Severas (p = {probas_test[idx_barreras]:.2f})"),
    (idx_limite, "caso_limite", f"Perfil en Zona Límite / Potencial Activación (p = {probas_test[idx_limite]:.2f})")
]

if HAS_SHAP:
    for idx_caso, nombre_archivo, titulo in casos:
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values_pos[idx_caso], max_display=12, show=False)
        plt.title(f"Explicación SHAP: {titulo}", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        wf_path = os.path.join(REPORTS_DIR, f"shap_waterfall_{nombre_archivo}.png")
        plt.savefig(wf_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"  [+] Waterfall plot generado ({nombre_archivo}): {wf_path}")

# %% 7. Auditoría Ética y de Equidad (Fairness Audit)
print("\n" + "=" * 80)
print("AUDITORÍA DE EQUIDAD (FAIRNESS AUDIT) POR GÉNERO Y GRUPO DE EDAD")
print("=" * 80)

# Predicciones binarias con el umbral óptimo identificado en la Fase 3
y_pred_opt = (probas_test >= umbral_optimo).astype(int)

# Crear DataFrame de auditoría combinando predicciones y variables demográficas
df_audit = X_test.copy()
df_audit['y_real'] = y_test
df_audit['y_pred'] = y_pred_opt
df_audit['probabilidad'] = probas_test

# 1. Equidad por Sexo (sexo_Mujer = 1 vs 0)
es_mujer = (df_audit['sexo_Mujer'] == 1)

sel_rate_hombres = df_audit[~es_mujer]['y_pred'].mean()
sel_rate_mujeres = df_audit[es_mujer]['y_pred'].mean()
disparate_impact_sexo = sel_rate_mujeres / sel_rate_hombres if sel_rate_hombres > 0 else 1.0

recall_hombres = recall_score(df_audit[~es_mujer]['y_real'], df_audit[~es_mujer]['y_pred'])
recall_mujeres = recall_score(df_audit[es_mujer]['y_real'], df_audit[es_mujer]['y_pred'])

tn_h, fp_h, fn_h, tp_h = confusion_matrix(df_audit[~es_mujer]['y_real'], df_audit[~es_mujer]['y_pred']).ravel()
tn_m, fp_m, fn_m, tp_m = confusion_matrix(df_audit[es_mujer]['y_real'], df_audit[es_mujer]['y_pred']).ravel()

fpr_hombres = fp_h / (fp_h + tn_h)
fpr_mujeres = fp_m / (fp_m + tn_m)

# 2. Equidad por Grupo de Edad (Séniors >= 50 años vs Jóvenes/Adultos < 50)
# 'edad' está normalizada (StandardScaler), por lo que usamos edad_senior_con_limitacion o la mediana
es_senior = (df_audit['edad_senior_con_limitacion'] == 1) | (df_audit['edad'] >= df_audit['edad'].median())

sel_rate_joven = df_audit[~es_senior]['y_pred'].mean()
sel_rate_senior = df_audit[es_senior]['y_pred'].mean()
disparate_impact_edad = sel_rate_senior / sel_rate_joven if sel_rate_joven > 0 else 1.0

recall_joven = recall_score(df_audit[~es_senior]['y_real'], df_audit[~es_senior]['y_pred'])
recall_senior = recall_score(df_audit[es_senior]['y_real'], df_audit[es_senior]['y_pred'])

print("ANÁLISIS DE EQUIDAD POR GÉNERO (Hombre vs Mujer):")
print(f"  - Tasa de Selección (Hombres): {sel_rate_hombres*100:.2f}%")
print(f"  - Tasa de Selección (Mujeres): {sel_rate_mujeres*100:.2f}%")
print(f"  - Disparate Impact Ratio (Mujer / Hombre): {disparate_impact_sexo:.3f} "
      f"({'Cumple regla 80%' if disparate_impact_sexo >= 0.80 else 'Alerta: Posible sesgo adverso'})")
print(f"  - Igualdad de Oportunidades (Recall Hombres): {recall_hombres*100:.2f}%")
print(f"  - Igualdad de Oportunidades (Recall Mujeres): {recall_mujeres*100:.2f}%")
print(f"  - Brecha de Sensibilidad (Recall): {abs(recall_hombres - recall_mujeres)*100:.2f} p.p.")

print("\nANÁLISIS DE EQUIDAD POR EDAD (Jóvenes/Adultos vs Séniors):")
print(f"  - Tasa de Selección (<50 años): {sel_rate_joven*100:.2f}%")
print(f"  - Tasa de Selección (>=50 años): {sel_rate_senior*100:.2f}%")
print(f"  - Disparate Impact Ratio (Sénior / Menor): {disparate_impact_edad:.3f}")
print(f"  - Recall en <50 años: {recall_joven*100:.2f}%")
print(f"  - Recall en >=50 años: {recall_senior*100:.2f}%")

# Gráfico de Equidad
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Paridad de Género
metrics_sexo = pd.DataFrame({
    'Métrica': ['Tasa Selección', 'Recall (Oportunidad)', 'FPR (Falsos Positivos)'],
    'Hombres': [sel_rate_hombres, recall_hombres, fpr_hombres],
    'Mujeres': [sel_rate_mujeres, recall_mujeres, fpr_mujeres]
}).melt(id_vars='Métrica', var_name='Sexo', value_name='Valor')

sns.barplot(data=metrics_sexo, x='Métrica', y='Valor', hue='Sexo', palette=['#3498db', '#e84393'], ax=axes[0])
axes[0].set_title('Métricas de Equidad por Género')
axes[0].set_ylim(0, 1.0)
for p in axes[0].patches:
    h = p.get_height()
    if h > 0:
        axes[0].annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2., h / 2),
                         ha='center', va='center', color='white', fontweight='bold')

# Paridad de Edad
metrics_edad = pd.DataFrame({
    'Métrica': ['Tasa Selección', 'Recall (Oportunidad)'],
    'Menores de 50': [sel_rate_joven, recall_joven],
    'Séniors (>=50)': [sel_rate_senior, recall_senior]
}).melt(id_vars='Métrica', var_name='Grupo Edad', value_name='Valor')

sns.barplot(data=metrics_edad, x='Métrica', y='Valor', hue='Grupo Edad', palette=['#2ecc71', '#e67e22'], ax=axes[1])
axes[1].set_title('Métricas de Equidad por Grupo de Edad')
axes[1].set_ylim(0, 1.0)
for p in axes[1].patches:
    h = p.get_height()
    if h > 0:
        axes[1].annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2., h / 2),
                         ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
fairness_plot_path = os.path.join(REPORTS_DIR, "fairness_metrics_plot.png")
plt.savefig(fairness_plot_path, dpi=300)
plt.show()
print(f"  [+] Gráfico de auditoría de equidad guardado en: {fairness_plot_path}")

# Guardar informe JSON de auditoría de equidad
fairness_report = {
    'umbral_evaluado': float(umbral_optimo),
    'genero': {
        'tasa_seleccion_hombres': float(sel_rate_hombres),
        'tasa_seleccion_mujeres': float(sel_rate_mujeres),
        'disparate_impact_ratio': float(disparate_impact_sexo),
        'cumple_regla_cuatro_quintos': bool(disparate_impact_sexo >= 0.80),
        'recall_hombres': float(recall_hombres),
        'recall_mujeres': float(recall_mujeres),
        'fpr_hombres': float(fpr_hombres),
        'fpr_mujeres': float(fpr_mujeres)
    },
    'edad': {
        'tasa_seleccion_menores_50': float(sel_rate_joven),
        'tasa_seleccion_seniors': float(sel_rate_senior),
        'disparate_impact_ratio': float(disparate_impact_edad),
        'recall_menores_50': float(recall_joven),
        'recall_seniors': float(recall_senior)
    }
}

fairness_json_path = os.path.join(REPORTS_DIR, "fairness_audit_report.json")
with open(fairness_json_path, 'w', encoding='utf-8') as f:
    json.dump(fairness_report, f, indent=4, ensure_ascii=False)
print(f"  [+] Informe JSON de auditoría de equidad guardado en: {fairness_json_path}")

print("\n" + "=" * 80)
print("¡FASE 4 COMPLETADA CON ÉXITO! EXPLICABILIDAD Y EQUIDAD AUDITADAS.")
print("=" * 80)
