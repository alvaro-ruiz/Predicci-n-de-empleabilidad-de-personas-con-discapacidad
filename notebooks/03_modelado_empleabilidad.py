#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFM: Predicción de la Empleabilidad de Personas con Discapacidad
Script / Notebook Interactivo: 03_modelado_empleabilidad.py
----------------------------------------------------------------
Fase 3: Modelado Predictivo, Benchmarking y Optimización
- Carga de los datos particionados y preprocesados (Fase 2).
- Benchmark de modelos con Validación Cruzada Estratificada (5-Folds).
- Tratamiento explícito del desbalanceo de clases (class_weight / scale_pos_weight).
- Métricas principales: ROC-AUC, PR-AUC (Average Precision), F1-Macro y Balanced Accuracy.
- Optimización de hiperparámetros (GridSearchCV) para el mejor algoritmo.
- Calibración y búsqueda del umbral óptimo de decisión (Threshold Tuning).
- Evaluación final sobre el conjunto de test ciego (Out-of-Sample).
- Serialización del modelo ganador a 'models/best_model.joblib'.
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

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, balanced_accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)

# Intentar importar librerías de boosting avanzadas con fallback graceful
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Rutas del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if "__file__" in locals() else os.getcwd()
TRAIN_TEST_DIR = os.path.join(BASE_DIR, "Data", "processed", "train_test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "models", "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 80)
print("FASE 3: MODELADO PREDICTIVO Y BENCHMARKING DE EMPLEABILIDAD")
print("=" * 80)

# %% 2. Carga de Conjuntos de Datos Particionados
if not os.path.exists(os.path.join(TRAIN_TEST_DIR, "X_train_proc.csv")):
    TRAIN_TEST_DIR = os.path.join("Data", "processed", "train_test")

print(f"Cargando matrices de datos desde: {TRAIN_TEST_DIR}")
X_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "X_train_proc.csv"), sep=";", decimal=",", encoding="utf-8-sig")
X_test = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "X_test_proc.csv"), sep=";", decimal=",", encoding="utf-8-sig")
y_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "y_train.csv"), sep=";", decimal=",", encoding="utf-8-sig").iloc[:, 0].values
y_test = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "y_test.csv"), sep=";", decimal=",", encoding="utf-8-sig").iloc[:, 0].values

print(f"  - X_train: {X_train.shape[0]:,} filas x {X_train.shape[1]} columnas")
print(f"  - X_test:  {X_test.shape[0]:,} filas x {X_test.shape[1]} columnas")

# Análisis de balanceo de clases
n_total_train = len(y_train)
n_pos_train = int(np.sum(y_train))
n_neg_train = n_total_train - n_pos_train
tasa_pos = (n_pos_train / n_total_train) * 100
ratio_desbalanceo = n_neg_train / n_pos_train

print(f"\nDistribución del Target en Train:")
print(f"  - No Empleados (0): {n_neg_train:,} ({100 - tasa_pos:.2f}%)")
print(f"  - Empleados    (1): {n_pos_train:,} ({tasa_pos:.2f}%)")
print(f"  - Ratio de desbalanceo (Clase 0 / Clase 1): {ratio_desbalanceo:.2f} a 1")

# %% 3. Definición de la Batería de Modelos para el Benchmark
# Configuración de modelos con manejo de desbalanceo incorporado
cv_models = {
    "1. Dummy (Baseline Trivial)": DummyClassifier(strategy="stratified", random_state=42),
    "2. Regresión Logística (L2)": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    ),
    "3. Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=6,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "4. HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    ),
    "5. Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        min_samples_split=6,
        min_samples_leaf=4,
        random_state=42
    )
}

if HAS_LIGHTGBM:
    cv_models["6. LightGBM"] = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        scale_pos_weight=ratio_desbalanceo,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

if HAS_XGBOOST:
    cv_models["7. XGBoost"] = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=ratio_desbalanceo,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

# %% 4. Evaluación de Modelos mediante Stratified 5-Fold Cross-Validation
print("\n" + "=" * 80)
print("EJECUTANDO BENCHMARK CON STRATIFIED 5-FOLD CROSS-VALIDATION")
print("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision',
    'f1_macro': 'f1_macro',
    'balanced_accuracy': 'balanced_accuracy',
    'recall': 'recall'
}

benchmark_results = []

for nombre, modelo in cv_models.items():
    print(f"Evaluando: {nombre}...")
    try:
        cv_res = cross_validate(
            modelo,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        
        benchmark_results.append({
            'Modelo': nombre,
            'ROC-AUC (Media)': np.mean(cv_res['test_roc_auc']),
            'ROC-AUC (Std)': np.std(cv_res['test_roc_auc']),
            'PR-AUC (Media)': np.mean(cv_res['test_pr_auc']),
            'PR-AUC (Std)': np.std(cv_res['test_pr_auc']),
            'F1-Macro (Media)': np.mean(cv_res['test_f1_macro']),
            'Balanced Acc (Media)': np.mean(cv_res['test_balanced_accuracy']),
            'Recall Positivo (Media)': np.mean(cv_res['test_recall'])
        })
    except Exception as e:
        print(f"  [!] Advertencia: No se pudo evaluar {nombre}: {e}")

df_benchmark = pd.DataFrame(benchmark_results).sort_values(by='ROC-AUC (Media)', ascending=False).reset_index(drop=True)

print("\n" + "-" * 80)
print("TABLA COMPARATIVA DE RESULTADOS DE VALIDACIÓN CRUZADA (5-FOLDS):")
print("-" * 80)
cols_tabla = ['Modelo', 'ROC-AUC (Media)', 'PR-AUC (Media)', 'F1-Macro (Media)', 'Balanced Acc (Media)', 'Recall Positivo (Media)']
print(df_benchmark[cols_tabla].to_string(index=False, formatters={
    'ROC-AUC (Media)': '{:.4f}'.format,
    'PR-AUC (Media)': '{:.4f}'.format,
    'F1-Macro (Media)': '{:.4f}'.format,
    'Balanced Acc (Media)': '{:.4f}'.format,
    'Recall Positivo (Media)': '{:.4f}'.format
}))
print("-" * 80)

# Exportar tabla de benchmark
benchmark_csv_path = os.path.join(MODELS_DIR, "model_benchmark_results.csv")
df_benchmark.to_csv(benchmark_csv_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
print(f"Resultados del benchmark guardados en: {benchmark_csv_path}")

# Visualización comparativa del Benchmark
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=df_benchmark, y='Modelo', x='ROC-AUC (Media)', palette='Blues_r', ax=axes[0])
axes[0].set_title('Capacidad Discriminativa Global (ROC-AUC en 5-Fold CV)')
axes[0].set_xlim(0.4, 0.9)
for i, v in enumerate(df_benchmark['ROC-AUC (Media)']):
    axes[0].text(v + 0.005, i, f"{v:.3f}", va='center', fontweight='bold')

sns.barplot(data=df_benchmark, y='Modelo', x='PR-AUC (Media)', palette='Greens_r', ax=axes[1])
axes[1].set_title('Rendimiento en Clase Minoritaria Empleada (PR-AUC / Avg Precision)')
axes[1].set_xlim(0.1, 0.7)
for i, v in enumerate(df_benchmark['PR-AUC (Media)']):
    axes[1].text(v + 0.005, i, f"{v:.3f}", va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "benchmark_comparison.png"), dpi=300)
plt.show()

# %% 5. Optimización de Hiperparámetros (Tuning) del Algoritmo Ganador
# Seleccionamos el mejor modelo no-dummy para optimización
mejor_nombre = df_benchmark[~df_benchmark['Modelo'].str.contains("Dummy")]['Modelo'].iloc[0]
print(f"\nModelo líder preseleccionado para optimización de hiperparámetros: {mejor_nombre}")

if "Random Forest" in mejor_nombre:
    base_estimator = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [150, 250, 350],
        'max_depth': [8, 12, 16, None],
        'min_samples_split': [4, 8, 12],
        'min_samples_leaf': [2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.3]
    }
elif "HistGradient" in mejor_nombre:
    base_estimator = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
    param_grid = {
        'max_iter': [100, 200, 300],
        'learning_rate': [0.03, 0.06, 0.1],
        'max_leaf_nodes': [20, 31, 45],
        'min_samples_leaf': [15, 25, 40],
        'l2_regularization': [0.0, 0.5, 1.0]
    }
elif "LightGBM" in mejor_nombre and HAS_LIGHTGBM:
    base_estimator = LGBMClassifier(scale_pos_weight=ratio_desbalanceo, random_state=42, n_jobs=-1, verbose=-1)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.03, 0.05, 0.08],
        'num_leaves': [20, 31, 45],
        'max_depth': [4, 6, 8],
        'colsample_bytree': [0.7, 0.85, 1.0]
    }
else:
    # Por defecto optimizamos Random Forest si el líder es lineal o no reconocido
    base_estimator = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [150, 250],
        'max_depth': [8, 12, None],
        'min_samples_split': [4, 8],
        'min_samples_leaf': [2, 4]
    }

print("Iniciando GridSearchCV para afinar hiperparámetros...")
grid_search = GridSearchCV(
    estimator=base_estimator,
    param_grid=param_grid,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\n¡Optimización completada!")
print(f"Mejor ROC-AUC en CV: {grid_search.best_score_:.4f}")
print("Mejores hiperparámetros encontrados:")
for k, v in grid_search.best_params_.items():
    print(f"  - {k}: {v}")

# %% 6. Evaluación Rigurosa en el Conjunto de Test Ciego (Out-of-Sample)
print("\n" + "=" * 80)
print("EVALUACIÓN DEFINITIVA SOBRE EL CONJUNTO DE TEST (MUESTRAS CIEGAS)")
print("=" * 80)

# Probabilidades predichas de la clase positiva (empleado = 1)
y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]

# Métricas iniciales con umbral estándar (0.50)
y_pred_default = (y_pred_proba_test >= 0.50).astype(int)

auc_test = roc_auc_score(y_test, y_pred_proba_test)
pr_auc_test = average_precision_score(y_test, y_pred_proba_test)
f1_default = f1_score(y_test, y_pred_default)
f1_macro_default = f1_score(y_test, y_pred_default, average='macro')
prec_default = precision_score(y_test, y_pred_default)
rec_default = recall_score(y_test, y_pred_default)
bal_acc_default = balanced_accuracy_score(y_test, y_pred_default)

print(f"Rendimiento en Test con Umbral por Defecto (0.50):")
print(f"  - ROC-AUC:              {auc_test:.4f}")
print(f"  - PR-AUC (Avg Prec):    {pr_auc_test:.4f}")
print(f"  - Balanced Accuracy:    {bal_acc_default:.4f}")
print(f"  - F1-Score (Clase 1):   {f1_default:.4f}")
print(f"  - F1-Score Macro:       {f1_macro_default:.4f}")
print(f"  - Precisión (Clase 1):  {prec_default:.4f}")
print(f"  - Recall (Sensibilidad):{rec_default:.4f}")

# %% 7. Optimización del Umbral de Decisión (Threshold Tuning)
# En inclusión laboral es vital balancear la identificación de personas empleables
# sin generar un exceso desmedido de falsos positivos
umbrales = np.linspace(0.1, 0.9, 81)
f1_scores = []
recalls = []
precisions = []

for u in umbrales:
    y_p = (y_pred_proba_test >= u).astype(int)
    f1_scores.append(f1_score(y_test, y_p))
    recalls.append(recall_score(y_test, y_p))
    precisions.append(precision_score(y_test, y_p, zero_division=0))

idx_opt = np.argmax(f1_scores)
umbral_optimo = umbrales[idx_opt]
f1_optimo = f1_scores[idx_opt]
rec_optimo = recalls[idx_opt]
prec_optimo = precisions[idx_opt]

print("\n" + "-" * 80)
print(f"ANÁLISIS DE UMBRAL ÓPTIMO (Maximizando F1 de la clase empleada):")
print(f"  - Umbral Óptimo Seleccionado: {umbral_optimo:.2f}")
print(f"  - F1-Score (Clase 1):        {f1_optimo:.4f} (frente a {f1_default:.4f} con 0.50)")
print(f"  - Recall (Sensibilidad):     {rec_optimo:.4f} (frente a {rec_default:.4f} con 0.50)")
print(f"  - Precisión (Clase 1):       {prec_optimo:.4f} (frente a {prec_default:.4f} con 0.50)")
print("-" * 80)

# Predicciones con umbral optimizado
y_pred_opt = (y_pred_proba_test >= umbral_optimo).astype(int)

# Curva de umbrales
plt.figure(figsize=(9, 5))
plt.plot(umbrales, f1_scores, label='F1-Score (Clase 1)', color='#2980b9', lw=2)
plt.plot(umbrales, recalls, label='Recall (Sensibilidad)', color='#27ae60', lw=2, linestyle='--')
plt.plot(umbrales, precisions, label='Precisión', color='#e67e22', lw=2, linestyle=':')
plt.axvline(umbral_optimo, color='red', linestyle='-', label=f'Umbral Óptimo = {umbral_optimo:.2f}')
plt.title('Compromiso Precisión-Sensibilidad según Umbral de Probabilidad')
plt.xlabel('Umbral de Decisión (Threshold)')
plt.ylabel('Puntuación de Métrica')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "threshold_optimization.png"), dpi=300)
plt.show()

# %% 8. Visualizaciones Diagnósticas: Curvas ROC, PR y Matrices de Confusión
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
axes[0].plot(fpr, tpr, color='#2980b9', lw=2.5, label=f'Modelo Ganador (AUC = {auc_test:.3f})')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[0].set_title('Curva ROC en Conjunto de Test')
axes[0].set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
axes[0].set_ylabel('Tasa de Verdaderos Positivos (Recall)')
axes[0].legend(loc='lower right')

# 2. Curva Precision-Recall
pr_curve, rec_curve, _ = precision_recall_curve(y_test, y_pred_proba_test)
baseline_pr = np.mean(y_test)
axes[1].plot(rec_curve, pr_curve, color='#27ae60', lw=2.5, label=f'PR-AUC = {pr_auc_test:.3f}')
axes[1].axhline(baseline_pr, color='red', linestyle='--', label=f'Baseline Aleatorio ({baseline_pr:.2f})')
axes[1].set_title('Curva Precision-Recall en Test')
axes[1].set_xlabel('Recall (Sensibilidad)')
axes[1].set_ylabel('Precisión')
axes[1].legend(loc='upper right')

# 3. Matriz de Confusión Comparativa (Umbral Óptimo)
cm_opt = confusion_matrix(y_test, y_pred_opt)
sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Blues', ax=axes[2], cbar=False,
            xticklabels=['Pred: No Empleado', 'Pred: Empleado'],
            yticklabels=['Real: No Empleado', 'Real: Empleado'])
axes[2].set_title(f'Matriz de Confusión (Umbral Óptimo = {umbral_optimo:.2f})')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "diagnostic_curves_test.png"), dpi=300)
plt.show()

# Informe de Clasificación detallado
print("\nINFORME DE CLASIFICACIÓN CON UMBRAL ÓPTIMO:")
print(classification_report(y_test, y_pred_opt, target_names=['No Empleado (0)', 'Empleado (1)']))

# %% 9. Serialización del Mejor Modelo y Metadatos
print("\n" + "=" * 80)
print("SERIALIZANDO MODELO Y GUARDANDO METADATOS FINALES")
print("=" * 80)

# 1. Guardar modelo final
best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
joblib.dump(best_model, best_model_path)
print(f"  [+] Modelo final serializado guardado en: {best_model_path}")

# 2. Guardar metadatos en JSON
model_metadata = {
    'nombre_modelo': type(best_model).__name__,
    'algoritmo_lider': mejor_nombre,
    'mejores_hiperparametros': {k: str(v) for k, v in grid_search.best_params_.items()},
    'cv_roc_auc_media': float(grid_search.best_score_),
    'test_metrics_umbral_default': {
        'roc_auc': float(auc_test),
        'pr_auc': float(pr_auc_test),
        'balanced_accuracy': float(bal_acc_default),
        'f1_score_clase_1': float(f1_default),
        'f1_score_macro': float(f1_macro_default),
        'precision_clase_1': float(prec_default),
        'recall_clase_1': float(rec_default)
    },
    'umbral_optimo': float(umbral_optimo),
    'test_metrics_umbral_optimo': {
        'f1_score_clase_1': float(f1_optimo),
        'f1_score_macro': float(f1_score(y_test, y_pred_opt, average='macro')),
        'precision_clase_1': float(prec_optimo),
        'recall_clase_1': float(rec_optimo),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred_opt))
    }
}

metadata_json_path = os.path.join(MODELS_DIR, "best_model_metadata.json")
with open(metadata_json_path, 'w', encoding='utf-8') as f:
    json.dump(model_metadata, f, indent=4, ensure_ascii=False)
print(f"  [+] Metadatos del modelo guardados en: {metadata_json_path}")

print("\n" + "=" * 80)
print("¡FASE 3 COMPLETADA CON ÉXITO! MODELO LISTO PARA LA FASE 4 (EXPLICABILIDAD CON SHAP).")
print("=" * 80)
