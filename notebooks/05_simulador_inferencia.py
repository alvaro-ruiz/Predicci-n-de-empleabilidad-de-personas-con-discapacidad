#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFM: Predicción de la Empleabilidad de Personas con Discapacidad
Script / Notebook Interactivo: 05_simulador_inferencia.py
----------------------------------------------------------------
Fase 5: Pipeline de Inferencia, Simulador Predictivo y Motor de Recomendación (What-If)
- Carga de artefactos serializados (modelo Random Forest, ColumnTransformer y metadatos).
- Clase 'SimuladorEmpleabilidad':
  * Ingesta de datos de perfil en crudo (formulario de usuario/orientador).
  * Feature Engineering automático (conectividad TIC, severidad funcional, etc.).
  * Pipeline de transformación y predicción probabilística calibrada (umbral 0.43).
  * Diagnóstico cualitativo y semáforo de empleabilidad.
  * Explicabilidad local inmediata de factores clave (valores SHAP individuales).
  * Motor de simulación contrafactual What-If: cuantificación del impacto de palancas formativas y digitales.
- Batería de validación con perfiles arquetípicos y exportación de informe JSON.
"""

# %% 1. Importación de Librerías y Configuración
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

# Detección de SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings('ignore')

# Configuración de rutas
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if "__file__" in locals() else os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(MODELS_DIR, "reports")
TRAIN_TEST_DIR = os.path.join(BASE_DIR, "Data", "processed", "train_test")

os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 80)
print("FASE 5: PIPELINE DE INFERENCIA, SIMULADOR Y MOTOR DE RECOMENDACIÓN (WHAT-IF)")
print("=" * 80)


# %% 2. Definición de la Clase SimuladorEmpleabilidad
class SimuladorEmpleabilidad:
    """
    Clase integral para inferencia, diagnóstico, explicabilidad local y simulación contrafactual
    sobre la empleabilidad de personas con discapacidad.
    """

    COLUMNAS_RAW_ORDEN = [
        'edad', 'sexo', 'ccaa', 'tamano_municipio', 'nivel_estudios_agrupado',
        'certificado_oficial', 'tramo_grado_discapacidad', 'lim_vision', 'lim_audicion',
        'lim_comunicacion', 'lim_aprendizaje', 'lim_movilidad', 'lim_autocuidado',
        'num_limitaciones_graves', 'estado_civil', 'tipo_hogar', 'tamano_hogar',
        'ingresos_hogar', 'tiene_internet', 'tiene_ordenador', 'indice_conectividad',
        'pluridiscapacidad_grave', 'lim_sensorial', 'lim_autonomia_diaria',
        'lim_cognitivo_relacional', 'edad_senior_con_limitacion', 'vive_solo'
    ]

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "best_model.joblib")
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        self.metadata_path = os.path.join(models_dir, "best_model_metadata.json")

        self._cargar_artefactos()
        self._inicializar_explicador()

    def _cargar_artefactos(self):
        """Carga el modelo, el preprocesador y los metadatos de calibración."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {self.model_path}")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"No se encontró el preprocesador en: {self.preprocessor_path}")

        print(f"Cargando modelo serializado desde: {self.model_path}")
        self.model = joblib.load(self.model_path)

        print(f"Cargando preprocesador desde: {self.preprocessor_path}")
        self.preprocessor = joblib.load(self.preprocessor_path)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            self.umbral_optimo = self.metadata.get('umbral_optimo', 0.43)
        else:
            self.metadata = {}
            self.umbral_optimo = 0.43

        print(f"Modelo cargado: {type(self.model).__name__}")
        print(f"Umbral óptimo calibrado: {self.umbral_optimo:.2f}")

    def _inicializar_explicador(self):
        """Inicializa TreeExplainer de SHAP si está disponible en el entorno."""
        if HAS_SHAP:
            try:
                self.explainer = shap.TreeExplainer(self.model)
                self.has_shap_explainer = True
                print("Explicador SHAP TreeExplainer inicializado correctamente.")
            except Exception as e:
                print(f"[!] No se pudo inicializar SHAP TreeExplainer: {e}. Se utilizarán importancias relativas.")
                self.has_shap_explainer = False
        else:
            self.has_shap_explainer = False

    def procesar_perfil_crudo(self, perfil_dict):
        """
        Toma un diccionario con los datos base de una persona y aplica
        automáticamente la ingeniería de características (Fase 2).
        """
        p = perfil_dict.copy()

        # Variables base con valores por defecto si no vienen informados
        edad = int(p.get('edad', 40))
        tamano_hogar = int(p.get('tamano_hogar', 2))
        tipo_hogar = str(p.get('tipo_hogar', 'Pareja con hijos'))
        sexo = str(p.get('sexo', 'Mujer'))
        ccaa = str(p.get('ccaa', 'Andalucía'))
        tamano_municipio = str(p.get('tamano_municipio', '50.000 a 99.999 hab'))
        nivel_estudios = str(p.get('nivel_estudios_agrupado', 'Secundaria / Bachillerato'))
        certificado_oficial = int(p.get('certificado_oficial', 1))
        tramo_grado = str(p.get('tramo_grado_discapacidad', 'De 33% a 64% (Moderado)'))
        estado_civil = str(p.get('estado_civil', 'Soltero/a'))
        ingresos_hogar = str(p.get('ingresos_hogar', '1.000 a 1.499 €'))

        # Limitaciones funcionales
        lim_vis = int(p.get('lim_vision', 0))
        lim_aud = int(p.get('lim_audicion', 0))
        lim_com = int(p.get('lim_comunicacion', 0))
        lim_apr = int(p.get('lim_aprendizaje', 0))
        lim_mov = int(p.get('lim_movilidad', 0))
        lim_aut = int(p.get('lim_autocuidado', 0))

        # TIC
        tiene_internet = int(p.get('tiene_internet', 1))
        tiene_ordenador = int(p.get('tiene_ordenador', 1))

        # --- FEATURE ENGINEERING ---
        num_limitaciones = lim_vis + lim_aud + lim_com + lim_apr + lim_mov + lim_aut
        indice_conectividad = tiene_internet + tiene_ordenador
        pluridiscapacidad = 1 if num_limitaciones >= 2 else 0
        lim_sensorial = 1 if (lim_vis == 1 or lim_aud == 1) else 0
        lim_autonomia_diaria = 1 if (lim_mov == 1 or lim_aut == 1) else 0
        lim_cognitivo_relacional = 1 if (lim_apr == 1 or lim_com == 1) else 0
        edad_senior_con_limitacion = 1 if (edad >= 50 and num_limitaciones >= 1) else 0
        vive_solo = 1 if (tamano_hogar == 1 or tipo_hogar == 'Hogar unipersonal') else 0

        registro = {
            'edad': edad,
            'sexo': sexo,
            'ccaa': ccaa,
            'tamano_municipio': tamano_municipio,
            'nivel_estudios_agrupado': nivel_estudios,
            'certificado_oficial': certificado_oficial,
            'tramo_grado_discapacidad': tramo_grado,
            'lim_vision': lim_vis,
            'lim_audicion': lim_aud,
            'lim_comunicacion': lim_com,
            'lim_aprendizaje': lim_apr,
            'lim_movilidad': lim_mov,
            'lim_autocuidado': lim_aut,
            'num_limitaciones_graves': num_limitaciones,
            'estado_civil': estado_civil,
            'tipo_hogar': tipo_hogar,
            'tamano_hogar': tamano_hogar,
            'ingresos_hogar': ingresos_hogar,
            'tiene_internet': tiene_internet,
            'tiene_ordenador': tiene_ordenador,
            'indice_conectividad': indice_conectividad,
            'pluridiscapacidad_grave': pluridiscapacidad,
            'lim_sensorial': lim_sensorial,
            'lim_autonomia_diaria': lim_autonomia_diaria,
            'lim_cognitivo_relacional': lim_cognitivo_relacional,
            'edad_senior_con_limitacion': edad_senior_con_limitacion,
            'vive_solo': vive_solo
        }

        # Convertir a DataFrame asegurando el orden exacto de columnas
        df_raw = pd.DataFrame([registro])[self.COLUMNAS_RAW_ORDEN]
        return df_raw

    def predecir(self, perfil_dict):
        """
        Ejecuta la inferencia completa sobre un perfil y devuelve un dict con:
        - Probabilidad predicha (%)
        - Predicción binaria según el umbral óptimo (0.43)
        - Nivel cualitativo de empleabilidad
        - Semáforo (Verde, Amarillo, Naranja, Rojo)
        - Factores explicativos
        """
        df_raw = self.procesar_perfil_crudo(perfil_dict)
        X_proc = self.preprocessor.transform(df_raw)

        # Probabilidad de clase positiva (empleado = 1)
        probabilidad = float(self.model.predict_proba(X_proc)[0, 1])
        clasificacion = 1 if probabilidad >= self.umbral_optimo else 0

        # Diagnóstico cualitativo
        if probabilidad >= 0.65:
            nivel = "Muy Favorable"
            semaforo = "Verde"
            diagnostico = "Alta probabilidad de inserción o mantenimiento laboral activo."
        elif probabilidad >= self.umbral_optimo:
            nivel = "Favorable / Empleable"
            semaforo = "Verde Claro"
            diagnostico = "Supera el umbral óptimo de empleabilidad (>= 43%). Perfil con viabilidad laboral positiva."
        elif probabilidad >= 0.25:
            nivel = "Vulnerabilidad Moderada"
            semaforo = "Naranja"
            diagnostico = "Por debajo del umbral óptimo. Requiere activación formativa y apoyo sociolaboral."
        else:
            nivel = "Alta Vulnerabilidad / Barreras Severas"
            semaforo = "Rojo"
            diagnostico = "Probabilidad reducida. Presencia de múltiples barreras estructurales o funcionales."

        resultado = {
            'probabilidad': probabilidad,
            'probabilidad_pct': round(probabilidad * 100, 1),
            'umbral_utilizado': self.umbral_optimo,
            'clasificacion': clasificacion,
            'es_empleable': bool(clasificacion == 1),
            'nivel_empleabilidad': nivel,
            'color_semaforo': semaforo,
            'diagnostico': diagnostico,
            'datos_procesados': df_raw.iloc[0].to_dict()
        }

        return resultado

    def explicar_perfil(self, perfil_dict, top_n=5):
        """
        Devuelve los principales factores que impulsan a favor o restan
        probabilidad de inserción para este individuo específico.
        """
        df_raw = self.procesar_perfil_crudo(perfil_dict)
        X_proc = self.preprocessor.transform(df_raw)

        # Nombres de las columnas procesadas
        feature_names = self._obtener_nombres_features()

        if self.has_shap_explainer:
            try:
                shap_res = self.explainer(X_proc)
                # Formato clase positiva
                if len(shap_res.shape) == 3 and shap_res.shape[2] == 2:
                    vals = shap_res.values[0, :, 1]
                elif len(shap_res.shape) == 2:
                    vals = shap_res.values[0, :]
                else:
                    vals = shap_res.values[0]

                contribuciones = pd.DataFrame({
                    'caracteristica': feature_names,
                    'impacto_shap': vals
                }).sort_values(by='impacto_shap', ascending=False)

                positivos = contribuciones[contribuciones['impacto_shap'] > 0].head(top_n).to_dict(orient='records')
                negativos = contribuciones[contribuciones['impacto_shap'] < 0].tail(top_n).sort_values(by='impacto_shap').to_dict(orient='records')

                return {
                    'factores_favorables': positivos,
                    'factores_barrera': negativos,
                    'metodo': 'SHAP (Valores Shapley individuales)'
                }
            except Exception as e:
                print(f"[!] Error calculando SHAP local: {e}")

        # Fallback de explicabilidad si SHAP no está disponible
        return self._explicacion_heuristica_dominio(perfil_dict)

    def _obtener_nombres_features(self):
        """Extrae nombres de las 50 características resultantes del preprocesador."""
        try:
            return self.preprocessor.get_feature_names_out()
        except Exception:
            metadata_features = os.path.join(TRAIN_TEST_DIR, "features_metadata.json")
            if os.path.exists(metadata_features):
                with open(metadata_features, 'r', encoding='utf-8') as f:
                    return json.load(f).get('feature_names', [])
            return [f"feature_{i}" for i in range(50)]

    def _explicacion_heuristica_dominio(self, perfil_dict):
        """Explicación basada en reglas del dominio si falla SHAP."""
        p = perfil_dict
        favorables = []
        barreras = []

        if p.get('nivel_estudios_agrupado') in ['Universidad / Posgrado', 'Formación Profesional']:
            favorables.append({'factor': 'Nivel Educativo Alto', 'detalle': p.get('nivel_estudios_agrupado')})
        else:
            barreras.append({'factor': 'Bajo Nivel Educativo', 'detalle': p.get('nivel_estudios_agrupado')})

        if p.get('tiene_internet', 0) == 1 and p.get('tiene_ordenador', 0) == 1:
            favorables.append({'factor': 'Conectividad TIC Plena', 'detalle': 'Dispone de PC e Internet'})
        elif p.get('tiene_internet', 0) == 0 and p.get('tiene_ordenador', 0) == 0:
            barreras.append({'factor': 'Brecha Digital Total', 'detalle': 'Sin PC ni conexión a Internet'})

        if int(p.get('edad', 40)) < 40:
            favorables.append({'factor': 'Edad Joven / Madurez Temprana', 'detalle': f"{p.get('edad')} años"})
        elif int(p.get('edad', 40)) >= 50:
            barreras.append({'factor': 'Edad Sénior', 'detalle': f"{p.get('edad')} años"})

        return {
            'factores_favorables': favorables,
            'factores_barrera': barreras,
            'metodo': 'Reglas Heurísticas del Dominio'
        }

    def simular_palancas_what_if(self, perfil_dict):
        """
        Motor de Simulación Contrafactual ("What-If"):
        Evalúa cómo cambiaría la probabilidad predicha si el usuario:
        1. Resuelve la brecha digital (adquiere Internet y Ordenador).
        2. Mejora su cualificación a Formación Profesional (FP).
        3. Obtiene titulación Universitaria.
        4. Intervención integral combinada: FP + Conectividad TIC plena.
        """
        # Predicción base
        base_res = self.predecir(perfil_dict)
        p_base = base_res['probabilidad_pct']

        escenarios = []

        # Escenario 1: Conectividad Digital Plena
        sc1 = perfil_dict.copy()
        sc1['tiene_internet'] = 1
        sc1['tiene_ordenador'] = 1
        res1 = self.predecir(sc1)
        escenarios.append({
            'escenario': '1. Inclusión Digital Plena (Internet + PC)',
            'probabilidad_pct': res1['probabilidad_pct'],
            'diferencia_pct': round(res1['probabilidad_pct'] - p_base, 1),
            'clasificacion': res1['es_empleable'],
            'cambia_decision': res1['es_empleable'] != base_res['es_empleable']
        })

        # Escenario 2: Cualificación Formación Profesional
        sc2 = perfil_dict.copy()
        sc2['nivel_estudios_agrupado'] = 'Formación Profesional'
        res2 = self.predecir(sc2)
        escenarios.append({
            'escenario': '2. Cualificación en Formación Profesional (FP)',
            'probabilidad_pct': res2['probabilidad_pct'],
            'diferencia_pct': round(res2['probabilidad_pct'] - p_base, 1),
            'clasificacion': res2['es_empleable'],
            'cambia_decision': res2['es_empleable'] != base_res['es_empleable']
        })

        # Escenario 3: Titulación Universitaria
        sc3 = perfil_dict.copy()
        sc3['nivel_estudios_agrupado'] = 'Universidad / Posgrado'
        res3 = self.predecir(sc3)
        escenarios.append({
            'escenario': '3. Titulación Universitaria / Posgrado',
            'probabilidad_pct': res3['probabilidad_pct'],
            'diferencia_pct': round(res3['probabilidad_pct'] - p_base, 1),
            'clasificacion': res3['es_empleable'],
            'cambia_decision': res3['es_empleable'] != base_res['es_empleable']
        })

        # Escenario 4: Plan Integral (FP + Conectividad Plena)
        sc4 = perfil_dict.copy()
        sc4['tiene_internet'] = 1
        sc4['tiene_ordenador'] = 1
        sc4['nivel_estudios_agrupado'] = 'Formación Profesional'
        res4 = self.predecir(sc4)
        escenarios.append({
            'escenario': '4. Intervención Integral (FP + Conectividad Plena)',
            'probabilidad_pct': res4['probabilidad_pct'],
            'diferencia_pct': round(res4['probabilidad_pct'] - p_base, 1),
            'clasificacion': res4['es_empleable'],
            'cambia_decision': res4['es_empleable'] != base_res['es_empleable']
        })

        df_what_if = pd.DataFrame(escenarios)
        return {
            'probabilidad_base_pct': p_base,
            'es_empleable_base': base_res['es_empleable'],
            'escenarios': df_what_if.to_dict(orient='records')
        }


# %% 3. Batería de Validación con Perfiles Arquetípicos
print("\n" + "=" * 80)
print("EJECUCIÓN DE PRUEBAS DE VALIDACIÓN CON PERFILES ARQUETÍPICOS")
print("=" * 80)

simulador = SimuladorEmpleabilidad()

# Definición de 3 perfiles prototípicos de estudio
perfil_alta_empleabilidad = {
    'edad': 28,
    'sexo': 'Mujer',
    'ccaa': 'Comunidad de Madrid',
    'tamano_municipio': 'Más de 500.000 hab',
    'nivel_estudios_agrupado': 'Universidad / Posgrado',
    'certificado_oficial': 1,
    'tramo_grado_discapacidad': 'De 33% a 64% (Moderado)',
    'lim_vision': 1,
    'lim_audicion': 0,
    'lim_comunicacion': 0,
    'lim_aprendizaje': 0,
    'lim_movilidad': 0,
    'lim_autocuidado': 0,
    'estado_civil': 'Soltero/a',
    'tipo_hogar': 'Pareja sin hijos',
    'tamano_hogar': 2,
    'ingresos_hogar': '2.500 a 2.999 €',
    'tiene_internet': 1,
    'tiene_ordenador': 1
}

perfil_vulnerable = {
    'edad': 56,
    'sexo': 'Hombre',
    'ccaa': 'Andalucía',
    'tamano_municipio': 'Menos de 10.000 hab',
    'nivel_estudios_agrupado': 'Sin estudios / Primaria',
    'certificado_oficial': 1,
    'tramo_grado_discapacidad': 'De 65% a 74% (Severo)',
    'lim_vision': 0,
    'lim_audicion': 0,
    'lim_comunicacion': 0,
    'lim_aprendizaje': 0,
    'lim_movilidad': 1,
    'lim_autocuidado': 1,
    'estado_civil': 'Separado/a o Divorciado/a',
    'tipo_hogar': 'Hogar unipersonal',
    'tamano_hogar': 1,
    'ingresos_hogar': 'Menos de 500 €',
    'tiene_internet': 0,
    'tiene_ordenador': 0
}

perfil_limite_what_if = {
    'edad': 47,
    'sexo': 'Mujer',
    'ccaa': 'Comunidad Valenciana',
    'tamano_municipio': '50.000 a 99.999 hab',
    'nivel_estudios_agrupado': 'Sin estudios / Primaria',
    'certificado_oficial': 1,
    'tramo_grado_discapacidad': 'De 33% a 64% (Moderado)',
    'lim_vision': 0,
    'lim_audicion': 0,
    'lim_comunicacion': 0,
    'lim_aprendizaje': 0,
    'lim_movilidad': 1,
    'lim_autocuidado': 0,
    'estado_civil': 'Soltero/a',
    'tipo_hogar': 'Monoparental',
    'tamano_hogar': 3,
    'ingresos_hogar': '500 a 999 €',
    'tiene_internet': 0,
    'tiene_ordenador': 0
}

casos_prueba = [
    ("Caso A: Perfil Joven con Titulación Superior y Discapacidad Sensorial", perfil_alta_empleabilidad),
    ("Caso B: Perfil Sénior en Situación de Vulnerabilidad y Brecha Digital", perfil_vulnerable),
    ("Caso C: Perfil en Zona Límite (Candidato a Intervención What-If)", perfil_limite_what_if)
]

informe_casos = []

for titulo, perfil in casos_prueba:
    print("\n" + "-" * 75)
    print(f"EVALUANDO: {titulo}")
    print("-" * 75)

    res = simulador.predecir(perfil)
    print(f"Probabilidad Estimada:   {res['probabilidad_pct']}% (Umbral Óptimo: {res['umbral_utilizado'] * 100:.1f}%)")
    print(f"Diagnóstico:             {res['nivel_empleabilidad']} [{res['color_semaforo']}]")
    print(f"Veredicto Clasificación: {'EMPLEABLE (Positivo)' if res['es_empleable'] else 'VULNERABLE (Negativo)'}")
    print(f"Detalle:                 {res['diagnostico']}")

    # Explicabilidad local
    exp = simulador.explicar_perfil(perfil, top_n=3)
    print("\nFactores Determinantes (Top 3):")
    if 'factores_favorables' in exp and exp['factores_favorables']:
        print("  [+] A Favor:", exp['factores_favorables'][:3])
    if 'factores_barrera' in exp and exp['factores_barrera']:
        print("  [-] En Contra:", exp['factores_barrera'][:3])

    # Simulación contrafactual What-If
    what_if = simulador.simular_palancas_what_if(perfil)
    print("\nSimulación Contrafactual de Palancas (What-If):")
    for esc in what_if['escenarios']:
        signo = "+" if esc['diferencia_pct'] >= 0 else ""
        cambio = " -> ¡CAMBIA A EMPLEABLE!" if esc['cambia_decision'] else ""
        print(f"  * {esc['escenario']}: {esc['probabilidad_pct']}% ({signo}{esc['diferencia_pct']} pts){cambio}")

    informe_casos.append({
        'caso': titulo,
        'perfil_entrada': perfil,
        'resultado_prediccion': res,
        'explicabilidad': exp,
        'simulacion_what_if': what_if
    })

# %% 4. Guardado de Informe de Simulación en JSON
json_output_path = os.path.join(REPORTS_DIR, "simulacion_casos_report.json")

# Serializador seguro para objetos numpy/bool
def serializador_seguro(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(informe_casos, f, indent=4, ensure_ascii=False, default=serializador_seguro)

print("\n" + "=" * 80)
print(f"[+] Informe completo de simulación guardado en: {json_output_path}")
print("¡FASE 5: PIPELINE DE INFERENCIA Y SIMULADOR COMPLETADOS CON ÉXITO!")
print("=" * 80)
