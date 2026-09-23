#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Simulación e Inferencia de Empleabilidad (TFM)
Ruta: Scrips/simulador.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAIN_TEST_DIR = os.path.join(BASE_DIR, "Data", "processed", "train_test")


class SimuladorEmpleabilidad:
    """
    Motor de predicción, diagnóstico, explicabilidad y simulación contrafactual
    de la empleabilidad en personas con discapacidad.
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
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {self.model_path}")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"No se encontró el preprocesador en: {self.preprocessor_path}")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            self.umbral_optimo = self.metadata.get('umbral_optimo', 0.43)
        else:
            self.metadata = {}
            self.umbral_optimo = 0.43

    def _inicializar_explicador(self):
        if HAS_SHAP:
            try:
                self.explainer = shap.TreeExplainer(self.model)
                self.has_shap_explainer = True
            except Exception:
                self.has_shap_explainer = False
        else:
            self.has_shap_explainer = False

    def procesar_perfil_crudo(self, perfil_dict):
        p = perfil_dict.copy()

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

        lim_vis = int(p.get('lim_vision', 0))
        lim_aud = int(p.get('lim_audicion', 0))
        lim_com = int(p.get('lim_comunicacion', 0))
        lim_apr = int(p.get('lim_aprendizaje', 0))
        lim_mov = int(p.get('lim_movilidad', 0))
        lim_aut = int(p.get('lim_autocuidado', 0))

        tiene_internet = int(p.get('tiene_internet', 1))
        tiene_ordenador = int(p.get('tiene_ordenador', 1))

        # Variables engineered
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

        return pd.DataFrame([registro])[self.COLUMNAS_RAW_ORDEN]

    def predecir(self, perfil_dict):
        df_raw = self.procesar_perfil_crudo(perfil_dict)
        X_proc = self.preprocessor.transform(df_raw)

        probabilidad = float(self.model.predict_proba(X_proc)[0, 1])
        clasificacion = 1 if probabilidad >= self.umbral_optimo else 0

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

        return {
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

    def explicar_perfil(self, perfil_dict, top_n=5):
        df_raw = self.procesar_perfil_crudo(perfil_dict)
        X_proc = self.preprocessor.transform(df_raw)
        feature_names = self._obtener_nombres_features()

        if self.has_shap_explainer:
            try:
                shap_res = self.explainer(X_proc)
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
            except Exception:
                pass

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

        return {
            'factores_favorables': favorables,
            'factores_barrera': barreras,
            'metodo': 'Reglas Heurísticas'
        }

    def _obtener_nombres_features(self):
        try:
            return self.preprocessor.get_feature_names_out()
        except Exception:
            metadata_features = os.path.join(TRAIN_TEST_DIR, "features_metadata.json")
            if os.path.exists(metadata_features):
                with open(metadata_features, 'r', encoding='utf-8') as f:
                    return json.load(f).get('feature_names', [])
            return [f"feature_{i}" for i in range(50)]

    def simular_palancas_what_if(self, perfil_dict):
        base_res = self.predecir(perfil_dict)
        p_base = base_res['probabilidad_pct']

        escenarios = []

        # 1. Inclusión digital
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

        # 2. Cualificación FP
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

        # 3. Universidad
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

        # 4. Plan Integral
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

        return {
            'probabilidad_base_pct': p_base,
            'es_empleable_base': base_res['es_empleable'],
            'escenarios': escenarios
        }
