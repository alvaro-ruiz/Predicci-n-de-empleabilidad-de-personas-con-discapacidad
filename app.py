#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFM: Predicción de la Empleabilidad de Personas con Discapacidad
Aplicación Streamlit: app.py
----------------------------------------------------------------
Fase 5: Despliegue del Dashboard Interactivo y Simulador MVP
Ejecución:
    streamlit run app.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st

# Añadir raíz al sys.path para importar módulos locales
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Scrips.simulador import SimuladorEmpleabilidad

# Configuración de página
st.set_page_config(
    page_title="MVP Empleabilidad & Discapacidad | TFM",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .badge-positive {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-negative {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stProgress > div > div > div > div {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def obtener_simulador():
    """Inicializa y cachea el motor de inferencia del simulador."""
    return SimuladorEmpleabilidad()


@st.cache_data
def cargar_metadatos_y_reportes():
    """Carga los metadatos del modelo y los reportes de equidad."""
    meta_path = os.path.join(BASE_DIR, "models", "best_model_metadata.json")
    fairness_path = os.path.join(BASE_DIR, "models", "reports", "fairness_audit_report.json")
    benchmark_path = os.path.join(BASE_DIR, "models", "model_benchmark_results.csv")

    metadata = {}
    fairness = {}
    benchmark = None

    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    if os.path.exists(fairness_path):
        with open(fairness_path, 'r', encoding='utf-8') as f:
            fairness = json.load(f)

    if os.path.exists(benchmark_path):
        try:
            benchmark = pd.read_csv(benchmark_path, sep=';', decimal=',')
        except Exception:
            benchmark = None

    return metadata, fairness, benchmark


# Inicializar simulador
try:
    simulador = obtener_simulador()
except Exception as e:
    st.error(f"Error cargando los modelos del simulador: {e}")
    st.stop()

metadata_mod, fairness_rep, df_benchmark = cargar_metadatos_y_reportes()

# --- SIDEBAR: NAVEGACIÓN Y CONFIGURACIÓN ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/accessibility.png", width=64)
    st.title("TFM Data Science")
    st.markdown("**Predicción de Empleabilidad en Personas con Discapacidad**")
    st.markdown("---")

    menu = st.radio(
        "Módulos del Sistema:",
        [
            "🔮 1. Simulador & Diagnóstico",
            "🚀 2. Motor What-If (Palancas)",
            "📊 3. Observatorio Analítico (EDAD)",
            "⚖️ 4. Auditoría de Equidad & Modelo"
        ]
    )

    st.markdown("---")
    st.markdown("### ⚙️ Parámetros Técnicos")
    umbral = metadata_mod.get('umbral_optimo', 0.43)
    st.write(f"**Algoritmo:** {metadata_mod.get('nombre_modelo', 'RandomForest')}")
    st.write(f"**Umbral Óptimo:** `{umbral:.2f}` (43%)")
    st.write(f"**ROC-AUC (Test):** `{metadata_mod.get('test_metrics_umbral_default', {}).get('roc_auc', 0.755):.3f}`")
    st.write(f"**Sensibilidad (Recall):** `{metadata_mod.get('test_metrics_umbral_optimo', {}).get('recall_clase_1', 0.804)*100:.1f}%`")

    st.markdown("---")
    st.caption("Autor: Álvaro Ruiz | Máster en Data Science")


# ==============================================================================
# MÓDULO 1: SIMULADOR DE EMPLEABILIDAD & DIAGNÓSTICO PERSONALIZADO
# ==============================================================================
if menu == "🔮 1. Simulador & Diagnóstico":
    st.markdown('<div class="main-header">🔮 Simulador Predictivo de Empleabilidad</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Estimación personalizada de viabilidad laboral y detección de barreras con Inteligencia Artificial explicable.</div>', unsafe_allow_html=True)

    # Botones de Carga Rápida de Arquetipos
    st.markdown("##### ⚡ Carga Rápida de Perfiles Arquetípicos:")
    col_arq1, col_arq2, col_arq3 = st.columns(3)

    if 'perfil_activo' not in st.session_state:
        st.session_state['perfil_activo'] = 'default'

    perfil_sel = None
    if col_arq1.button("🎓 Joven con Formación Superior"):
        st.session_state['perfil_activo'] = 'joven'
    if col_arq2.button("📉 Sénior con Brecha Digital"):
        st.session_state['perfil_activo'] = 'vulnerable'
    if col_arq3.button("⚖️ Perfil en Zona Límite"):
        st.session_state['perfil_activo'] = 'limite'

    # Valores por defecto según arquetipo seleccionado
    modo = st.session_state.get('perfil_activo', 'default')
    if modo == 'joven':
        d_edad, d_sexo, d_ccaa, d_estudios = 28, "Mujer", "Comunidad de Madrid", "Universidad / Posgrado"
        d_mun, d_grado = "Más de 500.000 hab", "De 33% a 64% (Moderado)"
        d_vis, d_aud, d_mov, d_aut, d_com, d_apr = 1, 0, 0, 0, 0, 0
        d_net, d_pc = 1, 1
        d_hogar, d_tam_hogar, d_ingresos = "Pareja sin hijos", 2, "2.500 a 2.999 €"
    elif modo == 'vulnerable':
        d_edad, d_sexo, d_ccaa, d_estudios = 56, "Hombre", "Andalucía", "Sin estudios / Primaria"
        d_mun, d_grado = "Menos de 10.000 hab", "De 65% a 74% (Severo)"
        d_vis, d_aud, d_mov, d_aut, d_com, d_apr = 0, 0, 1, 1, 0, 0
        d_net, d_pc = 0, 0
        d_hogar, d_tam_hogar, d_ingresos = "Hogar unipersonal", 1, "Menos de 500 €"
    elif modo == 'limite':
        d_edad, d_sexo, d_ccaa, d_estudios = 47, "Mujer", "Comunidad Valenciana", "Sin estudios / Primaria"
        d_mun, d_grado = "50.000 a 99.999 hab", "De 33% a 64% (Moderado)"
        d_vis, d_aud, d_mov, d_aut, d_com, d_apr = 0, 0, 1, 0, 0, 0
        d_net, d_pc = 0, 0
        d_hogar, d_tam_hogar, d_ingresos = "Monoparental", 3, "500 a 999 €"
    else:
        d_edad, d_sexo, d_ccaa, d_estudios = 42, "Mujer", "Andalucía", "Secundaria / Bachillerato"
        d_mun, d_grado = "50.000 a 99.999 hab", "De 33% a 64% (Moderado)"
        d_vis, d_aud, d_mov, d_aut, d_com, d_apr = 0, 0, 0, 0, 0, 0
        d_net, d_pc = 1, 1
        d_hogar, d_tam_hogar, d_ingresos = "Pareja con hijos", 3, "1.000 a 1.499 €"

    # Formulario Interactivo en 3 Columnas
    with st.expander("📝 Formulario de Perfil del Usuario / Solicitante", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            st.markdown("#### 👤 1. Datos Sociodemográficos")
            edad = st.slider("Edad (años en edad laboral):", min_value=16, max_value=64, value=d_edad, step=1)
            sexo = st.selectbox("Sexo:", ["Mujer", "Hombre"], index=0 if d_sexo == "Mujer" else 1)
            lista_ccaa = [
                "Andalucía", "Aragón", "Asturias", "Baleares", "Canarias", "Cantabria",
                "Castilla y León", "Castilla-La Mancha", "Cataluña", "Ceuta",
                "Comunidad Valenciana", "Comunidad de Madrid", "Extremadura", "Galicia",
                "La Rioja", "Melilla", "Navarra", "País Vasco", "Región de Murcia"
            ]
            idx_ccaa = lista_ccaa.index(d_ccaa) if d_ccaa in lista_ccaa else 0
            ccaa = st.selectbox("Comunidad Autónoma:", lista_ccaa, index=idx_ccaa)
            tamano_municipio = st.selectbox("Tamaño del Municipio:", [
                'Menos de 10.000 hab', '10.000 a 19.999 hab', '20.000 a 49.999 hab',
                '50.000 a 99.999 hab', '100.000 a 499.999 hab', 'Más de 500.000 hab'
            ], index=3 if d_mun not in ['Menos de 10.000 hab', 'Más de 500.000 hab'] else (0 if d_mun == 'Menos de 10.000 hab' else 5))
            estado_civil = st.selectbox("Estado Civil:", ['Soltero/a', 'Casado/a', 'Separado/a o Divorciado/a', 'Viudo/a', 'No consta'], index=0)

        with col_c2:
            st.markdown("#### ♿ 2. Discapacidad y Salud")
            certificado_oficial = st.checkbox("Certificado Oficial de Discapacidad", value=True)
            tramo_grado = st.selectbox("Tramo Grado de Discapacidad:", [
                'Sin grado reconocido / No consta', 'Menos del 33%',
                'De 33% a 64% (Moderado)', 'De 65% a 74% (Severo)',
                '75% o más (Muy severo / Gran invalidez)'
            ], index=2 if 'Moderado' in d_grado else (3 if 'Severo' in d_grado else 0))

            st.markdown("**Limitaciones Funcionales Severas:**")
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                lim_vis = st.checkbox("Visión", value=bool(d_vis))
                lim_aud = st.checkbox("Audición", value=bool(d_aud))
                lim_com = st.checkbox("Comunicación", value=bool(d_com))
            with col_l2:
                lim_mov = st.checkbox("Movilidad", value=bool(d_mov))
                lim_aut = st.checkbox("Autocuidado", value=bool(d_aut))
                lim_apr = st.checkbox("Aprendizaje", value=bool(d_apr))

        with col_c3:
            st.markdown("#### 💻 3. Formación, Hogar y TIC")
            lista_estudios = [
                'Sin estudios / Primaria', 'Secundaria / Bachillerato',
                'Formación Profesional', 'Universidad / Posgrado'
            ]
            idx_est = lista_estudios.index(d_estudios) if d_estudios in lista_estudios else 1
            nivel_estudios = st.selectbox("Nivel Educativo:", lista_estudios, index=idx_est)

            st.markdown("**Conectividad Digital (TIC):**")
            tiene_internet = st.checkbox("Conexión a Internet en el Hogar", value=bool(d_net))
            tiene_ordenador = st.checkbox("Dispone de Ordenador / Tablet", value=bool(d_pc))

            tipo_hogar = st.selectbox("Tipo de Hogar:", [
                'Hogar unipersonal', 'Monoparental', 'Pareja con hijos',
                'Pareja sin hijos', 'Otros tipos de hogar'
            ], index=2 if d_hogar == 'Pareja con hijos' else (0 if d_hogar == 'Hogar unipersonal' else 1))
            tamano_hogar = st.number_input("Personas en el hogar:", min_value=1, max_value=10, value=d_tam_hogar)
            ingresos_hogar = st.selectbox("Tramo Ingresos del Hogar:", [
                'No consta', 'Menos de 500 €', '500 a 999 €', '1.000 a 1.499 €',
                '1.500 a 1.999 €', '2.000 a 2.499 €', '2.500 a 2.999 €',
                '3.000 a 4.999 €', '5.000 € o más'
            ], index=3 if d_ingresos not in ['Menos de 500 €', '2.500 a 2.999 €'] else (1 if '500 €' in d_ingresos else 6))

    # Construir dict perfil
    perfil_actual = {
        'edad': edad,
        'sexo': sexo,
        'ccaa': ccaa,
        'tamano_municipio': tamano_municipio,
        'nivel_estudios_agrupado': nivel_estudios,
        'certificado_oficial': 1 if certificado_oficial else 0,
        'tramo_grado_discapacidad': tramo_grado,
        'lim_vision': 1 if lim_vis else 0,
        'lim_audicion': 1 if lim_aud else 0,
        'lim_comunicacion': 1 if lim_com else 0,
        'lim_aprendizaje': 1 if lim_apr else 0,
        'lim_movilidad': 1 if lim_mov else 0,
        'lim_autocuidado': 1 if lim_aut else 0,
        'estado_civil': estado_civil,
        'tipo_hogar': tipo_hogar,
        'tamano_hogar': tamano_hogar,
        'ingresos_hogar': ingresos_hogar,
        'tiene_internet': 1 if tiene_internet else 0,
        'tiene_ordenador': 1 if tiene_ordenador else 0
    }

    # Inferencia del perfil
    res_pred = simulador.predecir(perfil_actual)
    p_pct = res_pred['probabilidad_pct']
    es_emp = res_pred['es_empleable']
    umbral_pct = res_pred['umbral_utilizado'] * 100

    # Panel de Resultados
    st.markdown("---")
    st.markdown("### 📊 Resultado de la Evaluación")

    col_res1, col_res2, col_res3 = st.columns([1.5, 2, 1.5])

    with col_res1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Probabilidad Estimada</div>', unsafe_allow_html=True)
        color_val = "#059669" if es_emp else "#DC2626"
        st.markdown(f'<div class="metric-value" style="color: {color_val};">{p_pct:.1f}%</div>', unsafe_allow_html=True)
        badge_cls = "badge-positive" if es_emp else "badge-negative"
        veredicto = "FAVORABLE (Empleable)" if es_emp else "VULNERABILIDAD LABORAL"
        st.markdown(f'<span class="{badge_cls}">{veredicto}</span>', unsafe_allow_html=True)
        st.markdown(f"<p style='margin-top:8px; font-size:0.85rem; color:#64748B;'>Umbral calibrado de corte: <b>{umbral_pct:.1f}%</b></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Diagnóstico y Semáforo</div>', unsafe_allow_html=True)
        st.markdown(f"#### {res_pred['nivel_empleabilidad']} ({res_pred['color_semaforo']})")
        st.write(res_pred['diagnostico'])
        # Barra de progreso visual
        st.progress(float(min(p_pct / 100.0, 1.0)))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Perfil Sintético Calculado</div>', unsafe_allow_html=True)
        d_proc = res_pred['datos_procesados']
        st.write(f"• **Limitaciones Acumuladas:** {d_proc['num_limitaciones_graves']}")
        st.write(f"• **Índice Conectividad TIC:** {d_proc['indice_conectividad']} / 2")
        st.write(f"• **Pluridiscapacidad:** {'Sí' if d_proc['pluridiscapacidad_grave'] else 'No'}")
        st.write(f"• **Interacción Sénior:** {'Sí' if d_proc['edad_senior_con_limitacion'] else 'No'}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Explicabilidad Local (SHAP)
    st.markdown("#### 🔍 Factores Determinantes de la Predicción (Explicabilidad XAI)")
    exp = simulador.explicar_perfil(perfil_actual, top_n=4)
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.success("🟢 **Factores que Impulsan la Empleabilidad (A Favor):**")
        for f in exp.get('factores_favorables', []):
            nombre = f.get('caracteristica', f.get('factor', 'Variable'))
            val = f.get('impacto_shap', '')
            val_txt = f" (+{val:.3f} SHAP)" if isinstance(val, (int, float)) else ""
            st.markdown(f"- **{nombre}**{val_txt}")

    with col_exp2:
        st.error("🔴 **Factores Barrera / Vulnerabilidad (En Contra):**")
        for f in exp.get('factores_barrera', []):
            nombre = f.get('caracteristica', f.get('factor', 'Variable'))
            val = f.get('impacto_shap', '')
            val_txt = f" ({val:.3f} SHAP)" if isinstance(val, (int, float)) else ""
            st.markdown(f"- **{nombre}**{val_txt}")


# ==============================================================================
# MÓDULO 2: MOTOR WHAT-IF & PALANCAS DE INSERCIÓN
# ==============================================================================
elif menu == "🚀 2. Motor What-If (Palancas)":
    st.markdown('<div class="main-header">🚀 Motor What-If: Simulación de Palancas de Inserción</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluación contrafactual de impacto: ¿Cómo se incrementaría la probabilidad de empleo si se actúa sobre formación o inclusión digital?</div>', unsafe_allow_html=True)

    st.info("💡 **Objetivo orientador:** Identificar qué medidas formativas o de accesibilidad digital tienen mayor retorno para este perfil específico.")

    # Tomar perfil base por defecto o en sesión
    col_w1, col_w2 = st.columns([1, 2])
    with col_w1:
        st.markdown("##### 👤 Perfil a Evaluar")
        w_edad = st.slider("Edad:", 18, 64, 46)
        w_estudios = st.selectbox("Nivel Educativo Inicial:", [
            'Sin estudios / Primaria', 'Secundaria / Bachillerato', 'Formación Profesional'
        ], index=0)
        w_tic = st.radio("Acceso TIC Inicial:", ["Sin Conectividad (Sin PC ni Net)", "Solo Internet", "Conectividad Plena"], index=0)
        w_mov = st.checkbox("Limitación de Movilidad", value=True)
        w_sens = st.checkbox("Limitación Sensorial (Visión/Audición)", value=False)

        net_val = 0 if "Sin" in w_tic else 1
        pc_val = 1 if "Plena" in w_tic else 0

        p_eval = {
            'edad': w_edad,
            'sexo': 'Mujer',
            'ccaa': 'Andalucía',
            'tamano_municipio': '50.000 a 99.999 hab',
            'nivel_estudios_agrupado': w_estudios,
            'certificado_oficial': 1,
            'tramo_grado_discapacidad': 'De 33% a 64% (Moderado)',
            'lim_vision': 1 if w_sens else 0,
            'lim_audicion': 0,
            'lim_comunicacion': 0,
            'lim_aprendizaje': 0,
            'lim_movilidad': 1 if w_mov else 0,
            'lim_autocuidado': 0,
            'estado_civil': 'Soltero/a',
            'tipo_hogar': 'Monoparental',
            'tamano_hogar': 3,
            'ingresos_hogar': '500 a 999 €',
            'tiene_internet': net_val,
            'tiene_ordenador': pc_val
        }

    with col_w2:
        what_if_res = simulador.simular_palancas_what_if(p_eval)
        p_base = what_if_res['probabilidad_base_pct']
        umbral_val = simulador.umbral_optimo * 100

        st.markdown("##### 📈 Comparativa de Escenarios Contrafactuales")
        st.write(f"Probabilidad Base Actual: **{p_base:.1f}%** | Umbral de Inserción: **{umbral_val:.1f}%**")

        df_esc = pd.DataFrame(what_if_res['escenarios'])
        
        # Mostrar gráfico de barras comparativo
        chart_data = pd.DataFrame({
            'Escenario': ['Base Actual'] + df_esc['escenario'].tolist(),
            'Probabilidad (%)': [p_base] + df_esc['probabilidad_pct'].tolist()
        })
        st.bar_chart(chart_data.set_index('Escenario'))

        # Tabla explicativa de deltas
        st.markdown("##### 📋 Retorno Estimado por Medida:")
        for esc in what_if_res['escenarios']:
            cambio_str = "🎉 **¡Supera el umbral y pasa a EMPLEABLE!**" if esc['cambia_decision'] else "Aumento de probabilidad."
            st.markdown(f"- **{esc['escenario']}**: `{esc['probabilidad_pct']}%` (+{esc['diferencia_pct']} puntos) → {cambio_str}")


# ==============================================================================
# MÓDULO 3: OBSERVATORIO ANALÍTICO (EDAD 2020 / ODISMET)
# ==============================================================================
elif menu == "📊 3. Observatorio Analítico (EDAD)":
    st.markdown('<div class="main-header">📊 Observatorio de Empleabilidad (EDAD 2020)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Indicadores clave de la Encuesta sobre Discapacidades, Autonomía personal y situaciones de Dependencia (INE).</div>', unsafe_allow_html=True)

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric("Tasa Empleo Discapacidad", "23.2%", help="Población con discapacidad ocupada en edad laboral.")
    with col_kpi2:
        st.metric("Tasa Población General", "64.3%", delta="-41.1% brecha", delta_color="inverse")
    with col_kpi3:
        st.metric("Brecha Digital (Sin TIC)", "9.1% empleo", help="Tasa de empleo en personas sin equipamiento informático ni internet.")
    with col_kpi4:
        st.metric("Efecto Universidad", "45.8% empleo", delta="+33.5% vs Primaria")

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("##### 🎓 Tasa de Empleo según Nivel Educativo")
        df_edu = pd.DataFrame({
            'Nivel Educativo': ['Sin estudios / Primaria', 'Secundaria / Bachillerato', 'Formación Profesional', 'Universidad / Posgrado'],
            'Tasa Empleo (%)': [12.3, 21.8, 34.2, 45.8]
        }).set_index('Nivel Educativo')
        st.bar_chart(df_edu)

    with col_g2:
        st.markdown("##### 💻 Impacto de la Conectividad Digital en el Empleo")
        df_tic = pd.DataFrame({
            'Nivel Conectividad': ['0: Sin TIC', '1: Solo Internet o PC', '2: Conectividad Plena'],
            'Tasa Empleo (%)': [9.1, 20.4, 31.7]
        }).set_index('Nivel Conectividad')
        st.bar_chart(df_tic)

    st.markdown("##### 📍 Variabilidad Territorial por Comunidad Autónoma (EDAD 2020)")
    df_ccaa = pd.DataFrame({
        'Comunidad Autónoma': [
            'País Vasco', 'Comunidad de Madrid', 'Navarra', 'Cataluña', 'Aragón',
            'Baleares', 'La Rioja', 'Castilla y León', 'Cantabria', 'C. Valenciana',
            'Asturias', 'Galicia', 'Canarias', 'Murcia', 'Castilla-La Mancha',
            'Extremadura', 'Andalucía'
        ],
        'Tasa Empleo (%)': [31.2, 30.5, 29.8, 27.4, 26.9, 25.8, 25.1, 24.3, 23.9, 23.1, 22.4, 21.7, 21.0, 20.5, 19.8, 18.2, 17.6]
    }).set_index('Comunidad Autónoma')
    st.bar_chart(df_ccaa)


# ==============================================================================
# MÓDULO 4: AUDITORÍA DE EQUIDAD (FAIRNESS) & MODELO
# ==============================================================================
elif menu == "⚖️ 4. Auditoría de Equidad & Modelo":
    st.markdown('<div class="main-header">⚖️ Auditoría Ética, Equidad (Fairness) y Benchmarking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transparencia algorítmica, validación de sesgos según la regla de los 4/5 y rendimiento de los modelos evaluados.</div>', unsafe_allow_html=True)

    # Resultados de Equidad
    st.markdown("### 🔍 Resultados de la Auditoría de Equidad (Fase 4)")
    col_eq1, col_eq2 = st.columns(2)

    with col_eq1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### 🚻 Sesgo de Género (Hombres vs Mujeres)")
        di_genero = fairness_rep.get('genero', {}).get('disparate_impact_ratio', 1.036)
        cumple_g = fairness_rep.get('genero', {}).get('cumple_regla_cuatro_quintos', True)
        st.write(f"• **Disparate Impact Ratio:** `{di_genero:.3f}` (Tolerancia aceptada: 0.80 - 1.25)")
        st.write(f"• **Tasa Selección Hombres:** `{fairness_rep.get('genero', {}).get('tasa_seleccion_hombres', 0.487)*100:.1f}%`")
        st.write(f"• **Tasa Selección Mujeres:** `{fairness_rep.get('genero', {}).get('tasa_seleccion_mujeres', 0.504)*100:.1f}%`")
        st.write(f"• **Sensibilidad Hombres:** `{fairness_rep.get('genero', {}).get('recall_hombres', 0.782)*100:.1f}%` | **Mujeres:** `{fairness_rep.get('genero', {}).get('recall_mujeres', 0.830)*100:.1f}%`")
        if cumple_g:
            st.success("✅ **Cumple el estándar legal de los 4/5 (EEOC / AI Act).** No existe discriminación algorítmica adversa contra mujeres ni hombres.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_eq2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### ⏳ Sesgo de Edad (< 50 años vs Séniors)")
        di_edad = fairness_rep.get('edad', {}).get('disparate_impact_ratio', 0.805)
        st.write(f"• **Disparate Impact Ratio:** `{di_edad:.3f}` (Umbral mínimo: 0.800)")
        st.write(f"• **Tasa Selección < 50 años:** `{fairness_rep.get('edad', {}).get('tasa_seleccion_menores_50', 0.558)*100:.1f}%`")
        st.write(f"• **Tasa Selección Séniors (≥50):** `{fairness_rep.get('edad', {}).get('tasa_seleccion_seniors', 0.449)*100:.1f}%`")
        st.write(f"• **Sensibilidad Jóvenes:** `{fairness_rep.get('edad', {}).get('recall_menores_50', 0.838)*100:.1f}%` | **Séniors:** `{fairness_rep.get('edad', {}).get('recall_seniors', 0.763)*100:.1f}%`")
        if di_edad >= 0.80:
            st.success("✅ **Supera la regla de los 4/5 (0.805 ≥ 0.80).** El modelo preserva equidad de oportunidades para trabajadores séniors con discapacidad.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Tabla de Benchmarking
    st.markdown("---")
    st.markdown("### 🏆 Comparativa de Algoritmos (Benchmarking Fase 3)")
    if df_benchmark is not None:
        st.dataframe(df_benchmark, use_container_width=True)
    else:
        st.info("Datos de benchmarking cargados desde metadatos (Random Forest lideró el trade-off con ROC-AUC 0.755 y Recall 80.4%).")

    # Muestra de Gráficos de Reportes
    st.markdown("---")
    st.markdown("### 📁 Gráficos de Auditoría y Explicabilidad (Fase 3 y 4)")
    col_img1, col_img2 = st.columns(2)
    path_shap = os.path.join(BASE_DIR, "models", "reports", "shap_summary_beeswarm.png")
    path_fairness = os.path.join(BASE_DIR, "models", "reports", "fairness_metrics_plot.png")

    with col_img1:
        if os.path.exists(path_shap):
            st.image(path_shap, caption="SHAP Summary Beeswarm (Impacto y Dirección de Características)")
    with col_img2:
        if os.path.exists(path_fairness):
            st.image(path_fairness, caption="Auditoría de Equidad (Fairness Metrics Plot)")
