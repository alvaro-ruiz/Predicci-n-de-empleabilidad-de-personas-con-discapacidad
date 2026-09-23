#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFM: Predicción de la empleabilidad de personas con discapacidad
Script de Unificación y Preprocesamiento de Microdatos (INE EDAD 2020)
---------------------------------------------------------------------
Este script realiza las siguientes tareas:
1. Carga los microdatos de personas con discapacidad (EDADdiscapacidad_2020.csv)
   y los datos del censo de hogares (EDADhogar_2020.csv).
2. Cruza ambos datasets mediante las claves de hogar y orden de individuo.
3. Filtra la población en edad de trabajar (16 a 64 años).
4. Construye la variable objetivo de empleabilidad ('empleado': 0/1) y la
   situación laboral detallada ('situacion_laboral': Ocupado, Desempleado, Inactivo).
5. Mapea y limpia variables demográficas, geográficas, educativas, socioeconómicas
   y de severidad/tipo de discapacidad.
6. Exporta el dataset final procesado a 'Data/processed/dataset_empleabilidad_edad2020.csv'.
"""

import os
import sys
import numpy as np
import pandas as pd

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "Data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DISC_FILE = os.path.join(RAW_DIR, "EDADdiscapacidad_2020.csv")
HOGAR_FILE = os.path.join(RAW_DIR, "EDADhogar_2020.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset_empleabilidad_edad2020.csv")

# =============================================================================
# 2. DICCIONARIOS DE MAPEO SEGÚN METODOLOGÍA INE (EDAD 2020)
# =============================================================================
MAPEO_CCAA = {
    '01': 'Andalucía',
    '02': 'Aragón',
    '03': 'Asturias',
    '04': 'Baleares',
    '05': 'Canarias',
    '06': 'Cantabria',
    '07': 'Castilla y León',
    '08': 'Castilla-La Mancha',
    '09': 'Cataluña',
    '10': 'Comunidad Valenciana',
    '11': 'Extremadura',
    '12': 'Galicia',
    '13': 'Comunidad de Madrid',
    '14': 'Región de Murcia',
    '15': 'Navarra',
    '16': 'País Vasco',
    '17': 'La Rioja',
    '18': 'Ceuta',
    '19': 'Melilla'
}

MAPEO_TMUNI = {
    '1': 'Más de 500.000 hab',
    '2': '100.000 a 499.999 hab',
    '3': '50.000 a 99.999 hab',
    '4': '20.000 a 49.999 hab',
    '5': '10.000 a 19.999 hab',
    '6': 'Menos de 10.000 hab'
}

MAPEO_SEXO = {
    '1': 'Hombre',
    '2': 'Mujer'
}

MAPEO_ESTUDIO = {
    '01': 'Menores de 16 años',
    '02': 'Analfabetos',
    '03': 'Sin estudios / Primaria incompleta',
    '04': 'Educación Primaria',
    '05': 'Educación Secundaria (1ª etapa)',
    '06': 'Bachillerato / 2ª etapa ESO',
    '07': 'FP Grado Medio',
    '08': 'FP Grado Superior',
    '09': 'Grado Universitario o equivalente',
    '10': 'Máster o Doctorado'
}

MAPEO_NIVEL_EDUCATIVO_AGRUPADO = {
    '02': 'Sin estudios / Primaria',
    '03': 'Sin estudios / Primaria',
    '04': 'Sin estudios / Primaria',
    '05': 'Secundaria / Bachillerato',
    '06': 'Secundaria / Bachillerato',
    '07': 'Formación Profesional',
    '08': 'Formación Profesional',
    '09': 'Universidad / Posgrado',
    '10': 'Universidad / Posgrado'
}

MAPEO_INGRESO = {
    '01': 'Menos de 500 €',
    '02': '500 a 999 €',
    '03': '1.000 a 1.499 €',
    '04': '1.500 a 1.999 €',
    '05': '2.000 a 2.499 €',
    '06': '2.500 a 2.999 €',
    '07': '3.000 a 4.999 €',
    '08': '5.000 € o más'
}

MAPEO_TIPOHOG = {
    '1': 'Hogar unipersonal',
    '2': 'Pareja sin hijos',
    '3': 'Pareja con hijos',
    '4': 'Monoparental',
    '5': 'Otros tipos de hogar'
}

MAPEO_ECIVIL = {
    '1': 'Soltero/a',
    '2': 'Casado/a',
    '3': 'Separado/a o Divorciado/a',
    '4': 'Viudo/a'
}


def clean_code(val):
    """Limpia cadenas de texto numéricas eliminando comillas y espacios."""
    if pd.isna(val):
        return ""
    val_str = str(val).strip().strip('"').strip("'")
    return val_str.zfill(2) if len(val_str) == 1 and val_str.isdigit() else val_str


def clasificar_situacion_laboral(row):
    """
    Clasifica la situación laboral a partir de las preguntas de empleo del INE:
    - G_0: ¿Trabajó la semana pasada al menos una hora? (1 = Sí, 6 = No)
    - G_1: ¿Tenía un trabajo del que estuvo ausente? (1 = Sí, 6 = No)
    - G_2: ¿Ha buscado trabajo en las últimas 4 semanas? (1 = Sí, 6 = No)
    """
    g0 = str(row.get('G_0', '')).strip().strip('"')
    g1 = str(row.get('G_1', '')).strip().strip('"')
    g2 = str(row.get('G_2', '')).strip().strip('"')

    if g0 == '1' or g1 == '1':
        return 'Ocupado'
    elif g2 == '1':
        return 'Desempleado'
    else:
        return 'Inactivo'


def categorizar_grado(grado_pct, k2_val):
    """Agrupa el grado de discapacidad reconocido en tramos estándar."""
    k2 = str(k2_val).strip().strip('"')
    if k2 != '1' or pd.isna(grado_pct):
        return 'Sin grado reconocido / No consta'
    try:
        val = float(grado_pct)
        if val < 33:
            return 'Menos del 33%'
        elif val < 65:
            return 'De 33% a 64% (Moderado)'
        elif val < 75:
            return 'De 65% a 74% (Severo)'
        else:
            return '75% o más (Muy severo / Gran invalidez)'
    except (ValueError, TypeError):
        return 'Sin grado reconocido / No consta'


# =============================================================================
# 3. EJECUCIÓN DEL PIPELINE
# =============================================================================
def main():
    print("=" * 70)
    print("Iniciando procesamiento de microdatos EDAD 2020 (INE)")
    print("=" * 70)

    # 1. Comprobar existencia de archivos
    for path, desc in [(DISC_FILE, "Discapacidad"), (HOGAR_FILE, "Hogar")]:
        if not os.path.exists(path):
            print(f"ERROR: No se encuentra el archivo de {desc} en: {path}")
            sys.exit(1)

    # 2. Cargar Cuestionario de Discapacidad
    print(f"\n[1/5] Cargando microdatos de discapacidad: {os.path.basename(DISC_FILE)}...")
    df_disc = pd.read_csv(
        DISC_FILE,
        sep="\t",
        dtype=str,
        low_memory=False,
        encoding="utf-8"
    )
    print(f"      -> Registros cargados: {len(df_disc):,}")
    print(f"      -> Total columnas: {df_disc.shape[1]}")

    # 3. Cargar Cuestionario de Hogar (solo columnas necesarias para optimizar memoria)
    print(f"\n[2/5] Cargando microdatos del hogar: {os.path.basename(HOGAR_FILE)}...")
    cols_hogar_necesarias = [
        'IDENTHOGAR', 'NORDEN', 'ESTUDIO', 'ECIVIL', 'NACIO',
        'TIPOHOG', 'NUMPERHOGAR', 'INGRESO', 'INTERN', 'ORDENAD',
        'IN_TRAB', 'IN_PENS'
    ]

    df_hogar = pd.read_csv(
        HOGAR_FILE,
        sep="\t",
        usecols=cols_hogar_necesarias,
        dtype=str,
        low_memory=False,
        encoding="utf-8"
    )
    print(f"      -> Registros de hogar cargados: {len(df_hogar):,}")

    # Limpiar identificadores para asegurar un cruce perfecto
    df_disc['IDENTHOGAR'] = df_disc['IDENTHOGAR'].astype(str).str.strip().str.strip('"')
    df_disc['NORDEN'] = df_disc['NORDEN'].astype(str).str.strip().str.strip('"')
    df_hogar['IDENTHOGAR'] = df_hogar['IDENTHOGAR'].astype(str).str.strip().str.strip('"')
    df_hogar['NORDEN'] = df_hogar['NORDEN'].astype(str).str.strip().str.strip('"')

    # 4. Cruce (Merge) entre Discapacidad y Hogar
    print("\n[3/5] Cruzando datos individuales con contexto del hogar...")
    df_merged = pd.merge(
        df_disc,
        df_hogar,
        on=['IDENTHOGAR', 'NORDEN'],
        how='left'
    )
    print(f"      -> Registros tras el cruce: {len(df_merged):,}")

    # 5. Filtrar población en edad laboral (16 a 64 años)
    print("\n[4/5] Filtrando población en edad laboral (16 - 64 años) y limpiando variables...")
    df_merged['EDAD_num'] = pd.to_numeric(df_merged['EDAD'].astype(str).str.strip().str.strip('"'), errors='coerce')
    df_laboral = df_merged[(df_merged['EDAD_num'] >= 16) & (df_merged['EDAD_num'] <= 64)].copy()
    print(f"      -> Personas con discapacidad en edad de trabajar: {len(df_laboral):,}")

    # Construcción de la Variable Objetivo:
    # 1: Empleado / Ocupado
    # 0: No empleado (Desempleado activo o Inactivo)
    g0_clean = df_laboral['G_0'].astype(str).str.strip().str.strip('"')
    g1_clean = df_laboral['G_1'].astype(str).str.strip().str.strip('"')
    df_laboral['empleado'] = ((g0_clean == '1') | (g1_clean == '1')).astype(int)
    df_laboral['situacion_laboral'] = df_laboral.apply(clasificar_situacion_laboral, axis=1)

    # Variables Demográficas y Geográficas
    df_laboral['ccaa_codigo'] = df_laboral['CCAA'].apply(clean_code)
    df_laboral['ccaa_nombre'] = df_laboral['ccaa_codigo'].map(MAPEO_CCAA).fillna('Desconocida')

    df_laboral['sexo_codigo'] = df_laboral['SEXO'].astype(str).str.strip().str.strip('"')
    df_laboral['sexo_nombre'] = df_laboral['sexo_codigo'].map(MAPEO_SEXO).fillna('No consta')

    df_laboral['tmuni_codigo'] = df_laboral['TMUNI'].astype(str).str.strip().str.strip('"')
    df_laboral['tamano_municipio'] = df_laboral['tmuni_codigo'].map(MAPEO_TMUNI).fillna('No consta')

    df_laboral['tramo_edad'] = pd.cut(
        df_laboral['EDAD_num'],
        bins=[15, 24, 34, 44, 54, 64],
        labels=['16-24 años', '25-34 años', '35-44 años', '45-54 años', '55-64 años']
    )

    # Nivel de Estudios
    df_laboral['estudio_codigo'] = df_laboral['ESTUDIO'].apply(clean_code)
    df_laboral['nivel_estudios'] = df_laboral['estudio_codigo'].map(MAPEO_ESTUDIO).fillna('No consta')
    df_laboral['nivel_estudios_agrupado'] = df_laboral['estudio_codigo'].map(MAPEO_NIVEL_EDUCATIVO_AGRUPADO).fillna('No consta')

    # Grado de Discapacidad
    df_laboral['grado_pct_num'] = pd.to_numeric(df_laboral['K_3'].astype(str).str.strip().str.strip('"'), errors='coerce')
    df_laboral['certificado_oficial'] = (df_laboral['K_2'].astype(str).str.strip().str.strip('"') == '1').astype(int)
    df_laboral['tramo_grado_discapacidad'] = df_laboral.apply(
        lambda r: categorizar_grado(r['grado_pct_num'], r['K_2']),
        axis=1
    )

    # Limitaciones Funcionales Específicas (1 = Dificultad severa / absoluta, 0 = No / Leve)
    def es_limitacion_grave(val):
        val_clean = str(val).strip().strip('"')
        return 1 if val_clean == '1' else 0

    df_laboral['lim_vision'] = df_laboral.get('VISI_1_1', '6').apply(es_limitacion_grave)
    df_laboral['lim_audicion'] = df_laboral.get('AUDI_5_1', '6').apply(es_limitacion_grave)
    df_laboral['lim_comunicacion'] = (
        (df_laboral.get('COMU_8_1', '6').apply(es_limitacion_grave) == 1) |
        (df_laboral.get('COMU_9_1', '6').apply(es_limitacion_grave) == 1)
    ).astype(int)
    df_laboral['lim_aprendizaje'] = df_laboral.get('APRE_15_1', '6').apply(es_limitacion_grave)
    df_laboral['lim_movilidad'] = (
        (df_laboral.get('MOVI_19_1', '6').apply(es_limitacion_grave) == 1) |
        (df_laboral.get('MOVI_20_1', '6').apply(es_limitacion_grave) == 1)
    ).astype(int)
    df_laboral['lim_autocuidado'] = (
        (df_laboral.get('AUTO_27_1', '6').apply(es_limitacion_grave) == 1) |
        (df_laboral.get('AUTO_28_1', '6').apply(es_limitacion_grave) == 1)
    ).astype(int)

    # Conteo de tipos de limitaciones graves
    df_laboral['num_limitaciones_graves'] = (
        df_laboral['lim_vision'] +
        df_laboral['lim_audicion'] +
        df_laboral['lim_comunicacion'] +
        df_laboral['lim_aprendizaje'] +
        df_laboral['lim_movilidad'] +
        df_laboral['lim_autocuidado']
    )

    # Contexto del Hogar
    df_laboral['ingreso_codigo'] = df_laboral['INGRESO'].apply(clean_code)
    df_laboral['ingresos_hogar'] = df_laboral['ingreso_codigo'].map(MAPEO_INGRESO).fillna('No consta')

    df_laboral['tipo_hogar'] = df_laboral['TIPOHOG'].astype(str).str.strip().str.strip('"').map(MAPEO_TIPOHOG).fillna('Otros tipos de hogar')
    df_laboral['tamano_hogar'] = pd.to_numeric(df_laboral['NUMPERHOGAR'].astype(str).str.strip().str.strip('"'), errors='coerce').fillna(1)

    df_laboral['estado_civil'] = df_laboral['ECIVIL'].astype(str).str.strip().str.strip('"').map(MAPEO_ECIVIL).fillna('No consta')
    df_laboral['tiene_internet'] = (df_laboral['INTERN'].astype(str).str.strip().str.strip('"') == '1').astype(int)
    df_laboral['tiene_ordenador'] = (df_laboral['ORDENAD'].astype(str).str.strip().str.strip('"') == '1').astype(int)

    # Factor de elevación muestral
    df_laboral['factor_elevacion'] = pd.to_numeric(
        df_laboral['FACTOR'].astype(str).str.strip().str.strip('"').str.replace(',', '.'),
        errors='coerce'
    ).fillna(1.0)

    # Selección y Ordenación de Columnas Finales
    columnas_finales = [
        # Identificadores
        'IDENTHOGAR', 'NORDEN',
        # Target
        'empleado', 'situacion_laboral',
        # Demográficas
        'EDAD_num', 'tramo_edad', 'sexo_nombre', 'ccaa_nombre', 'tamano_municipio',
        # Formación
        'nivel_estudios', 'nivel_estudios_agrupado',
        # Discapacidad
        'certificado_oficial', 'grado_pct_num', 'tramo_grado_discapacidad',
        'lim_vision', 'lim_audicion', 'lim_comunicacion',
        'lim_aprendizaje', 'lim_movilidad', 'lim_autocuidado',
        'num_limitaciones_graves',
        # Hogar y Contexto
        'estado_civil', 'tipo_hogar', 'tamano_hogar', 'ingresos_hogar',
        'tiene_internet', 'tiene_ordenador',
        # Ponderación
        'factor_elevacion'
    ]

    df_final = df_laboral[columnas_finales].rename(columns={
        'EDAD_num': 'edad',
        'sexo_nombre': 'sexo',
        'ccaa_nombre': 'ccaa',
        'grado_pct_num': 'grado_discapacidad_pct'
    })

    # 6. Guardar Dataset Procesado
    print(f"\n[5/5] Exportando dataset procesado a: {OUTPUT_FILE}...")
    df_final.to_csv(OUTPUT_FILE, index=False, sep=";", decimal=",", encoding="utf-8-sig")

    # =============================================================================
    # RESUMEN ESTADÍSTICO PARA EL USUARIO
    # =============================================================================
    print("\n" + "=" * 70)
    print("¡PROCESO COMPLETADO CON ÉXITO!")
    print("=" * 70)
    print(f"Dimensiones del dataset final: {df_final.shape[0]:,} filas x {df_final.shape[1]} columnas")
    print("\nDistribución de la Variable Objetivo ('empleado'):")
    dist_target = df_final['empleado'].value_counts()
    pct_target = df_final['empleado'].value_counts(normalize=True) * 100
    for val, count in dist_target.items():
        etiqueta = "Empleado (1)" if val == 1 else "No empleado (0)"
        print(f"  - {etiqueta:17s}: {count:,} ({pct_target[val]:.2f}%)")

    print("\nDesglose por Situación Laboral detallada:")
    for sit, count in df_final['situacion_laboral'].value_counts().items():
        pct = (count / len(df_final)) * 100
        print(f"  - {sit:17s}: {count:,} ({pct:.2f}%)")

    print("\nDesglose por Nivel de Estudios:")
    for est, count in df_final['nivel_estudios_agrupado'].value_counts().items():
        pct = (count / len(df_final)) * 100
        print(f"  - {est:28s}: {count:,} ({pct:.2f}%)")

    print(f"\nArchivo generado: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
