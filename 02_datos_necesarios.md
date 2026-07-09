# Proyecto: Predicción de la empleabilidad de las personas con discapacidad mediante técnicas de Machine Learning

## 1. Idea seleccionada

### Problema que resuelve

Las personas con discapacidad presentan tasas de empleo inferiores a las
de la población general. Aunque existen numerosos informes, resulta
difícil identificar de forma integrada qué factores influyen en su
empleabilidad. Este proyecto pretende ayudar a comprender dichos
factores para apoyar la toma de decisiones de administraciones y
entidades sociales.

### Solución planteada

Se plantea una solución de Data Science que integre datos oficiales
de ODISMET y la Base Estatal de Datos de Personas con Discapacidad
(IMSERSO). Tras limpiar e integrar los datos, se realizará un análisis
exploratorio y se entrenará un modelo de Machine Learning para estimar
la empleabilidad a partir de variables demográficas y de discapacidad.

### MVP

El producto final consistirá en un dashboard interactivo con
indicadores, gráficos y mapas, además de un modelo predictivo que
permita estimar la empleabilidad y mostrar la importancia de las
variables utilizadas.

## 2. Datos necesarios

### Variables

**Imprescindibles** - Edad - Sexo - Comunidad Autónoma - Tipo y grado de
discapacidad - Situación laboral - Tasas de empleo y paro - Número de
contratos - Año

**Deseables** - Nivel educativo - Sector económico - Tipo de contrato -
Ocupación - Tipo de empresa

### Granularidad

Preferentemente por individuo; alternativamente por comunidad autónoma,
provincia y año.

### Profundidad histórica

Entre 10 y 15 años.

## 3. Fuentes de datos previstas

### ODISMET

- Datos abiertos sobre empleo y discapacidad.
- Formato: Excel.
- Histórico disponible.
- Fuente estable.

### Base Estatal de Datos de Personas con Discapacidad (IMSERSO)

- Microdatos anonimizados.
- Formato: CSV.
- Histórico disponible.
- Fuente oficial.

### Riesgos

- Diferencias metodológicas entre fuentes.
- Variables no disponibles para todos los años.
- Necesidad de integrar varios conjuntos de datos.

## 4. Privacidad

Los datos utilizados serán públicos y anonimizados. No se tratarán datos
personales identificables. El proyecto tendrá fines exclusivamente
académicos y respetará las condiciones de reutilización de la
información pública.

## 5. Viabilidad

El proyecto es viable porque las principales fuentes son oficiales y
públicas, con suficiente calidad y profundidad histórica. El mayor
riesgo es la integración de distintas fuentes de datos. Como alternativa
podrán utilizarse datos del INE, EPA y datos.gob.es.
