"""Módulo de configuración central del proyecto de Machine Learning.

Define rutas absolutas/relativas, semillas de reproducibilidad, nombres de variables
clave, descripciones de atributos y configuraciones de experimentación.
"""

from pathlib import Path

# ==============================================================================
# 1. RUTAS DEL PROYECTO
# ==============================================================================
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

DIRECTORIO_DATOS = RAIZ_PROYECTO / "data"
RUTA_DATOS_ORIGINALES = DIRECTORIO_DATOS / "raw" / "student-por.csv"
RUTA_ZIP_ORIGINAL = DIRECTORIO_DATOS / "raw" / "archive.zip"
DIRECTORIO_DATOS_PROCESADOS = DIRECTORIO_DATOS / "processed"

DIRECTORIO_MODELOS = RAIZ_PROYECTO / "models"
DIRECTORIO_INFORMES = RAIZ_PROYECTO / "reports"
DIRECTORIO_TABLAS = DIRECTORIO_INFORMES / "tables"
DIRECTORIO_FIGURAS = DIRECTORIO_INFORMES / "figures"
DIRECTORIO_NOTEBOOKS = RAIZ_PROYECTO / "notebooks"

# Crear directorios clave si no existen
for directorio in [DIRECTORIO_DATOS_PROCESADOS, DIRECTORIO_MODELOS, DIRECTORIO_TABLAS, DIRECTORIO_FIGURAS, DIRECTORIO_NOTEBOOKS]:
    directorio.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. PARÁMETROS EXPERIMENTALES Y REPRODUCIBILIDAD
# ==============================================================================
SEMILLA_ALEATORIA = 42
PROPORCION_PRUEBA = 0.20
NUMERO_PLIEGUES_CV = 5

COLUMNA_OBJETIVO = "nota_final"
NOTAS_PREVIAS = ["nota_primer_periodo", "nota_segundo_periodo"]

# Metadatos del dataset
METADATOS_DATOS = {
    "nombre": "Student Performance Data Set",
    "archivo_csv": "student-por.csv",
    "fuente_kaggle": "https://www.kaggle.com/datasets/larsen0966/student-performance-data-set",
    "autor_original": "Paulo Cortez, Alice Silva (Universidade do Minho, Portugal)",
    "licencia": "Open Data Commons Open Database License (ODbL) / CC BY 4.0",
    "unidad_analisis": "Un estudiante de secundaria de asignatura de Lengua Portuguesa",
}

# Diccionario completo de descripción de atributos
DESCRIPCIONES_VARIABLES = {
    "escuela": "Escuela del estudiante ('GP' - Gabriel Pereira, 'MS' - Mousinho da Silveira)",
    "sexo": "Sexo del estudiante ('F' - femenino, 'M' - masculino)",
    "edad": "Edad del estudiante (numérico: 15 a 22 años)",
    "zona_residencia": "Tipo de dirección de residencia ('U' - urbana, 'R' - rural)",
    "tamano_familia": "Tamaño de la familia ('LE3' - menor o igual a 3, 'GT3' - mayor a 3)",
    "convivencia_padres": "Estado de cohabitación de los padres ('T' - juntos, 'A' - separados)",
    "educacion_madre": "Nivel de educación de la madre (0: ninguno, 1: primaria, 2: 5to a 9no grado, 3: secundaria, 4: superior)",
    "educacion_padre": "Nivel de educación del padre (0: ninguno, 1: primaria, 2: 5to a 9no grado, 3: secundaria, 4: superior)",
    "ocupacion_madre": "Ocupación de la madre ('teacher', 'health', 'services', 'at_home', 'other')",
    "ocupacion_padre": "Ocupación del padre ('teacher', 'health', 'services', 'at_home', 'other')",
    "motivo_escuela": "Razón para elegir la escuela ('home', 'reputation', 'course', 'other')",
    "tutor": "Tutor legal del estudiante ('mother', 'father', 'other')",
    "tiempo_viaje": "Tiempo de viaje a la escuela (1: <15 min, 2: 15-30 min, 3: 30 min-1 hora, 4: >1 hora)",
    "tiempo_estudio": "Tiempo de estudio semanal (1: <2 horas, 2: 2-5 horas, 3: 5-10 horas, 4: >10 horas)",
    "asignaturas_reprobadas": "Número de asignaturas reprobadas anteriormente (0 a 3, 4 si n >= 4)",
    "apoyo_escolar": "Apoyo educativo escolar adicional ('yes', 'no')",
    "apoyo_familiar": "Apoyo educativo familiar ('yes', 'no')",
    "clases_pagadas": "Clases particulares pagadas de la asignatura ('yes', 'no')",
    "actividades": "Actividades extracurriculares ('yes', 'no')",
    "educacion_inicial": "Asistió a guardería / educación inicial ('yes', 'no')",
    "aspira_educacion_superior": "Desea cursar educación superior ('yes', 'no')",
    "acceso_internet": "Acceso a Internet en el hogar ('yes', 'no')",
    "relacion_sentimental": "En una relación sentimental ('yes', 'no')",
    "relaciones_familiares": "Calidad de relaciones familiares (1: muy mala a 5: excelente)",
    "tiempo_libre": "Tiempo libre después de la escuela (1: muy bajo a 5: muy alto)",
    "salidas_amigos": "Frecuencia de salidas con amigos (1: muy baja a 5: muy alta)",
    "alcohol_laboral": "Consumo de alcohol en días laborales (1: muy bajo a 5: muy alto)",
    "alcohol_fin_semana": "Consumo de alcohol en fines de semana (1: muy bajo a 5: muy alto)",
    "salud": "Estado de salud actual (1: muy malo a 5: muy bueno)",
    "ausencias": "Número de ausencias escolares (0 a 93)",
    "nota_primer_periodo": "Calificación del primer periodo escolar (0 a 20)",
    "nota_segundo_periodo": "Calificación del segundo periodo escolar (0 a 20)",
    "nota_final": "Calificación final del curso (0 a 20) — Variable Objetivo",
}

# ==============================================================================
# 3. REJILLAS DE BÚSQUEDA PARA HIPERPARÁMETROS (GridSearchCV)
# ==============================================================================
# Para KNeighborsRegressor
REJILLA_KNN = {
    "modelo__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "modelo__weights": ["uniform", "distance"],
    "modelo__p": [1, 2],  # 1: Manhattan, 2: Euclidiana
}

# Para SVR
REJILLA_SVR = {
    "modelo__C": [0.1, 1.0, 5.0, 10.0, 20.0],
    "modelo__epsilon": [0.01, 0.1, 0.2, 0.5],
    "modelo__kernel": ["rbf", "linear"],
}

# Se conservan intactos los archivos CSV originales.
COLUMNAS_EN_ESPANOL = {
    "school": "escuela",
    "sex": "sexo",
    "age": "edad",
    "address": "zona_residencia",
    "famsize": "tamano_familia",
    "Pstatus": "convivencia_padres",
    "Medu": "educacion_madre",
    "Fedu": "educacion_padre",
    "Mjob": "ocupacion_madre",
    "Fjob": "ocupacion_padre",
    "reason": "motivo_escuela",
    "guardian": "tutor",
    "traveltime": "tiempo_viaje",
    "studytime": "tiempo_estudio",
    "failures": "asignaturas_reprobadas",
    "schoolsup": "apoyo_escolar",
    "famsup": "apoyo_familiar",
    "paid": "clases_pagadas",
    "activities": "actividades",
    "nursery": "educacion_inicial",
    "higher": "aspira_educacion_superior",
    "internet": "acceso_internet",
    "romantic": "relacion_sentimental",
    "famrel": "relaciones_familiares",
    "freetime": "tiempo_libre",
    "goout": "salidas_amigos",
    "Dalc": "alcohol_laboral",
    "Walc": "alcohol_fin_semana",
    "health": "salud",
    "absences": "ausencias",
    "G1": "nota_primer_periodo",
    "G2": "nota_segundo_periodo",
    "G3": "nota_final"
}
