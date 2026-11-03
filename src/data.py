"""Carga, verificación, descripción y valores del formulario de estudiantes."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CSV = RAIZ / "data" / "raw" / "student-por.csv"
OBJETIVO = "nota_final"
NOTAS_PREVIAS = ["nota_primer_periodo", "nota_segundo_periodo"]
FUENTE_KAGGLE = "https://www.kaggle.com/datasets/larsen0966/student-performance-data-set"
FUENTE_UCI = "https://archive.ics.uci.edu/dataset/320/student+performance"
LICENCIA = "Creative Commons Attribution 4.0 International (CC BY 4.0)"

COLUMNAS_EN_ESPANOL = {'school': 'escuela',
 'sex': 'sexo',
 'age': 'edad',
 'address': 'zona_residencia',
 'famsize': 'tamano_familia',
 'Pstatus': 'convivencia_padres',
 'Medu': 'educacion_madre',
 'Fedu': 'educacion_padre',
 'Mjob': 'ocupacion_madre',
 'Fjob': 'ocupacion_padre',
 'reason': 'motivo_escuela',
 'guardian': 'tutor',
 'traveltime': 'tiempo_viaje',
 'studytime': 'tiempo_estudio',
 'failures': 'asignaturas_reprobadas',
 'schoolsup': 'apoyo_escolar',
 'famsup': 'apoyo_familiar',
 'paid': 'clases_pagadas',
 'activities': 'actividades',
 'nursery': 'educacion_inicial',
 'higher': 'aspira_educacion_superior',
 'internet': 'acceso_internet',
 'romantic': 'relacion_sentimental',
 'famrel': 'relaciones_familiares',
 'freetime': 'tiempo_libre',
 'goout': 'salidas_amigos',
 'Dalc': 'alcohol_laboral',
 'Walc': 'alcohol_fin_semana',
 'health': 'salud',
 'absences': 'ausencias',
 'G1': 'nota_primer_periodo',
 'G2': 'nota_segundo_periodo',
 'G3': 'nota_final'}

DESCRIPCIONES_VARIABLES = {'escuela': "Escuela del estudiante ('GP' - Gabriel Pereira, 'MS' - Mousinho da Silveira)",
 'sexo': "Sexo del estudiante ('F' - femenino, 'M' - masculino)",
 'edad': 'Edad del estudiante (numérico: 15 a 22 años)',
 'zona_residencia': "Tipo de dirección de residencia ('U' - urbana, 'R' - rural)",
 'tamano_familia': "Tamaño de la familia ('LE3' - menor o igual a 3, 'GT3' - mayor a 3)",
 'convivencia_padres': "Estado de cohabitación de los padres ('T' - juntos, 'A' - separados)",
 'educacion_madre': 'Nivel de educación de la madre (0: ninguno, 1: primaria, 2: 5to a 9no grado, 3: '
                    'secundaria, 4: superior)',
 'educacion_padre': 'Nivel de educación del padre (0: ninguno, 1: primaria, 2: 5to a 9no grado, 3: '
                    'secundaria, 4: superior)',
 'ocupacion_madre': "Ocupación de la madre ('teacher', 'health', 'services', 'at_home', 'other')",
 'ocupacion_padre': "Ocupación del padre ('teacher', 'health', 'services', 'at_home', 'other')",
 'motivo_escuela': "Razón para elegir la escuela ('home', 'reputation', 'course', 'other')",
 'tutor': "Tutor legal del estudiante ('mother', 'father', 'other')",
 'tiempo_viaje': 'Tiempo de viaje a la escuela (1: <15 min, 2: 15-30 min, 3: 30 min-1 hora, 4: >1 hora)',
 'tiempo_estudio': 'Tiempo de estudio semanal (1: <2 horas, 2: 2-5 horas, 3: 5-10 horas, 4: >10 horas)',
 'asignaturas_reprobadas': 'Número de asignaturas reprobadas anteriormente (0 a 3, 4 si n >= 4)',
 'apoyo_escolar': "Apoyo educativo escolar adicional ('yes', 'no')",
 'apoyo_familiar': "Apoyo educativo familiar ('yes', 'no')",
 'clases_pagadas': "Clases particulares pagadas de la asignatura ('yes', 'no')",
 'actividades': "Actividades extracurriculares ('yes', 'no')",
 'educacion_inicial': "Asistió a guardería / educación inicial ('yes', 'no')",
 'aspira_educacion_superior': "Desea cursar educación superior ('yes', 'no')",
 'acceso_internet': "Acceso a Internet en el hogar ('yes', 'no')",
 'relacion_sentimental': "En una relación sentimental ('yes', 'no')",
 'relaciones_familiares': 'Calidad de relaciones familiares (1: muy mala a 5: excelente)',
 'tiempo_libre': 'Tiempo libre después de la escuela (1: muy bajo a 5: muy alto)',
 'salidas_amigos': 'Frecuencia de salidas con amigos (1: muy baja a 5: muy alta)',
 'alcohol_laboral': 'Consumo de alcohol en días laborales (1: muy bajo a 5: muy alto)',
 'alcohol_fin_semana': 'Consumo de alcohol en fines de semana (1: muy bajo a 5: muy alto)',
 'salud': 'Estado de salud actual (1: muy malo a 5: muy bueno)',
 'ausencias': 'Número de ausencias escolares (0 a 93)',
 'nota_primer_periodo': 'Calificación del primer periodo escolar (0 a 20)',
 'nota_segundo_periodo': 'Calificación del segundo periodo escolar (0 a 20)',
 'nota_final': 'Calificación final del curso (0 a 20) — Variable Objetivo'}

ETIQUETAS = {'escuela': 'Escuela',
 'sexo': 'Sexo',
 'edad': 'Edad',
 'zona_residencia': 'Zona de residencia',
 'tamano_familia': 'Tamaño de la familia',
 'convivencia_padres': 'Convivencia de los padres',
 'educacion_madre': 'Educación de la madre',
 'educacion_padre': 'Educación del padre',
 'ocupacion_madre': 'Ocupación de la madre',
 'ocupacion_padre': 'Ocupación del padre',
 'motivo_escuela': 'Motivo de elección de escuela',
 'tutor': 'Tutor del estudiante',
 'tiempo_viaje': 'Tiempo de viaje a la escuela',
 'tiempo_estudio': 'Estudio semanal',
 'asignaturas_reprobadas': 'Asignaturas reprobadas',
 'apoyo_escolar': 'Apoyo educativo escolar',
 'apoyo_familiar': 'Apoyo educativo familiar',
 'clases_pagadas': 'Clases particulares',
 'actividades': 'Actividades extracurriculares',
 'educacion_inicial': 'Asistió a educación inicial',
 'aspira_educacion_superior': 'Aspira a educación superior',
 'acceso_internet': 'Internet en el hogar',
 'relacion_sentimental': 'Relación sentimental',
 'relaciones_familiares': 'Relaciones familiares',
 'tiempo_libre': 'Tiempo libre',
 'salidas_amigos': 'Salidas con amigos',
 'alcohol_laboral': 'Alcohol en días laborales',
 'alcohol_fin_semana': 'Alcohol en fin de semana',
 'salud': 'Estado de salud',
 'ausencias': 'Ausencias escolares',
 'nota_primer_periodo': 'Nota del primer periodo',
 'nota_segundo_periodo': 'Nota del segundo periodo'}

ESCALAS = {'tiempo_estudio': [('Menos de 2 horas', 1),
                    ('Entre 2 y 5 horas', 2),
                    ('Entre 5 y 10 horas', 3),
                    ('Más de 10 horas', 4)],
 'tiempo_viaje': [('Menos de 15 minutos', 1),
                  ('Entre 15 y 30 minutos', 2),
                  ('Entre 30 y 60 minutos', 3),
                  ('Más de 60 minutos', 4)],
 'educacion_madre': [('Sin educación formal', 0),
                     ('Educación primaria', 1),
                     ('De 5.º a 9.º grado', 2),
                     ('Educación secundaria', 3),
                     ('Educación superior', 4)],
 'educacion_padre': [('Sin educación formal', 0),
                     ('Educación primaria', 1),
                     ('De 5.º a 9.º grado', 2),
                     ('Educación secundaria', 3),
                     ('Educación superior', 4)],
 'salud': [('1 · Muy mala', 1), ('2 · Mala', 2), ('3 · Regular', 3), ('4 · Buena', 4), ('5 · Muy buena', 5)],
 'relaciones_familiares': [('1 · Muy malas', 1),
                           ('2 · Malas', 2),
                           ('3 · Regulares', 3),
                           ('4 · Buenas', 4),
                           ('5 · Excelentes', 5)]}

for nombre in ["tiempo_libre", "salidas_amigos", "alcohol_laboral", "alcohol_fin_semana"]:
    ESCALAS[nombre] = [(f"{numero} - {nivel}", numero) for numero, nivel in
                      enumerate(["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"], 1)]

TRADUCCIONES = {
    "yes": "Sí", "no": "No", "teacher": "Docencia", "health": "Salud",
    "services": "Servicios", "at_home": "Trabajo en el hogar", "other": "Otro",
    "mother": "Madre", "father": "Padre", "home": "Cercanía al hogar", "course": "Oferta académica",
    "reputation": "Reputación", "U": "Urbana", "R": "Rural", "F": "Femenino", "M": "Masculino",
    "T": "Juntos", "A": "Separados", "LE3": "Hasta 3 personas", "GT3": "Más de 3 personas",
    "GP": "Gabriel Pereira", "MS": "Mousinho da Silveira",
}

SECCIONES = {
    "Académico": {"titulo": "Rendimiento y acompañamiento",
        "descripcion": "Notas previas, hábitos de estudio y apoyos disponibles.",
        "columnas": ["nota_primer_periodo", "nota_segundo_periodo", "tiempo_estudio",
                     "asignaturas_reprobadas", "ausencias", "apoyo_escolar", "apoyo_familiar",
                     "clases_pagadas", "aspira_educacion_superior"]},
    "Estudiante": {"titulo": "Perfil del estudiante",
        "descripcion": "Información personal y condiciones de acceso a la escuela.",
        "columnas": ["escuela", "edad", "sexo", "zona_residencia", "tiempo_viaje",
                     "motivo_escuela", "tutor", "educacion_inicial"]},
    "Familia": {"titulo": "Entorno familiar",
        "descripcion": "Composición del hogar, educación y ocupación de los padres.",
        "columnas": ["tamano_familia", "convivencia_padres", "educacion_madre", "educacion_padre",
                     "ocupacion_madre", "ocupacion_padre", "relaciones_familiares", "acceso_internet"]},
    "Bienestar": {"titulo": "Rutina y bienestar",
        "descripcion": "Actividades, relaciones y hábitos fuera del aula.",
        "columnas": ["actividades", "tiempo_libre", "salidas_amigos", "relacion_sentimental",
                     "alcohol_laboral", "alcohol_fin_semana", "salud"]},
}

NUMERICAS = ["edad", "educacion_madre", "educacion_padre", "tiempo_viaje", "tiempo_estudio",
             "asignaturas_reprobadas", "relaciones_familiares", "tiempo_libre", "salidas_amigos",
             "alcohol_laboral", "alcohol_fin_semana", "salud", "ausencias", *NOTAS_PREVIAS, OBJETIVO]


def calcular_sha256(ruta=RUTA_CSV):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def cargar_datos(ruta=RUTA_CSV):
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró {ruta}. Consulta la fuente de datos en README.md.")
    datos = pd.read_csv(ruta, sep=None, engine="python").rename(columns=COLUMNAS_EN_ESPANOL)
    if datos.empty:
        raise ValueError("El CSV no contiene estudiantes.")
    if datos.columns.duplicated().any():
        raise ValueError("El CSV contiene columnas duplicadas después de traducir sus nombres.")
    faltantes = set(COLUMNAS_EN_ESPANOL.values()) - set(datos.columns)
    if faltantes:
        raise ValueError("Faltan columnas: " + ", ".join(sorted(faltantes)))
    datos = datos[list(COLUMNAS_EN_ESPANOL.values())].copy()
    for columna in NUMERICAS:
        datos[columna] = pd.to_numeric(datos[columna], errors="raise")
        if not np.isfinite(datos[columna].dropna()).all():
            raise ValueError(f"La columna {columna} contiene valores infinitos.")
    if datos[OBJETIVO].isna().any() or not datos[OBJETIVO].between(0, 20).all():
        raise ValueError("La nota final debe contener valores entre 0 y 20, sin nulos.")
    vacias = datos.columns[datos.isna().all()].tolist()
    if vacias:
        raise ValueError("Columnas completamente vacías: " + ", ".join(vacias))
    return datos


def describir_datos(datos):
    return pd.DataFrame([{
        "Variable": columna,
        "Descripción": DESCRIPCIONES_VARIABLES[columna],
        "Tipo": "Numérica" if columna in NUMERICAS else "Categórica",
        "Nulos": int(datos[columna].isna().sum()),
        "Valores únicos": int(datos[columna].nunique()),
        "Rol": "Objetivo" if columna == OBJETIVO else "Predictor",
    } for columna in datos.columns])


def valores_iniciales(datos):
    perfil = {}
    for columna in datos.columns.drop(OBJETIVO):
        serie = datos[columna]
        perfil[columna] = int(serie.median()) if columna in NUMERICAS else str(serie.mode().iloc[0])
    return perfil


def opciones_columna(columna, datos):
    if columna in ESCALAS:
        return ESCALAS[columna]
    return [(TRADUCCIONES.get(str(valor), str(valor)), str(valor))
            for valor in sorted(datos[columna].dropna().unique())]
