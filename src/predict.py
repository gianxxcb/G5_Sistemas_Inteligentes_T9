"""Módulo para inferencia con modelos entrenados y demostración de nuevas predicciones.

Genera los 5 casos hipotéticos plausibles exigidos en la Etapa 15 (Tabla T9).
"""

from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.config import DIRECTORIO_MODELOS, DIRECTORIO_TABLAS


def cargar_pipeline_entrenado(ruta_modelo: Optional[Path] = None) -> object:
    """Carga el flujo entrenado y serializado desde el directorio de modelos."""
    if ruta_modelo is None:
        archivos = sorted(DIRECTORIO_MODELOS.glob("modelo_final_*.joblib"), key=lambda ruta: ruta.stat().st_mtime, reverse=True)
        if not archivos:
            raise FileNotFoundError(f"No se encontraron modelos serializados en: {DIRECTORIO_MODELOS}")
        ruta_modelo = archivos[0]
        
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"Archivo de modelo no encontrado: {ruta_modelo}")
        
    return joblib.load(ruta_modelo)


def generar_cinco_casos_nuevos(escenario: str = "tardio") -> pd.DataFrame:
    """Construye 5 perfiles plausibles y representativos de estudiantes hipotéticos."""
    casos = [
        {
            "Caso": "Caso 1 (Estudiante Destacado)",
            "escuela": "GP", "sexo": "F", "edad": 16, "zona_residencia": "U", "tamano_familia": "LE3",
            "convivencia_padres": "T", "educacion_madre": 4, "educacion_padre": 4, "ocupacion_madre": "health", "ocupacion_padre": "teacher",
            "motivo_escuela": "course", "tutor": "mother", "tiempo_viaje": 1, "tiempo_estudio": 4,
            "asignaturas_reprobadas": 0, "apoyo_escolar": "no", "apoyo_familiar": "yes", "clases_pagadas": "yes",
            "actividades": "yes", "educacion_inicial": "yes", "aspira_educacion_superior": "yes", "acceso_internet": "yes",
            "relacion_sentimental": "no", "relaciones_familiares": 5, "tiempo_libre": 2, "salidas_amigos": 2, "alcohol_laboral": 1,
            "alcohol_fin_semana": 1, "salud": 5, "ausencias": 2, "nota_primer_periodo": 17, "nota_segundo_periodo": 18,
            "Perfil_Cualitativo": "Alto tiempo de estudio, soporte familiar, sin ausencias y calificaciones previas sobresalientes.",
        },
        {
            "Caso": "Caso 2 (Estudiante en Riesgo Académico)",
            "escuela": "MS", "sexo": "M", "edad": 18, "zona_residencia": "R", "tamano_familia": "GT3",
            "convivencia_padres": "A", "educacion_madre": 1, "educacion_padre": 1, "ocupacion_madre": "other", "ocupacion_padre": "other",
            "motivo_escuela": "other", "tutor": "other", "tiempo_viaje": 3, "tiempo_estudio": 1,
            "asignaturas_reprobadas": 2, "apoyo_escolar": "no", "apoyo_familiar": "no", "clases_pagadas": "no",
            "actividades": "no", "educacion_inicial": "no", "aspira_educacion_superior": "no", "acceso_internet": "no",
            "relacion_sentimental": "yes", "relaciones_familiares": 2, "tiempo_libre": 4, "salidas_amigos": 4, "alcohol_laboral": 3,
            "alcohol_fin_semana": 4, "salud": 2, "ausencias": 18, "nota_primer_periodo": 7, "nota_segundo_periodo": 8,
            "Perfil_Cualitativo": "Historial de repitencia, ausentismo alto, bajo estudio semanal y notas previas desaprobatorias.",
        },
        {
            "Caso": "Caso 3 (Estudiante de Rendimiento Promedio)",
            "escuela": "GP", "sexo": "F", "edad": 17, "zona_residencia": "U", "tamano_familia": "GT3",
            "convivencia_padres": "T", "educacion_madre": 2, "educacion_padre": 3, "ocupacion_madre": "services", "ocupacion_padre": "services",
            "motivo_escuela": "reputation", "tutor": "father", "tiempo_viaje": 2, "tiempo_estudio": 2,
            "asignaturas_reprobadas": 0, "apoyo_escolar": "no", "apoyo_familiar": "yes", "clases_pagadas": "no",
            "actividades": "yes", "educacion_inicial": "yes", "aspira_educacion_superior": "yes", "acceso_internet": "yes",
            "relacion_sentimental": "no", "relaciones_familiares": 4, "tiempo_libre": 3, "salidas_amigos": 3, "alcohol_laboral": 1,
            "alcohol_fin_semana": 2, "salud": 4, "ausencias": 4, "nota_primer_periodo": 12, "nota_segundo_periodo": 12,
            "Perfil_Cualitativo": "Parámetros académicos moderados, asistencia regular y calificaciones previas constantes.",
        },
        {
            "Caso": "Caso 4 (Estudiante en Recuperación Notable)",
            "escuela": "GP", "sexo": "M", "edad": 16, "zona_residencia": "U", "tamano_familia": "LE3",
            "convivencia_padres": "T", "educacion_madre": 3, "educacion_padre": 2, "ocupacion_madre": "other", "ocupacion_padre": "services",
            "motivo_escuela": "home", "tutor": "mother", "tiempo_viaje": 1, "tiempo_estudio": 3,
            "asignaturas_reprobadas": 0, "apoyo_escolar": "yes", "apoyo_familiar": "yes", "clases_pagadas": "yes",
            "actividades": "no", "educacion_inicial": "yes", "aspira_educacion_superior": "yes", "acceso_internet": "yes",
            "relacion_sentimental": "no", "relaciones_familiares": 4, "tiempo_libre": 2, "salidas_amigos": 2, "alcohol_laboral": 1,
            "alcohol_fin_semana": 1, "salud": 3, "ausencias": 2, "nota_primer_periodo": 9, "nota_segundo_periodo": 14,
            "Perfil_Cualitativo": "Inicio flojo en nota_primer_periodo pero recuperación sustancial en nota_segundo_periodo gracias a apoyo escolar y estudio guiado.",
        },
        {
            "Caso": "Caso 5 (Estudiante en Declinación y Deserción Potencial)",
            "escuela": "MS", "sexo": "F", "edad": 19, "zona_residencia": "R", "tamano_familia": "GT3",
            "convivencia_padres": "T", "educacion_madre": 2, "educacion_padre": 2, "ocupacion_madre": "at_home", "ocupacion_padre": "other",
            "motivo_escuela": "course", "tutor": "mother", "tiempo_viaje": 3, "tiempo_estudio": 1,
            "asignaturas_reprobadas": 1, "apoyo_escolar": "no", "apoyo_familiar": "no", "clases_pagadas": "no",
            "actividades": "no", "educacion_inicial": "no", "aspira_educacion_superior": "yes", "acceso_internet": "no",
            "relacion_sentimental": "yes", "relaciones_familiares": 3, "tiempo_libre": 4, "salidas_amigos": 5, "alcohol_laboral": 2,
            "alcohol_fin_semana": 3, "salud": 1, "ausencias": 14, "nota_primer_periodo": 11, "nota_segundo_periodo": 6,
            "Perfil_Cualitativo": "Edad superior al promedio, desmejora drástica en el segundo periodo, salud baja y ausentismo.",
        },
    ]
    datos_casos = pd.DataFrame(casos)
    
    if escenario == "temprano":
        # Se eliminan nota_primer_periodo y nota_segundo_periodo si el escenario es temprano
        datos_casos = datos_casos.drop(columns=["nota_primer_periodo", "nota_segundo_periodo"])
        
    return datos_casos


def realizar_predicciones_nuevas(
    flujo: object,
    escenario: str = "tardio",
) -> pd.DataFrame:
    """Ejecuta inferencia directa con el flujo sobre los 5 nuevos casos y genera la Tabla T9."""
    datos_casos = generar_cinco_casos_nuevos(escenario=escenario)
    
    # Extraemos características predictores (quitamos columnas descriptivas de reporte)
    columnas_descriptivas = ["Caso", "Perfil_Cualitativo"]
    predictores_nuevos = datos_casos.drop(columns=columnas_descriptivas).copy()
    
    # Predicción sin reentrenamiento
    predicciones = flujo.predict(predictores_nuevos)
    
    filas_t9 = []
    for i, fila in datos_casos.iterrows():
        nota_pred = float(np.round(predicciones[i], 2))
        estado_aprobacion = "Aprobado" if nota_pred >= 10.0 else "Desaprobado / En Riesgo"
        
        datos_clave = f"Edad={fila['edad']}, Estudio={fila['tiempo_estudio']}, Reprobadas={fila['asignaturas_reprobadas']}, Ausencias={fila['ausencias']}"
        if escenario == "tardio":
            datos_clave += f", nota_primer_periodo={fila['nota_primer_periodo']}, nota_segundo_periodo={fila['nota_segundo_periodo']}"
            
        filas_t9.append({
            "Identificador": fila["Caso"],
            "Factores Clave": datos_clave,
            "Perfil Resumido": fila["Perfil_Cualitativo"],
            "Nota Estimada (nota_final)": nota_pred,
            "Estado Proyectado": estado_aprobacion,
            "Decisión / Acción Recomendada": (
                "Reconocimiento y estímulo académico" if nota_pred >= 14 else
                ("Tutoría preventiva y refuerzo en áreas críticas" if nota_pred >= 10 else
                 "Intervención urgente del departamento psicopedagógico y plan de recuperación")
            ),
        })
        
    datos_t9 = pd.DataFrame(filas_t9)
    datos_t9.to_csv(DIRECTORIO_TABLAS / f"T9_predicciones_nuevas_{escenario}.csv", index=False, encoding="utf-8-sig")
    return datos_t9
