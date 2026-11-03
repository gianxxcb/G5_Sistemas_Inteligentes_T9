"""Paquete modular de Machine Learning para la estimación de notas estudiantiles (nota_final)."""

from src.config import (
    DIRECTORIO_DATOS,
    DIRECTORIO_FIGURAS,
    DIRECTORIO_MODELOS,
    SEMILLA_ALEATORIA,
    RUTA_DATOS_ORIGINALES,
    DIRECTORIO_INFORMES,
    DIRECTORIO_TABLAS,
    COLUMNA_OBJETIVO,
    DESCRIPCIONES_VARIABLES,
)
from src.data_loader import (
    calcular_sha256,
    cargar_datos,
    clasificar_columnas,
    construir_escenarios,
    obtener_particiones_escenario,
    realizar_particion_datos,
)
from src.eda import (
    generar_graficos_eda,
    generar_tabla_t1_diccionario,
    generar_tabla_t2_calidad,
)
from src.evaluation import (
    evaluar_test_final,
    evaluar_validacion_cruzada,
    generar_tabla_t7_decision,
    optimizar_hiperparametros,
)
from src.models import (
    ensamblar_pipelines,
    generar_tabla_fundamentos_modelos,
    obtener_estimadores_base,
)
from src.predict import (
    cargar_pipeline_entrenado,
    generar_cinco_casos_nuevos,
    realizar_predicciones_nuevas,
)
from src.preprocessing import (
    crear_preprocesador,
    generar_tabla_t4_preprocesamiento,
    verificar_transformacion_datos_no_vistos,
)

__all__ = [
    "DIRECTORIO_DATOS",
    "DIRECTORIO_FIGURAS",
    "DIRECTORIO_MODELOS",
    "SEMILLA_ALEATORIA",
    "RUTA_DATOS_ORIGINALES",
    "DIRECTORIO_INFORMES",
    "DIRECTORIO_TABLAS",
    "COLUMNA_OBJETIVO",
    "DESCRIPCIONES_VARIABLES",
    "calcular_sha256",
    "cargar_datos",
    "clasificar_columnas",
    "construir_escenarios",
    "obtener_particiones_escenario",
    "realizar_particion_datos",
    "generar_graficos_eda",
    "generar_tabla_t1_diccionario",
    "generar_tabla_t2_calidad",
    "evaluar_test_final",
    "evaluar_validacion_cruzada",
    "generar_tabla_t7_decision",
    "optimizar_hiperparametros",
    "ensamblar_pipelines",
    "generar_tabla_fundamentos_modelos",
    "obtener_estimadores_base",
    "cargar_pipeline_entrenado",
    "generar_cinco_casos_nuevos",
    "realizar_predicciones_nuevas",
    "crear_preprocesador",
    "generar_tabla_t4_preprocesamiento",
    "verificar_transformacion_datos_no_vistos",
]
