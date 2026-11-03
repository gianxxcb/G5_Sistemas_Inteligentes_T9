"""Módulo de preprocesamiento de datos sin fuga de información.

Implementa ColumnTransformer y flujos que ajustan sus transformaciones
exclusivamente dentro de los pliegues de entrenamiento.
"""

from typing import Dict, List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DIRECTORIO_TABLAS


def crear_preprocesador(
    columnas_numericas: List[str],
    columnas_categoricas: List[str],
) -> ColumnTransformer:
    """Construye un ColumnTransformer estricto para evitar fuga de datos.
    
    - Rama Numérica: Imputación por mediana + Escalado estándar (StandardScaler).
    - Rama Categórica: Imputación por moda + Codificación One-Hot ignorando categorías nuevas.
    """
    flujo_numerico = Pipeline(
        steps=[
            ("imputador_mediana", SimpleImputer(strategy="median")),
            ("escalador_estandar", StandardScaler()),
        ]
    )
    
    flujo_categorico = Pipeline(
        steps=[
            ("imputador_frecuente", SimpleImputer(strategy="most_frequent")),
            (
                "codificador_onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    
    preprocesador = ColumnTransformer(
        transformers=[
            ("num", flujo_numerico, columnas_numericas),
            ("cat", flujo_categorico, columnas_categoricas),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    
    return preprocesador


def generar_tabla_t4_preprocesamiento(
    columnas_num: List[str],
    columnas_cat: List[str],
    escenario: str = "ambos",
) -> pd.DataFrame:
    """Genera la Tabla T4 oficial documentando la estrategia de transformación."""
    filas = [
        {
            "Grupo": "Variables Numéricas",
            "Columnas Incluidas": ", ".join(columnas_num),
            "Cantidad": len(columnas_num),
            "Imputación": "SimpleImputer(strategy='median')",
            "Codificación": "Ninguna (mantiene naturaleza continua/ordinal)",
            "Escalado": "StandardScaler() [media 0, varianza 1]",
            "Justificación Técnica": (
                "La mediana es robusta frente a asimetrías o valores extremos (ej. ausencias). "
                "El escalado es imprescindible para modelos basados en distancias (KNN) y márgenes (SVR)."
            ),
        },
        {
            "Grupo": "Variables Categóricas",
            "Columnas Incluidas": ", ".join(columnas_cat),
            "Cantidad": len(columnas_cat),
            "Imputación": "SimpleImputer(strategy='most_frequent')",
            "Codificación": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "Escalado": "No aplica (las variables dummy son binarias 0/1)",
            "Justificación Técnica": (
                "Permite representar factores cualitativos sin imponer un orden artificial. "
                "El parámetro handle_unknown='ignore' previene caídas ante categorías imprevistas en producción."
            ),
        },
    ]
    t4_df = pd.DataFrame(filas)
    t4_df.to_csv(DIRECTORIO_TABLAS / f"T4_configuracion_preprocesamiento_{escenario}.csv", index=False, encoding="utf-8-sig")
    return t4_df


def verificar_transformacion_datos_no_vistos(
    preprocesador: ColumnTransformer,
    datos_ejemplo_entrenamiento: pd.DataFrame,
) -> bool:
    """Verifica técnicamente que el transformador maneja categorías nuevas o nulos sin romperse."""
    preprocesador.fit(datos_ejemplo_entrenamiento)
    
    # Creamos un registro con valores fuera de dominio
    datos_invalido = datos_ejemplo_entrenamiento.iloc[[0]].copy()
    for col in datos_invalido.select_dtypes(exclude="number").columns:
        datos_invalido[col] = "CATEGORIA_DESCONOCIDA_X"
    for col in datos_invalido.select_dtypes(include="number").columns:
        datos_invalido[col] = None  # Comprobación de imputación
        
    try:
        transformado = preprocesador.transform(datos_invalido)
        assert transformado.shape[0] == 1
        return True
    except Exception as e:
        raise RuntimeError(f"Fallo en la prueba de robustez de preprocesamiento: {e}")
