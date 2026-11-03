"""Módulo para carga, verificación de integridad y partición de datos.

Garantiza la ausencia de fuga de información y la reproducibilidad técnica.
"""

from pathlib import Path
import hashlib
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    NOTAS_PREVIAS,
    DIRECTORIO_DATOS_PROCESADOS,
    SEMILLA_ALEATORIA,
    RUTA_DATOS_ORIGINALES,
    COLUMNA_OBJETIVO,
    PROPORCION_PRUEBA,
)


def calcular_sha256(ruta_archivo: Path) -> str:
    """Calcula la huella criptográfica SHA-256 de un archivo para control de integridad."""
    if not ruta_archivo.exists():
        raise FileNotFoundError(f"No existe el archivo especificado: {ruta_archivo}")
    
    huella_sha256 = hashlib.sha256()
    with ruta_archivo.open("rb") as f:
        for bloque in iter(lambda: f.read(65_536), b""):
            huella_sha256.update(bloque)
    return huella_sha256.hexdigest()


def cargar_datos(ruta_csv: Path = RUTA_DATOS_ORIGINALES) -> pd.DataFrame:
    """Carga y valida el CSV con el mismo lector que utiliza Streamlit."""
    from src.data import cargar_datos as cargar_datos_verificados
    return cargar_datos_verificados(ruta_csv)


def construir_escenarios(datos: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Construye las matrices de características predictores y el vector objetivo para dos momentos.
    
    - Escenario Temprano: alerta previa al inicio de evaluaciones (excluye nota_primer_periodo, nota_segundo_periodo y nota_final).
    - Escenario Tardío: evaluación antes del cierre definitivo (incluye nota_primer_periodo y nota_segundo_periodo, excluye nota_final).
    """
    objetivo = datos[COLUMNA_OBJETIVO].copy()
    
    # Escenario Temprano: sin notas parciales para evitar fuga temporal
    predictores_temprano = datos.drop(columns=[COLUMNA_OBJETIVO, *NOTAS_PREVIAS]).copy()
    
    # Escenario Tardío: con notas parciales
    predictores_tardio = datos.drop(columns=[COLUMNA_OBJETIVO]).copy()
    
    return {
        "temprano": (predictores_temprano, objetivo),
        "tardio": (predictores_tardio, objetivo),
    }


def clasificar_columnas(predictores: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Clasifica los predictores en numéricos y categóricos."""
    cols_num = predictores.select_dtypes(include="number").columns.tolist()
    cols_cat = predictores.select_dtypes(exclude="number").columns.tolist()
    return cols_num, cols_cat


def realizar_particion_datos(
    datos: pd.DataFrame,
    proporcion_prueba: float = PROPORCION_PRUEBA,
    semilla: int = SEMILLA_ALEATORIA,
    guardar_indices: bool = True,
) -> Tuple[List[int], List[int]]:
    """Realiza una división común y reproducible de índices para ambos escenarios (80/20).
    
    Guarda los índices de entrenamiento y prueba en disco para máxima trazabilidad.
    """
    total_filas = len(datos)
    indices = list(range(total_filas))
    
    indices_entrenamiento, indices_prueba = train_test_split(
        indices, test_size=proporcion_prueba, random_state=semilla, shuffle=True
    )
    indices_entrenamiento = sorted(indices_entrenamiento)
    indices_prueba = sorted(indices_prueba)
    
    # Control de fuga y solapamiento
    conjunto_entrenamiento = set(indices_entrenamiento)
    conjunto_prueba = set(indices_prueba)
    assert conjunto_entrenamiento.isdisjoint(conjunto_prueba), "¡Error crítico! Existen índices compartidos entre train y test."
    assert len(conjunto_entrenamiento | conjunto_prueba) == total_filas, "¡Error! La unión de particiones no coincide con el total."
    
    if guardar_indices:
        DIRECTORIO_DATOS_PROCESADOS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"indice": indices_entrenamiento, "particion": "entrenamiento"}).to_csv(
            DIRECTORIO_DATOS_PROCESADOS / "indices_entrenamiento.csv", index=False
        )
        pd.DataFrame({"indice": indices_prueba, "particion": "prueba"}).to_csv(
            DIRECTORIO_DATOS_PROCESADOS / "indices_prueba.csv", index=False
        )
        
    return indices_entrenamiento, indices_prueba


def obtener_particiones_escenario(
    datos: pd.DataFrame,
    nombre_escenario: str,
    indices_entrenamiento: List[int],
    indices_prueba: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Retorna predictores_entrenamiento, predictores_prueba, objetivo_entrenamiento, objetivo_prueba para el escenario solicitado."""
    escenarios = construir_escenarios(datos)
    if nombre_escenario not in escenarios:
        raise ValueError(f"Escenario '{nombre_escenario}' no reconocido. Opciones: 'temprano', 'tardio'.")
        
    predictores, objetivo = escenarios[nombre_escenario]
    predictores_entrenamiento = predictores.iloc[indices_entrenamiento].copy()
    predictores_prueba = predictores.iloc[indices_prueba].copy()
    objetivo_entrenamiento = objetivo.iloc[indices_entrenamiento].copy()
    objetivo_prueba = objetivo.iloc[indices_prueba].copy()
    
    return predictores_entrenamiento, predictores_prueba, objetivo_entrenamiento, objetivo_prueba
