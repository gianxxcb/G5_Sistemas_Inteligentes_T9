"""Módulo para definición de algoritmos candidatos, fundamentos y ensamblaje de flujos.

Incluye los tres modelos supervisados obligatorios más el baseline DummyRegressor.
"""

from typing import Dict, List
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from src.config import DIRECTORIO_TABLAS
from src.preprocessing import crear_preprocesador


def obtener_estimadores_base() -> Dict[str, object]:
    """Retorna los estimadores no ajustados con sus configuraciones por defecto."""
    return {
        "referencia_media": DummyRegressor(strategy="mean"),
        "regresion_lineal": LinearRegression(),
        "vecinos_cercanos": KNeighborsRegressor(),
        "regresion_vectores_soporte": SVR(),
    }


def generar_tabla_fundamentos_modelos() -> pd.DataFrame:
    """Genera la tabla requerida en la Etapa 8: Fundamentos teóricos y pertinencia."""
    filas = [
        {
            "Algoritmo": "DummyRegressor (Línea Base)",
            "Mecanismo Matemático": "Predice la media constante $\\bar{objetivo}$ del conjunto de entrenamiento.",
            "Supuestos Principales": "Ninguno. Asume ausencia total de señal predictiva en $predictores$.",
            "Sensibilidad al Escalado": "Nula. No utiliza características $predictores$.",
            "Fortalezas": "Referencia mínima no trivial para contrastar si los modelos aprenden.",
            "Limitaciones": "Capacidad explicativa nula ($R^2 = 0$ esperado en train).",
            "Hiperparámetros Clave": "strategy='mean'",
        },
        {
            "Algoritmo": "LinearRegression",
            "Mecanismo Matemático": "Minimización analítica de mínimos cuadrados ordinarios (OLS): $\\min ||objetivo - Xw||^2$.",
            "Supuestos Principales": "Linealidad en parámetros, homocedasticidad, no multicolinealidad severa.",
            "Sensibilidad al Escalado": "Media/Baja en solución cerrada; esencial para interpretar pesos de coeficientes.",
            "Fortalezas": "Alta interpretabilidad; coste computacional casi nulo $O(p^2 n)$.",
            "Limitaciones": "Incapaz de capturar relaciones no lineales complejas sin transformaciones.",
            "Hiperparámetros Clave": "fit_intercept=True",
        },
        {
            "Algoritmo": "KNeighborsRegressor (KNN)",
            "Mecanismo Matemático": "Promedio ponderado o simple de los $k$ vecinos más cercanos según distancia métrica.",
            "Supuestos Principales": "Instancias con atributos similares poseen calificaciones similares (suavidad local).",
            "Sensibilidad al Escalado": "CRÍTICA. Variables de gran escala dominan por completo el cálculo euclidiano.",
            "Fortalezas": "No paramétrico; se adapta de manera flexible a fronteras no lineales.",
            "Limitaciones": "Degrada con alta dimensionalidad (maldición de la dimensión); inferencia lenta $O(n\\cdot p)$.",
            "Hiperparámetros Clave": "n_neighbors, weights ('uniform', 'distance'), p (métrica de Minkowski)",
        },
        {
            "Algoritmo": "Support Vector Regression (SVR)",
            "Mecanismo Matemático": "Optimización convexa con función de pérdida $\\epsilon$-insensible y penalización $C$.",
            "Supuestos Principales": "Margen de tolerancia simétrico $\\epsilon$; kernel RBF asume suavidad en espacio de Hilbert.",
            "Sensibilidad al Escalado": "CRÍTICA. La función de base radial $\\exp(-\\gamma ||x - x'||^2)$ depende directamente de distancias.",
            "Fortalezas": "Robusto ante valores atípicos dentro del margen $\\epsilon$; alto poder de generalización.",
            "Limitaciones": "Mayor costo de entrenamiento $O(n^2)$ a $O(n^3)$; opacidad en interpretabilidad.",
            "Hiperparámetros Clave": "C (regularización), epsilon (margen de tolerancia), kernel ('rbf', 'linear')",
        },
    ]
    datos_fundamentos = pd.DataFrame(filas)
    datos_fundamentos.to_csv(DIRECTORIO_TABLAS / "T_fundamentos_algoritmos.csv", index=False, encoding="utf-8-sig")
    return datos_fundamentos


def ensamblar_pipelines(
    columnas_numericas: List[str],
    columnas_categoricas: List[str],
) -> Dict[str, Pipeline]:
    """Crea flujos encapsulados conteniendo preprocesamiento y modelo estimador."""
    estimadores = obtener_estimadores_base()
    flujos = {}
    
    for nombre, estimador in estimadores.items():
        preprocesador = crear_preprocesador(columnas_numericas, columnas_categoricas)
        flujo = Pipeline(
            steps=[
                ("preprocesamiento", preprocesador),
                ("modelo", estimador),
            ]
        )
        flujos[nombre] = flujo
        
    return flujos
