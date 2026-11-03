"""Módulo de evaluación experimental, validación cruzada, optimización y métricas.

Aplica K-Fold estratificado/estándar y GridSearchCV únicamente sobre el conjunto de entrenamiento.
El conjunto de prueba se evalúa exclusivamente al final sobre el modelo ganador.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from src.config import (
    DIRECTORIO_FIGURAS,
    REJILLA_KNN,
    DIRECTORIO_MODELOS,
    NUMERO_PLIEGUES_CV,
    SEMILLA_ALEATORIA,
    REJILLA_SVR,
    DIRECTORIO_TABLAS,
)


def evaluar_validacion_cruzada(
    flujos: Dict[str, Pipeline],
    predictores_entrenamiento: pd.DataFrame,
    objetivo_entrenamiento: pd.Series,
    n_splits: int = NUMERO_PLIEGUES_CV,
    random_state: int = SEMILLA_ALEATORIA,
    nombre_escenario: str = "temprano",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """Ejecuta 5-Fold Cross Validation sobre predictores_entrenamiento calculando MAE, RMSE y R².
    
    Genera la Tabla T5 oficial con resultados por pliegue, media y desviación típica.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    metricas = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    
    resultados_detallados = {}
    filas_resumen = []
    
    for nombre_modelo, flujo in flujos.items():
        puntuaciones = cross_validate(
            flujo,
            predictores_entrenamiento,
            objetivo_entrenamiento,
            cv=kf,
            scoring=metricas,
            return_train_score=False,
            n_jobs=-1,
        )
        
        # Convertimos métricas de scikit-learn (negativas para errores) a valores positivos
        maes = -puntuaciones["test_mae"]
        rmses = -puntuaciones["test_rmse"]
        r2s = puntuaciones["test_r2"]
        
        resultados_detallados[nombre_modelo] = {
            "maes": maes,
            "rmses": rmses,
            "r2s": r2s,
            "fit_time": puntuaciones["fit_time"],
        }
        
        fila = {
            "Escenario": nombre_escenario.capitalize(),
            "Modelo": nombre_modelo,
            "Fold 1 (MAE)": round(maes[0], 3),
            "Fold 2 (MAE)": round(maes[1], 3),
            "Fold 3 (MAE)": round(maes[2], 3),
            "Fold 4 (MAE)": round(maes[3], 3),
            "Fold 5 (MAE)": round(maes[4], 3),
            "MAE Medio": round(float(np.mean(maes)), 3),
            "MAE Desv.": round(float(np.std(maes)), 3),
            "RMSE Medio": round(float(np.mean(rmses)), 3),
            "R2 Medio": round(float(np.mean(r2s)), 3),
            "Tiempo Medio (s)": round(float(np.mean(puntuaciones["fit_time"])), 4),
        }
        filas_resumen.append(fila)
        
    datos_t5 = pd.DataFrame(filas_resumen).sort_values("MAE Medio", ascending=True)
    datos_t5["Ranking MAE"] = range(1, len(datos_t5) + 1)
    
    datos_t5.to_csv(DIRECTORIO_TABLAS / f"T5_comparacion_cv_{nombre_escenario}.csv", index=False, encoding="utf-8-sig")
    return datos_t5, resultados_detallados


def optimizar_hiperparametros(
    flujos: Dict[str, Pipeline],
    predictores_entrenamiento: pd.DataFrame,
    objetivo_entrenamiento: pd.Series,
    n_splits: int = NUMERO_PLIEGUES_CV,
    random_state: int = SEMILLA_ALEATORIA,
    nombre_escenario: str = "temprano",
) -> Tuple[pd.DataFrame, Dict[str, GridSearchCV]]:
    """Ejecuta GridSearchCV exclusivamente dentro de los folds de train para KNN y SVR.
    
    Genera la Tabla T6 con comparativa antes y después de la optimización.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    rejillas = {
        "vecinos_cercanos": REJILLA_KNN,
        "regresion_vectores_soporte": REJILLA_SVR,
    }
    
    busquedas_rejilla = {}
    filas_t6 = []
    
    for nombre_modelo, rejilla in rejillas.items():
        flujo_base = flujos[nombre_modelo]
        
        # Rendimiento antes del GridSearch (por defecto)
        cv_defecto = cross_validate(
            flujo_base, predictores_entrenamiento, objetivo_entrenamiento, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        mae_previo = -np.mean(cv_defecto["test_score"])
        
        # Búsqueda exhaustiva
        busqueda = GridSearchCV(
            estimator=flujo_base,
            param_grid=rejilla,
            cv=kf,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            refit=True,
        )
        busqueda.fit(predictores_entrenamiento, objetivo_entrenamiento)
        busquedas_rejilla[nombre_modelo] = busqueda
        
        mae_optimizado = -busqueda.best_score_
        mejora_pct = ((mae_previo - mae_optimizado) / mae_previo) * 100
        
        filas_t6.append({
            "Escenario": nombre_escenario.capitalize(),
            "Algoritmo": nombre_modelo,
            "N° Configuraciones": len(busqueda.cv_results_["params"]),
            "Mejor Configuración": str(busqueda.best_params_),
            "MAE Previo (Default)": round(float(mae_previo), 3),
            "MAE Optimizado": round(float(mae_optimizado), 3),
            "Mejora (%)": round(float(mejora_pct), 2),
        })
        
    datos_t6 = pd.DataFrame(filas_t6)
    datos_t6.to_csv(DIRECTORIO_TABLAS / f"T6_optimizacion_grid_{nombre_escenario}.csv", index=False, encoding="utf-8-sig")
    return datos_t6, busquedas_rejilla


def generar_tabla_t7_decision(
    candidatos: List[Dict[str, Any]],
    nombre_escenario: str = "tardio",
) -> pd.DataFrame:
    """Genera la Tabla T7 formal de decisión previa al conjunto de prueba."""
    datos_t7 = pd.DataFrame(candidatos)
    datos_t7.to_csv(DIRECTORIO_TABLAS / f"T7_decision_ganador_{nombre_escenario}.csv", index=False, encoding="utf-8-sig")
    return datos_t7


def evaluar_test_final(
    flujo_ganador: Pipeline,
    flujo_referencia: Pipeline,
    predictores_entrenamiento: pd.DataFrame,
    objetivo_entrenamiento: pd.Series,
    predictores_prueba: pd.DataFrame,
    objetivo_prueba: pd.Series,
    nombre_modelo: str,
    nombre_escenario: str = "tardio",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, plt.Figure]]:
    """Evalúa el flujo ganador una única vez sobre el conjunto reservado de prueba.
    
    Genera la Tabla T8, la tabla de 10 ejemplos de prueba y las figuras de residuales.
    """
    # Aseguramos el ajuste final sobre todo el conjunto de entrenamiento
    flujo_ganador.fit(predictores_entrenamiento, objetivo_entrenamiento)
    flujo_referencia.fit(predictores_entrenamiento, objetivo_entrenamiento)
    
    # Inferencia en Test
    predicciones = flujo_ganador.predict(predictores_prueba)
    predicciones_base = flujo_referencia.predict(predictores_prueba)
    
    # Métricas del ganador
    mae_prueba = mean_absolute_error(objetivo_prueba, predicciones)
    rmse_prueba = np.sqrt(mean_squared_error(objetivo_prueba, predicciones))
    r2_prueba = r2_score(objetivo_prueba, predicciones)
    
    # Métricas del baseline
    mae_base = mean_absolute_error(objetivo_prueba, predicciones_base)
    rmse_base = np.sqrt(mean_squared_error(objetivo_prueba, predicciones_base))
    r2_base = r2_score(objetivo_prueba, predicciones_base)
    
    # Serializar el modelo final ganador
    DIRECTORIO_MODELOS.mkdir(parents=True, exist_ok=True)
    ruta_modelo = DIRECTORIO_MODELOS / f"modelo_final_{nombre_escenario}_{nombre_modelo}.joblib"
    joblib.dump(flujo_ganador, ruta_modelo)
    
    # 1. Tabla T8: Resumen de Evaluación Final
    t8_filas = [
        {
            "Estimador": f"{nombre_modelo} (Ganador)",
            "Escenario": nombre_escenario.capitalize(),
            "MAE Test": round(float(mae_prueba), 3),
            "RMSE Test": round(float(rmse_prueba), 3),
            "R2 Test": round(float(r2_prueba), 3),
            "Reducción de Error vs Baseline (%)": round(float(((mae_base - mae_prueba) / mae_base) * 100), 2),
            "Interpretación": (
                f"El modelo se equivoca en promedio en {mae_prueba:.2f} puntos en la escala 0-20. "
                f"Explica el {r2_prueba*100:.1f}% de la varianza en datos no vistos."
            ),
        },
        {
            "Estimador": "DummyRegressor (Baseline)",
            "Escenario": nombre_escenario.capitalize(),
            "MAE Test": round(float(mae_base), 3),
            "RMSE Test": round(float(rmse_base), 3),
            "R2 Test": round(float(r2_base), 3),
            "Reducción de Error vs Baseline (%)": 0.0,
            "Interpretación": "Predice el promedio histórico constante sin utilizar características.",
        },
    ]
    datos_t8 = pd.DataFrame(t8_filas)
    datos_t8.to_csv(DIRECTORIO_TABLAS / f"T8_evaluacion_test_{nombre_escenario}.csv", index=False, encoding="utf-8-sig")
    
    # 2. Tabla de 10 ejemplos de prueba
    indices_muestreados = objetivo_prueba.index[:10]
    datos_10_ejemplos = pd.DataFrame({
        "ID_Estudiante": indices_muestreados,
        "Nota_Real_G3": objetivo_prueba.loc[indices_muestreados].values,
        "Nota_Predicha": np.round(predicciones[:10], 2),
        "Error_Residual (Real - Pred)": np.round(objetivo_prueba.loc[indices_muestreados].values - predicciones[:10], 2),
        "Error_Absoluto": np.round(np.abs(objetivo_prueba.loc[indices_muestreados].values - predicciones[:10]), 2),
    })
    datos_10_ejemplos.to_csv(DIRECTORIO_TABLAS / f"T8_10_ejemplos_prueba_{nombre_escenario}.csv", index=False, encoding="utf-8-sig")
    
    # 3. Gráficas de Diagnóstico de Error y Residuales
    sns.set_theme(style="whitegrid")
    residuales = objetivo_prueba.values - predicciones
    figuras_eval = {}
    
    # Fig A: Real vs Predicho
    fig_a, ax_a = plt.subplots(figsize=(7, 6))
    ax_a.scatter(objetivo_prueba, predicciones, alpha=0.6, color="#1f77b4", edgecolor="k", label="Estudiantes (Test)")
    limites = [min(objetivo_prueba.min(), predicciones.min()) - 1, max(objetivo_prueba.max(), predicciones.max()) + 1]
    ax_a.plot(limites, limites, color="red", linestyle="--", linewidth=2, label="Predicción Ideal (objetivo = x)")
    ax_a.set_xlim(limites)
    ax_a.set_ylim(limites)
    ax_a.set_title(f"Figura 7: Calificación Real vs Predicha en Test ({nombre_modelo})", fontsize=12, weight="bold")
    ax_a.set_xlabel("Nota Real (nota_final)", fontsize=11)
    ax_a.set_ylabel("Nota Predicha", fontsize=11)
    ax_a.legend()
    figuras_eval["07_real_vs_predicho"] = fig_a
    
    # Fig B: Histograma de Residuales
    fig_b, ax_b = plt.subplots(figsize=(7, 5))
    sns.histplot(residuales, kde=True, bins=15, color="#2ca02c", edgecolor="k", ax=ax_b)
    ax_b.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Error Cero (Insesgado)")
    ax_b.axvline(np.mean(residuales), color="darkorange", linestyle="-.", linewidth=1.5,
                 label=f"Sesgo Medio: {np.mean(residuales):.2f}")
    ax_b.set_title("Figura 8: Distribución de Errores Residuales (objetivo - y_hat)", fontsize=12, weight="bold")
    ax_b.set_xlabel("Residual (Puntos en escala 0-20)", fontsize=11)
    ax_b.set_ylabel("Frecuencia", fontsize=11)
    ax_b.legend()
    figuras_eval["08_distribucion_residuales"] = fig_b
    
    # Guardar figuras
    DIRECTORIO_FIGURAS.mkdir(parents=True, exist_ok=True)
    for nombre_fig, fig in figuras_eval.items():
        fig.tight_layout()
        fig.savefig(DIRECTORIO_FIGURAS / f"{nombre_fig}_{nombre_escenario}.png", dpi=200, bbox_inches="tight")
        
    return datos_t8, datos_10_ejemplos, figuras_eval
