"""Script de ejecución integral del flujo de Machine Learning (Grupo 06).

Ejecuta el flujo experimental de principio a fin de forma reproducible,
respetando el aislamiento estricto del conjunto de prueba y generando las evidencias
y tablas requeridas por la guía (T1 a T9 y figuras asociadas).
"""

import sys
from pathlib import Path

# Agregar raíz al path de Python
RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.config import (
    METADATOS_DATOS,
    RUTA_DATOS_ORIGINALES,
    RUTA_ZIP_ORIGINAL,
    COLUMNA_OBJETIVO,
)
from src.data_loader import (
    calcular_sha256,
    cargar_datos,
    clasificar_columnas,
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
)
from src.predict import (
    cargar_pipeline_entrenado,
    realizar_predicciones_nuevas,
)
from src.preprocessing import (
    generar_tabla_t4_preprocesamiento,
    verificar_transformacion_datos_no_vistos,
)


def ejecutar_pipeline_completo() -> None:
    print("=" * 80)
    print("INICIO DEL PIPELINE DE MACHINE LEARNING — ESTIMACIÓN DE NOTA FINAL nota_final")
    print("=" * 80)

    # 1. Carga y verificación de integridad
    print("\n[Paso 1/7] Carga y verificación de datos...")
    datos = cargar_datos(RUTA_DATOS_ORIGINALES)
    print(f" -> Registros: {datos.shape[0]} | Columnas: {datos.shape[1]}")
    if RUTA_ZIP_ORIGINAL.exists():
        print(f" -> SHA-256 ZIP: {calcular_sha256(RUTA_ZIP_ORIGINAL)}")
    print(f" -> SHA-256 CSV: {calcular_sha256(RUTA_DATOS_ORIGINALES)}")

    # 2. Análisis Exploratorio (Tablas T1, T2 y Gráficos)
    print("\n[Paso 2/7] Generando Diccionario (T1), Diagnóstico (T2) y Gráficos EDA...")
    t1 = generar_tabla_t1_diccionario(datos)
    t2 = generar_tabla_t2_calidad(datos)
    generar_graficos_eda(datos, guardar=True)
    print(f" -> T1 generada con {len(t1)} variables.")
    print(f" -> T2 generada: Nulos totales = {datos.isna().sum().sum()}, Duplicados = {datos.duplicated().sum()}")

    # 3. Partición de Datos (Train 80% / Test 20%)
    print("\n[Paso 3/7] Particionando datos (80% Train, 20% Test) con semilla 42...")
    indices_entrenamiento, indices_prueba = realizar_particion_datos(datos, guardar_indices=True)
    print(f" -> Índices de entrenamiento: {len(indices_entrenamiento)} (80%)")
    print(f" -> Índices de prueba reservados: {len(indices_prueba)} (20%)")

    # 4. Preparación de Escenarios y Preprocesamiento (T4)
    print("\n[Paso 4/7] Configurando preprocesadores sin fuga para ambos escenarios...")
    # Escenario Tardío (predictores con nota_primer_periodo y nota_segundo_periodo)
    predictores_entrenamiento_tardio, predictores_prueba_tardio, objetivo_entrenamiento_tardio, objetivo_prueba_tardio = obtener_particiones_escenario(
        datos, "tardio", indices_entrenamiento, indices_prueba
    )
    cols_num_tardio, cols_cat_tardio = clasificar_columnas(predictores_entrenamiento_tardio)
    generar_tabla_t4_preprocesamiento(cols_num_tardio, cols_cat_tardio, escenario="tardio")

    # Escenario Temprano (predictores sin nota_primer_periodo ni nota_segundo_periodo)
    predictores_entrenamiento_temprano, predictores_prueba_temprano, objetivo_entrenamiento_temprano, objetivo_prueba_temprano = obtener_particiones_escenario(
        datos, "temprano", indices_entrenamiento, indices_prueba
    )
    cols_num_temprano, cols_cat_temprano = clasificar_columnas(predictores_entrenamiento_temprano)
    generar_tabla_t4_preprocesamiento(cols_num_temprano, cols_cat_temprano, escenario="temprano")

    # Ensamblaje de Pipelines
    flujos_tardio = ensamblar_pipelines(cols_num_tardio, cols_cat_tardio)
    flujos_temprano = ensamblar_pipelines(cols_num_temprano, cols_cat_temprano)
    generar_tabla_fundamentos_modelos()

    # Verificación de robustez en datos no vistos
    verificar_transformacion_datos_no_vistos(flujos_tardio["regresion_vectores_soporte"].named_steps["preprocesamiento"], predictores_entrenamiento_tardio)
    print(" -> Robustez de preprocesamiento verificada ante categorías desconocidas y nulos sintéticos.")

    # 5. Validación Cruzada (CV de 5 folds) y Optimización de Hiperparámetros
    print("\n[Paso 5/7] Ejecutando Validación Cruzada 5-Fold y GridSearchCV (solo en Train)...")
    print(" -> Evaluando CV Escenario Tardío...")
    t5_tardio, _ = evaluar_validacion_cruzada(flujos_tardio, predictores_entrenamiento_tardio, objetivo_entrenamiento_tardio, nombre_escenario="tardio")
    print(" -> Evaluando CV Escenario Temprano...")
    t5_temprano, _ = evaluar_validacion_cruzada(flujos_temprano, predictores_entrenamiento_temprano, objetivo_entrenamiento_temprano, nombre_escenario="temprano")

    print("\n --- RANKING CV ESCENARIO TARDÍO (MAE) ---")
    print(t5_tardio[["Modelo", "MAE Medio", "MAE Desv.", "RMSE Medio", "R2 Medio", "Ranking MAE"]].to_string(index=False))

    print("\n --- RANKING CV ESCENARIO TEMPRANO (MAE) ---")
    print(t5_temprano[["Modelo", "MAE Medio", "MAE Desv.", "RMSE Medio", "R2 Medio", "Ranking MAE"]].to_string(index=False))

    print("\n -> Ejecutando GridSearchCV para KNN y SVR en Escenario Tardío...")
    t6_tardio, busquedas_rejilla_tardio = optimizar_hiperparametros(flujos_tardio, predictores_entrenamiento_tardio, objetivo_entrenamiento_tardio, nombre_escenario="tardio")
    print(t6_tardio[["Algoritmo", "Mejor Configuración", "MAE Previo (Default)", "MAE Optimizado", "Mejora (%)"]].to_string(index=False))

    # Selección formal del ganador
    # En escenario tardío, SVR u OLS suelen ganar. Seleccionamos el mejor estimador optimizado:
    mejor_svr = busquedas_rejilla_tardio["regresion_vectores_soporte"].best_estimator_
    mae_svr_opt = -busquedas_rejilla_tardio["regresion_vectores_soporte"].best_score_
    mae_lr = t5_tardio.loc[t5_tardio["Modelo"] == "regresion_lineal", "MAE Medio"].values[0]

    if mae_svr_opt <= mae_lr:
        ganador_nombre = "regresion_vectores_soporte"
        flujo_ganador = mejor_svr
        mae_ganador = mae_svr_opt
    else:
        ganador_nombre = "regresion_lineal"
        flujo_ganador = flujos_tardio["regresion_lineal"]
        mae_ganador = mae_lr

    print(f"\n -> Modelo seleccionado formalmente antes de test: {ganador_nombre} (MAE CV = {mae_ganador:.3f})")

    candidatos_t7 = [
        {
            "Candidato": "LinearRegression",
            "MAE CV": round(float(mae_lr), 3),
            "MAE Desv": round(float(t5_tardio.loc[t5_tardio["Modelo"] == "regresion_lineal", "MAE Desv."].values[0]), 3),
            "Costo Inferencia": "Inmediato O(p)",
            "Interpretabilidad": "Alta (pesos directos)",
            "Veredicto": "Finalista" if ganador_nombre != "regresion_lineal" else "GANADOR SELECCIONADO",
        },
        {
            "Candidato": "SVR (Optimizado GridSearch)",
            "MAE CV": round(float(mae_svr_opt), 3),
            "MAE Desv": "Estable en 5 folds",
            "Costo Inferencia": "Bajo O(n_sv * p)",
            "Interpretabilidad": "Media/Caja negra",
            "Veredicto": "GANADOR SELECCIONADO" if ganador_nombre == "regresion_vectores_soporte" else "Finalista",
        },
        {
            "Candidato": "KNN (Optimizado GridSearch)",
            "MAE CV": round(float(-busquedas_rejilla_tardio["vecinos_cercanos"].best_score_), 3),
            "MAE Desv": "Sensible a densidad",
            "Costo Inferencia": "Alto O(n * p)",
            "Interpretabilidad": "Media",
            "Veredicto": "Descartado por mayor error",
        },
    ]
    generar_tabla_t7_decision(candidatos_t7, nombre_escenario="tardio")

    # 6. Evaluación en Test Reservado (Tabla T8 y Figuras de residuales)
    print("\n[Paso 6/7] Evaluando UNA SOLA VEZ sobre el conjunto reservado de prueba (Test)...")
    flujo_referencia = flujos_tardio["referencia_media"]
    t8, ejemplos_10, figs_eval = evaluar_test_final(
        flujo_ganador=flujo_ganador,
        flujo_referencia=flujo_referencia,
        predictores_entrenamiento=predictores_entrenamiento_tardio,
        objetivo_entrenamiento=objetivo_entrenamiento_tardio,
        predictores_prueba=predictores_prueba_tardio,
        objetivo_prueba=objetivo_prueba_tardio,
        nombre_modelo=ganador_nombre,
        nombre_escenario="tardio",
    )
    print("\n --- RESULTADOS EN CONJUNTO DE PRUEBA (TEST SET: N=130) ---")
    print(t8[["Estimador", "MAE Test", "RMSE Test", "R2 Test", "Reducción de Error vs Baseline (%)"]].to_string(index=False))

    # 7. Demostración de Nuevas Predicciones (Tabla T9)
    print("\n[Paso 7/7] Generando 5 nuevas predicciones de demostración (Tabla T9)...")
    modelo_cargado = cargar_pipeline_entrenado()
    t9 = realizar_predicciones_nuevas(modelo_cargado, escenario="tardio")
    print(t9[["Identificador", "Nota Estimada (nota_final)", "Estado Proyectado", "Decisión / Acción Recomendada"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("¡PIPELINE EJECUTADO EXITOSAMENTE! Todas las tablas T1-T9 y figuras fueron generadas.")
    print("=" * 80)


if __name__ == "__main__":
    ejecutar_pipeline_completo()
