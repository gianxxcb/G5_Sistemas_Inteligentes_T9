"""Preprocesamiento, entrenamiento, CV, selección y evaluación reproducibles.

Generación del artefacto: python -m src.ml --entrenar
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from src.data import NOTAS_PREVIAS, OBJETIVO, RAIZ, RUTA_CSV, calcular_sha256, cargar_datos

RUTA_MODELO = RAIZ / "models" / "modelo_final.joblib"
SEMILLA = 42
PLIEGUES = 5
REJILLAS = {
    "Vecinos cercanos": {"modelo__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                         "modelo__weights": ["uniform", "distance"], "modelo__p": [1, 2]},
    "Regresión de vectores de soporte": {"modelo__C": [0.1, 1.0, 5.0, 10.0, 20.0],
                                        "modelo__epsilon": [0.01, 0.1, 0.2, 0.5],
                                        "modelo__kernel": ["rbf", "linear"]},
}


def separar_escenario(datos, escenario="tardio"):
    if escenario not in {"temprano", "tardio"}:
        raise ValueError("El escenario debe ser temprano o tardio.")
    excluir = [OBJETIVO] + (NOTAS_PREVIAS if escenario == "temprano" else [])
    return datos.drop(columns=excluir).copy(), datos[OBJETIVO].copy()


def crear_preprocesador(predictores):
    numericas = predictores.select_dtypes(include="number").columns.tolist()
    categoricas = predictores.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("numericas", Pipeline([("imputacion", SimpleImputer(strategy="median")),
                                ("escalado", StandardScaler())]), numericas),
        ("categoricas", Pipeline([("imputacion", SimpleImputer(strategy="most_frequent")),
                                  ("codificacion", OneHotEncoder(handle_unknown="ignore",
                                                                 sparse_output=False))]), categoricas),
    ], remainder="drop", verbose_feature_names_out=False)


def construir_pipelines(predictores):
    estimadores = {
        "Referencia media": DummyRegressor(strategy="mean"),
        "Regresión lineal": LinearRegression(),
        "Vecinos cercanos": KNeighborsRegressor(),
        "Regresión de vectores de soporte": SVR(),
    }
    return {nombre: Pipeline([("preprocesamiento", crear_preprocesador(predictores)),
                               ("modelo", estimador)]) for nombre, estimador in estimadores.items()}


def evaluar_cv(flujos, predictores, objetivo, pliegues=PLIEGUES):
    particiones = KFold(n_splits=pliegues, shuffle=True, random_state=SEMILLA)
    filas = []
    for nombre, flujo in flujos.items():
        resultados = cross_validate(
            flujo, predictores, objetivo, cv=particiones, n_jobs=1,
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
        )
        filas.append({"Modelo": nombre, "MAE CV": float(-resultados["test_mae"].mean()),
                      "Desviación MAE": float(resultados["test_mae"].std()),
                      "RMSE CV": float(-resultados["test_rmse"].mean()),
                      "R2 CV": float(resultados["test_r2"].mean())})
    return pd.DataFrame(filas).sort_values("MAE CV").reset_index(drop=True)


def optimizar_modelos(flujos, predictores, objetivo, rejillas=None, pliegues=PLIEGUES):
    particiones = KFold(n_splits=pliegues, shuffle=True, random_state=SEMILLA)
    busquedas = {}
    for nombre, rejilla in (REJILLAS if rejillas is None else rejillas).items():
        busqueda = GridSearchCV(flujos[nombre], rejilla, scoring="neg_mean_absolute_error",
                                cv=particiones, n_jobs=1, refit=True, error_score="raise")
        busqueda.fit(predictores, objetivo)
        busquedas[nombre] = busqueda
    return busquedas


def seleccionar_modelo(flujos, comparacion, busquedas):
    # La decisión usa únicamente CV; no recibe ni consulta el conjunto de prueba.
    candidatos = []
    for fila in comparacion.to_dict(orient="records"):
        if fila["Modelo"] != "Referencia media":
            candidatos.append((fila["MAE CV"], fila["Modelo"], flujos[fila["Modelo"]], {}))
    for nombre, busqueda in busquedas.items():
        candidatos.append((-float(busqueda.best_score_), nombre,
                           busqueda.best_estimator_, busqueda.best_params_))
    if not candidatos:
        raise ValueError("No hay candidatos para seleccionar.")
    return min(candidatos, key=lambda candidato: candidato[0])


def evaluar_modelo(modelo, predictores_prueba, objetivo_prueba):
    predicciones = modelo.predict(predictores_prueba)
    return {"MAE": float(mean_absolute_error(objetivo_prueba, predicciones)),
            "RMSE": float(np.sqrt(mean_squared_error(objetivo_prueba, predicciones))),
            "R2": float(r2_score(objetivo_prueba, predicciones))}


def entrenar_modelo(datos, escenario="tardio", rejillas=None, pliegues=PLIEGUES):
    predictores, objetivo = separar_escenario(datos, escenario)
    indices_entrenamiento, indices_prueba = train_test_split(
        np.arange(len(datos)), test_size=0.20, random_state=SEMILLA, shuffle=True)
    indices_entrenamiento = sorted(map(int, indices_entrenamiento))
    indices_prueba = sorted(map(int, indices_prueba))
    if set(indices_entrenamiento) & set(indices_prueba):
        raise RuntimeError("Las particiones de entrenamiento y prueba se solapan.")
    entrenamiento = predictores.iloc[indices_entrenamiento]
    notas = objetivo.iloc[indices_entrenamiento]
    flujos = construir_pipelines(entrenamiento)
    comparacion = evaluar_cv(flujos, entrenamiento, notas, pliegues)
    busquedas = optimizar_modelos(flujos, entrenamiento, notas, rejillas, pliegues)
    mae_cv, nombre, candidato, parametros = seleccionar_modelo(flujos, comparacion, busquedas)
    ganador = clone(candidato).fit(entrenamiento, notas)
    referencia = clone(flujos["Referencia media"]).fit(entrenamiento, notas)
    # Solo después de la selección se consulta el 20 % reservado.
    metricas = evaluar_modelo(ganador, predictores.iloc[indices_prueba], objetivo.iloc[indices_prueba])
    base = evaluar_modelo(referencia, predictores.iloc[indices_prueba], objetivo.iloc[indices_prueba])
    informe = {
        "modelo": nombre, "escenario": escenario, "semilla": SEMILLA, "pliegues_cv": pliegues,
        "n_entrenamiento": len(indices_entrenamiento), "n_prueba": len(indices_prueba),
        "indices_entrenamiento": indices_entrenamiento, "indices_prueba": indices_prueba,
        "mae_cv_seleccionado": mae_cv, "hiperparametros": parametros,
        "comparacion_cv": comparacion.to_dict(orient="records"),
        "optimizacion": [{"modelo": n, "mae_cv": -float(b.best_score_), "parametros": b.best_params_}
                         for n, b in busquedas.items()],
        "prueba": metricas, "referencia_prueba": base,
        "reduccion_mae_porcentaje": 100 * (1 - metricas["MAE"] / base["MAE"]) if base["MAE"] else 0,
        "version_sklearn": sklearn.__version__, "columnas": list(predictores.columns),
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    ganador.notaclara_metadatos_ = informe
    return ganador, informe


def guardar_modelo(modelo, informe, ruta=RUTA_MODELO):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    contenido_informe = json.dumps(informe, ensure_ascii=False, indent=2)
    temporal = ruta.with_suffix(".temporal.joblib")
    try:
        joblib.dump(modelo, temporal)
        temporal.replace(ruta)
    finally:
        temporal.unlink(missing_ok=True)
    ruta.with_suffix(".json").write_text(contenido_informe, encoding="utf-8")


def principal():
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--entrenar", action="store_true", required=True)
    analizador.add_argument("--csv", type=Path, default=RUTA_CSV)
    analizador.add_argument("--salida", type=Path, default=RUTA_MODELO)
    analizador.add_argument("--escenario", choices=["temprano", "tardio"], default="tardio")
    argumentos = analizador.parse_args()
    print("Cargando y verificando datos...", flush=True)
    datos = cargar_datos(argumentos.csv)
    print("Comparando modelos y optimizando exclusivamente sobre entrenamiento...", flush=True)
    modelo, informe = entrenar_modelo(datos, argumentos.escenario)
    informe["sha256_csv"] = calcular_sha256(argumentos.csv)
    guardar_modelo(modelo, informe, argumentos.salida)
    print(f"Modelo guardado: {argumentos.salida}", flush=True)
    print(f"Ganador: {informe['modelo']} | MAE prueba: {informe['prueba']['MAE']:.3f}", flush=True)


if __name__ == "__main__":
    principal()
