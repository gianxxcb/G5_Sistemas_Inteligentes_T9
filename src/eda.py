"""Módulo para Análisis Exploratorio de Datos (EDA) y Diagnóstico de Calidad.

Genera las tablas T1 y T2 exigidas por la rúbrica y las visualizaciones interpretadas.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    DIRECTORIO_FIGURAS,
    DIRECTORIO_TABLAS,
    COLUMNA_OBJETIVO,
    DESCRIPCIONES_VARIABLES,
)


def generar_tabla_t1_diccionario(datos: pd.DataFrame) -> pd.DataFrame:
    """Genera la Tabla T1: Diccionario de variables completo con roles y momento de disponibilidad."""
    filas = []
    for col in datos.columns:
        es_objetivo = (col == COLUMNA_OBJETIVO)
        es_nota_parcial = (col in ["nota_primer_periodo", "nota_segundo_periodo"])
        
        if es_objetivo:
            rol = "Objetivo (Target)"
            disp_temprano = "No disponible"
            disp_tardio = "No disponible (a predecir)"
            estado_inclusion = "Excluido de predictores (Variable a estimar)"
            riesgo_fuga = "CRÍTICO: Si se incluye en predictores produce fuga total (target leakage)."
        elif es_nota_parcial:
            rol = "Predictor Académico Parcial"
            disp_temprano = "No disponible"
            disp_tardio = "Disponible"
            estado_inclusion = "Excluido en Escenario Temprano / Incluido en Escenario Tardío"
            riesgo_fuga = "ALTO: En predicción temprana genera fuga temporal (lookahead bias)."
        else:
            rol = "Predictor Sociodemográfico / Académico Inicial"
            disp_temprano = "Disponible"
            disp_tardio = "Disponible"
            estado_inclusion = "Incluido en ambos escenarios"
            riesgo_fuga = "Bajo: Datos disponibles desde la matrícula o inicio del periodo escolar."
            
        serie = datos[col]
        es_num = pd.api.types.is_numeric_dtype(serie)
        rango_valores = f"{serie.min()} a {serie.max()}" if es_num else " | ".join(sorted(serie.astype(str).unique()))
        
        filas.append({
            "Variable": col,
            "Definición": DESCRIPCIONES_VARIABLES.get(col, "Sin descripción"),
            "Tipo de Dato": str(serie.dtype),
            "Valores / Rango": rango_valores,
            "Rol": rol,
            "Disp. Temprana": disp_temprano,
            "Disp. Tardía": disp_tardio,
            "Inclusión": estado_inclusion,
            "Riesgo de Fuga": riesgo_fuga,
        })
        
    t1_df = pd.DataFrame(filas)
    t1_df.to_csv(DIRECTORIO_TABLAS / "T1_diccionario_variables.csv", index=False, encoding="utf-8-sig")
    return t1_df


def generar_tabla_t2_calidad(datos: pd.DataFrame) -> pd.DataFrame:
    """Genera la Tabla T2: Diagnóstico de calidad, anomalías y medidas preventivas."""
    filas = []
    total_filas = len(datos)
    
    for col in datos.columns:
        serie = datos[col]
        nulos = int(serie.isna().sum())
        pct_nulos = round((nulos / total_filas) * 100, 2)
        unicos = int(serie.nunique(dropna=True))
        es_num = pd.api.types.is_numeric_dtype(serie)
        
        min_val = str(serie.min()) if es_num else "-"
        max_val = str(serie.max()) if es_num else "-"
        media_val = f"{serie.mean():.2f}" if es_num else "-"
        
        # Detección de posibles valores atípicos o anomalías
        anomalias = "Ninguna identificada"
        if col == "ausencias" and serie.max() > 30:
            anomalias = f"Valores extremos detectados (máx={serie.max()} ausencias)"
        elif col in ["nota_primer_periodo", "nota_segundo_periodo", "nota_final"] and (serie == 0).sum() > 0:
            cant_ceros = (serie == 0).sum()
            anomalias = f"{cant_ceros} alumnos con nota 0 (posible abandono o no presentación)"
            
        medida = "Validación de esquema; conservación en Pipeline mediante SimpleImputer"
        if es_num:
            medida += " (imputación mediana + escalado robusto)"
        else:
            medida += " (imputación moda + OneHotEncoder ignorando categorías nuevas)"
            
        filas.append({
            "Variable": col,
            "Tipo": str(serie.dtype),
            "Nulos": nulos,
            "% Nulos": pct_nulos,
            "Valores Únicos": unicos,
            "Mínimo": min_val,
            "Máximo": max_val,
            "Media": media_val,
            "Anomalías Observadas": anomalias,
            "Medida Adoptada": medida,
        })
        
    t2_df = pd.DataFrame(filas)
    t2_df.to_csv(DIRECTORIO_TABLAS / "T2_diagnostico_calidad.csv", index=False, encoding="utf-8-sig")
    return t2_df


def generar_graficos_eda(datos: pd.DataFrame, guardar: bool = True) -> Dict[str, plt.Figure]:
    """Genera la batería de visualizaciones estadísticas obligatorias con estilo formal."""
    sns.set_theme(style="whitegrid", palette="deep")
    figuras = {}
    
    # 1. Histograma y KDE de la variable objetivo nota_final
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(datos[COLUMNA_OBJETIVO], kde=True, bins=20, color="#1f77b4", edgecolor="black", ax=ax1)
    media = datos[COLUMNA_OBJETIVO].mean()
    mediana = datos[COLUMNA_OBJETIVO].median()
    ax1.axvline(media, color="red", linestyle="--", linewidth=1.5, label=f"Media: {media:.2f}")
    ax1.axvline(mediana, color="green", linestyle="-.", linewidth=1.5, label=f"Mediana: {mediana:.2f}")
    ax1.set_title("Figura 1: Distribución de la Calificación Final (nota_final)", fontsize=13, weight="bold")
    ax1.set_xlabel("Nota Final (nota_final: escala 0-20)", fontsize=11)
    ax1.set_ylabel("Frecuencia (N° Estudiantes)", fontsize=11)
    ax1.legend()
    figuras["01_distribucion_g3"] = fig1
    
    # 2. nota_primer_periodo vs nota_final con recta de regresión
    fig2, ax2 = plt.subplots(figsize=(7.5, 5))
    sns.regplot(data=datos, x="nota_primer_periodo", y="nota_final", scatter_kws={"alpha": 0.45, "color": "#2ca02c"},
                line_kws={"color": "red", "linewidth": 2}, ax=ax2)
    corr_g1 = datos["nota_primer_periodo"].corr(datos["nota_final"])
    ax2.set_title(f"Figura 2: Relación nota_primer_periodo vs nota_final (Correlación r = {corr_g1:.3f})", fontsize=13, weight="bold")
    ax2.set_xlabel("Calificación Periodo 1 (nota_primer_periodo)", fontsize=11)
    ax2.set_ylabel("Calificación Final (nota_final)", fontsize=11)
    figuras["02_g1_versus_g3"] = fig2
    
    # 3. nota_segundo_periodo vs nota_final con recta de regresión
    fig3, ax3 = plt.subplots(figsize=(7.5, 5))
    sns.regplot(data=datos, x="nota_segundo_periodo", y="nota_final", scatter_kws={"alpha": 0.45, "color": "#ff7f0e"},
                line_kws={"color": "darkblue", "linewidth": 2}, ax=ax3)
    corr_g2 = datos["nota_segundo_periodo"].corr(datos["nota_final"])
    ax3.set_title(f"Figura 3: Relación nota_segundo_periodo vs nota_final (Correlación r = {corr_g2:.3f})", fontsize=13, weight="bold")
    ax3.set_xlabel("Calificación Periodo 2 (nota_segundo_periodo)", fontsize=11)
    ax3.set_ylabel("Calificación Final (nota_final)", fontsize=11)
    figuras["03_g2_versus_g3"] = fig3
    
    # 4. Ausencias vs nota_final
    fig4, ax4 = plt.subplots(figsize=(7.5, 5))
    sns.scatterplot(data=datos, x="ausencias", y="nota_final", alpha=0.5, color="#d62728", ax=ax4)
    corr_abs = datos["ausencias"].corr(datos["nota_final"])
    ax4.set_title(f"Figura 4: Ausencias vs nota_final (Correlación r = {corr_abs:.3f})", fontsize=13, weight="bold")
    ax4.set_xlabel("Número de Ausencias", fontsize=11)
    ax4.set_ylabel("Calificación Final (nota_final)", fontsize=11)
    figuras["04_ausencias_versus_g3"] = fig4
    
    # 5. Matriz de correlación de variables numéricas
    fig5, ax5 = plt.subplots(figsize=(11, 9))
    corr_mat = datos.select_dtypes(include="number").corr()
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8}, ax=ax5, annot_kws={"size": 7.5})
    ax5.set_title("Figura 5: Matriz de Correlación de Pearson (Variables Numéricas)", fontsize=13, weight="bold")
    figuras["05_matriz_correlacion"] = fig5
    
    # 6. Gráfico adicional: Boxplot de nota_final según Reprobaciones Previas (asignaturas_reprobadas)
    fig6, ax6 = plt.subplots(figsize=(7.5, 5))
    sns.boxplot(data=datos, x="asignaturas_reprobadas", y="nota_final", hue="asignaturas_reprobadas", palette="Blues_d", legend=False, ax=ax6)
    ax6.set_title("Figura 6: Calificación Final (nota_final) según Reprobaciones Previas", fontsize=13, weight="bold")
    ax6.set_xlabel("Historial de Asignaturas Reprobadas (Failures)", fontsize=11)
    ax6.set_ylabel("Calificación Final (nota_final)", fontsize=11)
    figuras["06_boxplot_failures_g3"] = fig6

    if guardar:
        DIRECTORIO_FIGURAS.mkdir(parents=True, exist_ok=True)
        for nombre, fig in figuras.items():
            fig.tight_layout()
            fig.savefig(DIRECTORIO_FIGURAS / f"{nombre}.png", dpi=200, bbox_inches="tight")
            
    return figuras
