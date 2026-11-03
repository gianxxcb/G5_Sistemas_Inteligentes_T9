"""Punto de entrada de NotaClara: ejecutar con streamlit run app.py."""
from html import escape

import streamlit as st

from src.data import (ETIQUETAS, FUENTE_KAGGLE, LICENCIA, NUMERICAS, RUTA_CSV, SECCIONES,
                      cargar_datos, describir_datos, opciones_columna, valores_iniciales)
from src.predict import RUTA_MODELO, cargar_pipeline, describir_resultado, predecir_perfil


st.set_page_config(page_title="NotaClara | Predicción de notas", layout="wide",
                   initial_sidebar_state="auto")

ESTILOS = """
<style>
.stApp {background:#fff;color:#111827}
[data-testid="stAppViewContainer"] ::selection {background:#cfe0ff;color:#12336a}
[data-testid="stMainBlockContainer"] {max-width:1400px;padding:3rem 3rem 2rem}
[data-testid="stSidebar"] {background:#12336a;border-right:0;min-width:235px;max-width:255px}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-top:1.5rem}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {color:#e1ecff}
[data-testid="stSidebar"] .marca {padding:0 0 1.8rem}
[data-testid="stSidebar"] .marca h1 {color:#fff;font-size:27px;font-weight:750;line-height:1.2;
    margin:0 0 .6rem;letter-spacing:-.8px;padding:0}
[data-testid="stSidebar"] .marca p {color:#d8e7ff;font-size:12px;line-height:1.7;margin:0;max-width:180px}
[data-testid="stSidebar"] button[aria-label="Collapse sidebar"],
[data-testid="stSidebar"] button[aria-label="Collapse sidebar"] span {color:#fff!important}
[data-testid="stSidebar"] [data-testid="stButton"] {margin:.2rem 0}
[data-testid="stSidebar"] [data-testid="stButton"] button {width:100%;height:49px;justify-content:flex-start;
    border:1px solid transparent;border-radius:8px;padding:.7rem .95rem;box-shadow:none;
    font-size:13px;font-weight:600;transition:background .15s,color .15s}
[data-testid="stSidebar"] button[kind="secondary"] {background:transparent;color:#e1ecff}
[data-testid="stSidebar"] button[kind="secondary"]:hover {background:#20477f;color:#fff;border-color:transparent}
[data-testid="stSidebar"] button[kind="primary"] {background:#e7f0ff;color:#12336a;border-color:#e7f0ff;font-weight:750}
[data-testid="stSidebar"] button[kind="primary"]:hover {background:#e7f0ff;color:#12336a;border-color:#e7f0ff}
[data-testid="stSidebar"] [data-testid="stButton"] button [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stButton"] button p {color:inherit!important;text-align:left!important}
[data-testid="stSidebar"] [data-testid="stButton"] button [data-testid="stMarkdownContainer"] {width:100%}
.cabecera {padding:0 0 1.5rem;border-bottom:1px solid #dbe3ef;margin-bottom:1.8rem}
.cabecera h1 {color:#111827;font-size:30px;font-weight:750;line-height:1.2;margin:0 0 .65rem;
    padding:0;letter-spacing:-.7px}
.cabecera p {color:#475569;font-size:14px;line-height:1.65;margin:0;max-width:65ch}
.titulo-formulario {color:#173e79!important;font-size:18px!important;font-weight:650!important;
    line-height:1.4!important;margin:0 0 1rem!important;padding:0!important}
[data-testid="stWidgetLabel"] p {color:#26364c;font-size:13px;font-weight:600;line-height:1.5}
[data-testid="stSelectbox"] {margin-bottom:.7rem}
[data-testid="stSelectbox"] [data-baseweb="select"]>div {background:#fbfdff;border-color:#bdcce0;
    border-radius:7px;min-height:42px;color:#111827;font-size:13px}
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within {outline:2px solid #78a8ff;outline-offset:2px}
[data-testid="stSlider"] {margin-bottom:.85rem}
[data-testid="stSlider"] [data-testid="stTickBar"] {font-variant-numeric:tabular-nums;color:#334155}
.st-key-acciones {padding-top:1.4rem;border-top:1px solid #e0e7f1;margin-top:1.2rem}
.st-key-acciones [data-testid="stButton"] button {min-height:44px;border-radius:7px;font-size:13px;font-weight:650}
.st-key-acciones button[kind="primary"] {background:#245bc0;border-color:#245bc0;color:#fff}
.st-key-acciones button[kind="secondary"] {background:#fff;border-color:#9baec8;color:#153b77}
.st-key-panel_resultado {background:#f0f5ff;border-radius:12px;padding:1.5rem}
.resultado h2 {font-size:17px;font-weight:650;color:#153b77;margin:0;padding:0;line-height:1.5}
.resultado .nota {font-size:58px;font-weight:750;line-height:1.2;letter-spacing:-1.5px;
    color:#1749a5;margin:1.25rem 0 .4rem;font-variant-numeric:tabular-nums}
.resultado .nota small {font-size:20px;letter-spacing:0;font-weight:500;color:#334155}
.resultado .estado {font-size:14px;font-weight:700;color:#153b77;padding:.6rem 0}
.resultado .riesgo {color:#111827;text-decoration:underline;text-underline-offset:4px}
.resultado p {font-size:13px;color:#334155;line-height:1.7;margin:.6rem 0}
.resultado .umbral {font-size:11px;margin-top:1.2rem;color:#475569}
.ayuda {border-top:1px solid #c9d8ef;margin-top:1.4rem;padding-top:1.2rem}
.ayuda h3 {font-size:13px;font-weight:650;color:#153b77;margin:0 0 .6rem;padding:0}
.ayuda p {font-size:12px;color:#334155;line-height:1.7;margin:.3rem 0}
button:focus-visible {outline:3px solid #78a8ff!important;outline-offset:3px!important}
[data-testid="stCaptionContainer"] p {font-size:12px;color:#475569;line-height:1.7}
[data-testid="stHeader"] {background:rgba(255,255,255,.96)}
@media (max-width:1180px) {
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] >
    [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
        flex-direction:column!important;align-items:stretch!important;
    }
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] >
    [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width:100%!important;flex:1 1 auto!important;
    }
}
@media (max-width:900px) {
    [data-testid="stMainBlockContainer"] {padding:2.5rem 1.4rem 2rem}
    .resultado .nota {font-size:46px}
}
@media (max-width:640px) {
    [data-testid="stMainBlockContainer"] {padding:4.5rem 1rem 2rem}
    .cabecera h1 {font-size:26px}
    .st-key-panel_resultado {padding:1.3rem}
}
@media(prefers-reduced-motion:reduce) {button {transition:none!important}}
</style>
"""


@st.cache_data(show_spinner=False)
def datos_aplicacion(ruta, modificado):
    return cargar_datos(ruta)


@st.cache_resource(show_spinner=False)
def modelo_aplicacion(ruta, modificado):
    return cargar_pipeline(ruta)


def cambiar_seccion(nombre):
    st.session_state["seccion"] = nombre


def actualizar_campo(columna):
    st.session_state["perfil"][columna] = st.session_state[f"campo_{columna}"]
    st.session_state["resultado"] = None
    st.session_state["error_prediccion"] = None


def restablecer_perfil(iniciales):
    st.session_state["perfil"] = iniciales.copy()
    st.session_state["resultado"] = None
    st.session_state["error_prediccion"] = None
    for columna, valor in iniciales.items():
        st.session_state[f"campo_{columna}"] = valor


def estimar_nota(modelo):
    st.session_state["error_prediccion"] = None
    try:
        nota = predecir_perfil(modelo, st.session_state["perfil"])
        st.session_state["resultado"] = describir_resultado(nota)
    except Exception as error:
        st.session_state["resultado"] = None
        st.session_state["error_prediccion"] = str(error)


def mostrar_campo(columna, datos):
    clave = f"campo_{columna}"
    # El perfil persistente está separado de las claves de los widgets. Así la
    # limpieza de widgets de Streamlit no borra campos de secciones no visibles.
    if clave not in st.session_state:
        st.session_state[clave] = st.session_state["perfil"][columna]
    argumentos = dict(key=clave, on_change=actualizar_campo, args=(columna,))
    if columna in NUMERICAS and columna in {
        "edad", "ausencias", "asignaturas_reprobadas", "nota_primer_periodo", "nota_segundo_periodo"
    }:
        minimo, maximo = ((0, 20) if columna.startswith("nota_") else
                          (int(datos[columna].min()), int(datos[columna].max())))
        st.slider(ETIQUETAS[columna], min_value=minimo, max_value=maximo, step=1, **argumentos)
    else:
        opciones = opciones_columna(columna, datos)
        etiquetas = {valor: etiqueta for etiqueta, valor in opciones}
        st.selectbox(ETIQUETAS[columna], options=[valor for _, valor in opciones],
                     format_func=lambda valor: etiquetas[valor], **argumentos)


def mostrar_resultado():
    resultado = st.session_state["resultado"]
    if resultado is None:
        contenido = '''<div class="resultado"><h2>Resultado de la estimación</h2>
            <div class="nota">— <small>/ 20</small></div>
            <p>Tu estimación aparecerá aquí.</p>
            <p>Completa el perfil y selecciona <b>Estimar nota final</b>.</p></div>'''
    else:
        clase = "riesgo" if resultado["nota"] < 10 else ""
        contenido = f'''<div class="resultado"><h2>Nota final estimada</h2>
            <div class="nota">{resultado['nota']:.2f} <small>/ 20</small></div>
            <div class="estado {clase}">{escape(resultado['estado'])}</div>
            <p>{escape(resultado['recomendacion'])}</p>
            <p class="umbral">Umbral de aprobación: 10 puntos.</p></div>'''
    st.markdown(contenido, unsafe_allow_html=True)
    if st.session_state["error_prediccion"]:
        st.error("No se pudo estimar la nota. " + st.session_state["error_prediccion"])
    if resultado and resultado["fuera_de_escala"]:
        st.warning("La estimación está fuera de la escala habitual de 0 a 20.")
    st.markdown('''<div class="ayuda"><h3>Cómo usar el simulador</h3>
        <p>1. Revisa las secciones del menú lateral.</p>
        <p>2. Ajusta los datos del estudiante.</p>
        <p>3. Selecciona Estimar nota final.</p>
        <p>Considera el contexto del estudiante al interpretar el resultado.</p></div>''',
        unsafe_allow_html=True)


def principal():
    st.markdown(ESTILOS, unsafe_allow_html=True)
    try:
        datos = datos_aplicacion(str(RUTA_CSV), RUTA_CSV.stat().st_mtime_ns)
    except (OSError, ValueError) as error:
        st.error(f"No se pudieron cargar los datos: {error}")
        st.stop()
    try:
        modelo = modelo_aplicacion(str(RUTA_MODELO), RUTA_MODELO.stat().st_mtime_ns)
    except Exception as error:
        st.error(f"No se pudo cargar el modelo: {error}")
        st.code("python -m src.ml --entrenar", language="bash")
        st.stop()
    iniciales = valores_iniciales(datos)
    if "perfil" not in st.session_state:
        st.session_state["perfil"] = iniciales.copy()
        st.session_state["seccion"] = "Académico"
        st.session_state["resultado"] = None
        st.session_state["error_prediccion"] = None
    with st.sidebar:
        st.markdown('''<div class="marca"><h1>NotaClara</h1>
            <p>Estimación del rendimiento académico</p></div>''', unsafe_allow_html=True)
        for nombre in SECCIONES:
            st.button(nombre, key=f"nav_{nombre}", width="stretch",
                      type="primary" if st.session_state["seccion"] == nombre else "secondary",
                      on_click=cambiar_seccion, args=(nombre,))
    nombre = st.session_state["seccion"]
    seccion = SECCIONES[nombre]
    st.markdown(f'''<div class="cabecera"><h1>{escape(nombre)}</h1>
        <p>{escape(seccion['descripcion'])}</p></div>''', unsafe_allow_html=True)
    formulario, resultado = st.columns([3, 1.3], gap="large")
    with formulario:
        st.markdown(f'<h2 class="titulo-formulario">{escape(seccion["titulo"])}</h2>', unsafe_allow_html=True)
        columnas = st.columns(2, gap="large")
        for posicion, columna in enumerate(seccion["columnas"]):
            with columnas[posicion % 2]:
                mostrar_campo(columna, datos)
        with st.container(key="acciones"):
            izquierda, derecha = st.columns(2)
            with izquierda:
                st.button("Estimar nota final", key="estimar", type="primary", width="stretch",
                          on_click=estimar_nota, args=(modelo,))
            with derecha:
                st.button("Restablecer perfil", key="restablecer", width="stretch",
                          on_click=restablecer_perfil, args=(iniciales,))
        st.caption("Ajusta el perfil y selecciona Estimar nota final.")
    with resultado:
        with st.container(key="panel_resultado"):
            mostrar_resultado()
    with st.expander("Acerca de las estimaciones"):
        st.write(f"La aplicación utiliza {len(datos)} registros del conjunto Student Performance, "
                 "de estudiantes de Lengua Portuguesa en dos escuelas de Portugal.")
        metadatos = getattr(modelo, "notaclara_metadatos_", {})
        if metadatos.get("prueba"):
            st.write(f"En el conjunto de prueba, el error absoluto medio fue de "
                     f"{metadatos['prueba']['MAE']:.2f} puntos. Este promedio no es un intervalo de confianza.")
        st.markdown(f"[Fuente Kaggle]({FUENTE_KAGGLE}). Licencia del conjunto original: {LICENCIA}.")
        st.dataframe(describir_datos(datos)[["Variable", "Descripción", "Tipo"]],
                     hide_index=True, width="stretch")


if __name__ == "__main__":
    principal()
