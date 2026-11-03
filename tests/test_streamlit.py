"""Pruebas de datos, pipeline y estado de la interfaz Streamlit."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.data import (OBJETIVO, RAIZ, RUTA_CSV, SECCIONES, cargar_datos,
                      calcular_sha256, describir_datos, valores_iniciales)
from src.ml import entrenar_modelo, guardar_modelo, separar_escenario
from src.predict import (cargar_pipeline, describir_resultado, generar_casos_ejemplo,
                         predecir_observaciones, predecir_perfil)


class PruebasDatosModelo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datos = cargar_datos()
        cls.modelo = cargar_pipeline()

    def test_csv_y_diccionario(self):
        self.assertEqual(self.datos.shape, (649, 33))
        self.assertEqual(len(describir_datos(self.datos)), 33)
        self.assertEqual(self.datos.isna().sum().sum(), 0)
        self.assertEqual(calcular_sha256(), "5be7b02f13f5800fddd66e2c8d834cd00ece14c552fb404def2397b31b63236b")
        columnas_formulario = [c for grupo in SECCIONES.values() for c in grupo["columnas"]]
        self.assertEqual(len(columnas_formulario), len(set(columnas_formulario)))
        self.assertEqual(set(columnas_formulario), set(self.datos.columns) - {OBJETIVO})

    def test_rechaza_csv_incompleto(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "incompleto.csv"
            self.datos.drop(columns="edad").to_csv(ruta, index=False)
            with self.assertRaisesRegex(ValueError, "Faltan columnas"):
                cargar_datos(ruta)

    def test_prediccion_respeta_orden_e_imputacion(self):
        perfil = valores_iniciales(self.datos)
        nota = predecir_perfil(self.modelo, perfil)
        invertido = dict(reversed(list(perfil.items())))
        self.assertAlmostEqual(nota, predecir_perfil(self.modelo, invertido))
        incompleto = perfil.copy()
        incompleto.pop("edad")
        with self.assertRaisesRegex(ValueError, "Faltan variables"):
            predecir_perfil(self.modelo, incompleto)
        robusto = perfil | {"ausencias": np.nan, "escuela": "ESCUELA_NUEVA"}
        self.assertTrue(np.isfinite(predecir_perfil(self.modelo, robusto)))
        # Un objetivo recibido como columna adicional nunca entra al modelo.
        self.assertAlmostEqual(nota, predecir_perfil(self.modelo, perfil | {OBJETIVO: 0}))
        self.assertAlmostEqual(nota, predecir_perfil(self.modelo, perfil | {OBJETIVO: 20}))

    def test_predice_cinco_observaciones_y_estados(self):
        predicciones = predecir_observaciones(self.modelo, generar_casos_ejemplo())
        self.assertEqual(len(predicciones), 5)
        self.assertTrue(np.isfinite(predicciones).all())
        self.assertEqual(describir_resultado(8)["estado"], "Requiere apoyo académico")
        self.assertEqual(describir_resultado(15)["estado"], "Aprobación proyectada")
        self.assertTrue(describir_resultado(21)["fuera_de_escala"])

    def test_escenarios_y_seleccion_antes_de_prueba(self):
        temprano, _ = separar_escenario(self.datos, "temprano")
        self.assertEqual(temprano.shape[1], 30)
        self.assertNotIn("nota_primer_periodo", temprano.columns)
        muestra = self.datos.sample(n=90, random_state=42).reset_index(drop=True)
        rejilla = {"Regresión de vectores de soporte": {
            "modelo__C": [.1], "modelo__epsilon": [.1], "modelo__kernel": ["linear"]}}
        modelo, informe = entrenar_modelo(muestra, rejillas=rejilla, pliegues=3)
        self.assertFalse(set(informe["indices_entrenamiento"]) & set(informe["indices_prueba"]))
        self.assertEqual(informe["n_entrenamiento"] + informe["n_prueba"], 90)
        self.assertGreaterEqual(informe["prueba"]["MAE"], 0)
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "modelo_final.joblib"
            guardar_modelo(modelo, informe, ruta)
            recargado = cargar_pipeline(ruta)
            perfil = valores_iniciales(muestra)
            self.assertAlmostEqual(predecir_perfil(modelo, perfil), predecir_perfil(recargado, perfil))


class PruebasInterfaz(unittest.TestCase):
    def test_navegacion_prediccion_e_invalidacion(self):
        app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=30).run()

        def boton(clave):
            return next(b for b in app.button if b.key == clave)

        def sin_errores():
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(len(app.error), 0, str(app.error))

        sin_errores()
        self.assertEqual(app.session_state["seccion"], "Académico")
        boton("nav_Estudiante").click().run()
        sin_errores()
        edad = next(c for c in app.slider if c.key == "campo_edad")
        edad.set_value(19).run()
        self.assertEqual(app.session_state["perfil"]["edad"], 19)
        boton("nav_Familia").click().run()
        boton("nav_Bienestar").click().run()
        boton("nav_Estudiante").click().run()
        sin_errores()
        self.assertEqual(next(c for c in app.slider if c.key == "campo_edad").value, 19)
        boton("estimar").click().run()
        sin_errores()
        self.assertTrue(np.isfinite(app.session_state["resultado"]["nota"]))
        next(c for c in app.slider if c.key == "campo_edad").set_value(20).run()
        self.assertIsNone(app.session_state["resultado"])
        boton("restablecer").click().run()
        self.assertEqual(app.session_state["seccion"], "Estudiante")
        self.assertEqual(app.session_state["perfil"], valores_iniciales(cargar_datos()))
        self.assertIsNone(app.session_state["resultado"])
        sin_errores()


if __name__ == "__main__":
    unittest.main()
