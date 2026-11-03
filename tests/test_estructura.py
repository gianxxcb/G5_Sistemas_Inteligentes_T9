"""Compatibilidad de la estructura académica con la aplicación web."""
import json
import unittest

from src.data import RAIZ, cargar_datos
from src.data_loader import cargar_datos as cargar_datos_academicos
from src.predict import cargar_pipeline_entrenado, generar_cinco_casos_nuevos, predecir_observaciones


class PruebasEstructura(unittest.TestCase):
    def test_estructura_y_cuaderno(self):
        for nombre in ("archive", "data", "models", "notebooks", "reports", "src"):
            self.assertTrue((RAIZ / nombre).is_dir(), nombre)
        for nombre in ("app.py", "crear_notebook.py", "run_pipeline.py"):
            self.assertTrue((RAIZ / nombre).is_file(), nombre)
        cuaderno = json.loads((RAIZ / "notebooks/Estimacion_Nota_Final_16_Etapas.ipynb").read_text(encoding="utf-8"))
        for celda in cuaderno["cells"]:
            if celda["cell_type"] == "code":
                compile("".join(celda["source"]), "cuaderno", "exec")

    def test_lectores_y_modelo_compartidos(self):
        self.assertTrue(cargar_datos().equals(cargar_datos_academicos()))
        modelo = cargar_pipeline_entrenado()
        predicciones = predecir_observaciones(modelo, generar_cinco_casos_nuevos())
        self.assertEqual(len(predicciones), 5)


if __name__ == "__main__":
    unittest.main()
