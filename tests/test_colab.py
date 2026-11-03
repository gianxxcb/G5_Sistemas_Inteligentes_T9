"""Verificaciones de los lanzadores sin credenciales ni conexiones externas."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import iniciar_colab
import preparar_colab


class PruebasColab(unittest.TestCase):
    def test_instalacion_no_modifica_el_python_activo(self):
        with tempfile.TemporaryDirectory() as temporal:
            with patch.object(preparar_colab.venv, "EnvBuilder") as constructor:
                with patch.object(preparar_colab.subprocess, "run") as ejecutar:
                    with patch.dict(os.environ, {"PYTHONPATH": "ruta-colab", "PYTHONHOME": "ruta-colab"}):
                        with redirect_stdout(io.StringIO()):
                            python_entorno = preparar_colab.preparar_entorno(temporal)
        constructor.assert_called_once_with(with_pip=False, system_site_packages=False)
        self.assertEqual(ejecutar.call_count, 4)
        bootstrap, instalacion, comprobacion, kernel = ejecutar.call_args_list
        self.assertEqual(bootstrap.args[0], [preparar_colab.sys.executable, "-m", "pip",
                         "--python", str(python_entorno), "install", "--upgrade", "pip"])
        self.assertEqual(instalacion.args[0][:4], [str(python_entorno), "-m", "pip", "install"])
        self.assertTrue(instalacion.args[0][-1].endswith("requisitos.txt"))
        self.assertEqual(comprobacion.args[0], [str(python_entorno), "-m", "pip", "check"])
        self.assertEqual(kernel.args[0][:4], [str(python_entorno), "-m", "ipykernel", "install"])
        self.assertIn("--prefix", kernel.args[0])
        self.assertNotIn("PYTHONPATH", instalacion.kwargs["env"])
        self.assertNotIn("PYTHONHOME", instalacion.kwargs["env"])

    def test_rechaza_instalacion_sobre_el_kernel(self):
        with self.assertRaises(ValueError):
            preparar_colab.preparar_entorno(preparar_colab.sys.prefix)

    def test_espera_detecta_servidor_caido(self):
        proceso = MagicMock()
        proceso.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "no inició"):
            iniciar_colab.esperar_servidor(proceso, 8899, "clave-prueba")

    def test_espera_autentica_la_peticion_local(self):
        proceso = MagicMock()
        proceso.poll.return_value = None
        with patch.object(iniciar_colab.urllib.request, "urlopen") as abrir:
            iniciar_colab.esperar_servidor(proceso, 8899, "clave-prueba")
        abrir.assert_called_once_with("http://127.0.0.1:8899/api?token=clave-prueba", timeout=1)

    def test_enlace_autenticado_y_cierre_de_recursos(self):
        configuracion = MagicMock()
        ngrok = MagicMock()
        ngrok.connect.return_value.public_url = "https://ejemplo.ngrok.test"
        paquete = types.ModuleType("pyngrok")
        paquete.conf = configuracion
        paquete.ngrok = ngrok
        proceso = MagicMock()
        proceso.poll.return_value = None
        salida = io.StringIO()
        with tempfile.TemporaryDirectory() as temporal:
            with patch.dict("sys.modules", {"pyngrok": paquete}):
                with patch.dict(os.environ, {"NOTAS_NGROK_AUTHTOKEN": "credencial-ficticia"}):
                    with patch.object(iniciar_colab.subprocess, "Popen", return_value=proceso) as iniciar:
                        with patch.object(iniciar_colab, "esperar_servidor"):
                            with patch.object(iniciar_colab.secrets, "token_urlsafe", return_value="clave-ficticia"):
                                with patch.object(iniciar_colab.time, "sleep", side_effect=KeyboardInterrupt):
                                    with redirect_stdout(salida):
                                        iniciar_colab.ejecutar_servidor(Path(temporal), 8899)
        comando = iniciar.call_args.args[0]
        self.assertIn("--ServerApp.ip=127.0.0.1", comando)
        self.assertIn("--IdentityProvider.token=clave-ficticia", comando)
        self.assertIn("/lab/tree/notebooks/PrediccionesNotasAlumnos.ipynb?token=clave-ficticia", salida.getvalue())
        self.assertNotIn("credencial-ficticia", salida.getvalue())
        ngrok.disconnect.assert_called_once()
        ngrok.kill.assert_called_once()
        proceso.terminate.assert_called_once()
        proceso.wait.assert_called_once_with(timeout=10)


if __name__ == "__main__":
    unittest.main()
