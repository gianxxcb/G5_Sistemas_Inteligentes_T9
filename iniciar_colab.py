"""Abre JupyterLab por ngrok usando únicamente el entorno aislado del proyecto."""
import argparse
from getpass import getpass
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request


CUADERNO = "PrediccionesNotasAlumnos.ipynb"


def esperar_servidor(proceso, puerto, clave, intentos=60):
    for _ in range(intentos):
        if proceso.poll() is not None:
            raise RuntimeError("JupyterLab no inició. Revisa .colab/jupyterlab.log.")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{puerto}/api?token={clave}", timeout=1
            ):
                return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.5)
    raise RuntimeError("JupyterLab no respondió a tiempo. Revisa .colab/jupyterlab.log.")


def ejecutar_servidor(raiz, puerto):
    # Estas dependencias se importan solo en el Python del entorno virtual.
    from pyngrok import conf, ngrok

    authtoken = os.environ.pop("NOTAS_NGROK_AUTHTOKEN", "")
    if not authtoken:
        raise ValueError("Falta el Authtoken de ngrok. Ejecuta el script sin --servidor.")
    configuracion = conf.PyngrokConfig(auth_token=authtoken)
    clave = secrets.token_urlsafe(32)
    entorno_servidor = os.environ.copy()
    entorno_servidor["PATH"] = str(Path(sys.executable).parent) + os.pathsep + entorno_servidor.get("PATH", "")
    entorno_servidor["VIRTUAL_ENV"] = sys.prefix
    directorio_registro = raiz / ".colab"
    directorio_registro.mkdir(exist_ok=True)
    proceso = None
    tunel = None
    with (directorio_registro / "jupyterlab.log").open("w", encoding="utf-8") as registro:
        try:
            proceso = subprocess.Popen(
                [sys.executable, "-m", "jupyterlab", "--no-browser", "--allow-root",
                 "--ServerApp.ip=127.0.0.1", f"--ServerApp.port={puerto}",
                 "--ServerApp.port_retries=0", "--ServerApp.allow_remote_access=True",
                 f"--ServerApp.root_dir={raiz}", f"--IdentityProvider.token={clave}"],
                cwd=raiz, stdout=registro, stderr=subprocess.STDOUT, env=entorno_servidor,
            )
            esperar_servidor(proceso, puerto, clave)
            tunel = ngrok.connect(puerto, "http", bind_tls=True, pyngrok_config=configuracion)
            enlace = f"{tunel.public_url}/lab/tree/notebooks/{CUADERNO}?token={clave}"
            print("\nAbre este enlace en otra pestaña:", flush=True)
            print(enlace, flush=True)
            print("\nEn JupyterLab selecciona Run > Run All Cells para abrir el formulario.", flush=True)
            print("Mantén esta celda en ejecución. Interrúmpela para cerrar el servidor.", flush=True)
            print("No compartas el enlace: incluye la clave de acceso a JupyterLab.", flush=True)
            while proceso.poll() is None:
                time.sleep(1)
            raise RuntimeError("El servidor de JupyterLab se detuvo.")
        except KeyboardInterrupt:
            print("\nCerrando JupyterLab y el enlace externo...", flush=True)
        finally:
            try:
                if tunel is not None:
                    ngrok.disconnect(tunel.public_url, pyngrok_config=configuracion)
            finally:
                try:
                    ngrok.kill(pyngrok_config=configuracion)
                finally:
                    if proceso is not None and proceso.poll() is None:
                        proceso.terminate()
                        try:
                            proceso.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proceso.kill()
                            proceso.wait()


def principal():
    raiz = Path(__file__).resolve().parent
    analizador = argparse.ArgumentParser(description=__doc__)
    predeterminado = "/content/notaclara_env" if Path("/content").is_dir() else str(raiz / ".venv-colab")
    analizador.add_argument("--entorno", default=predeterminado)
    analizador.add_argument("--puerto", default=8899, type=int)
    analizador.add_argument("--servidor", action="store_true", help=argparse.SUPPRESS)
    argumentos = analizador.parse_args()
    if not 1 <= argumentos.puerto <= 65535:
        analizador.error("El puerto debe estar entre 1 y 65535.")
    if not (raiz / "notebooks" / CUADERNO).is_file():
        raise FileNotFoundError(f"No se encontró notebooks/{CUADERNO}.")
    if argumentos.servidor:
        ejecutar_servidor(raiz, argumentos.puerto)
        return
    python_jupyter = Path(argumentos.entorno).resolve() / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    if not python_jupyter.is_file():
        raise FileNotFoundError("Primero ejecuta %run preparar_colab.py para instalar requisitos.txt.")
    authtoken = getpass("Pega tu Authtoken de ngrok (entrada oculta): ").strip()
    if not authtoken:
        raise ValueError("El Authtoken no puede estar vacío.")
    if any(caracter.isspace() for caracter in authtoken):
        raise ValueError("Pega solo el Authtoken, sin 'ngrok config add-authtoken'.")
    entorno = os.environ.copy()
    entorno.pop("PYTHONPATH", None)
    entorno.pop("PYTHONHOME", None)
    entorno["NOTAS_NGROK_AUTHTOKEN"] = authtoken
    entorno["PYTHONUNBUFFERED"] = "1"
    comando = [str(python_jupyter), str(Path(__file__).resolve()), "--servidor",
               "--puerto", str(argumentos.puerto)]
    proceso = subprocess.Popen(comando, cwd=raiz, env=entorno, start_new_session=(os.name == "posix"))
    del entorno["NOTAS_NGROK_AUTHTOKEN"]
    try:
        codigo = proceso.wait()
        if codigo:
            raise RuntimeError("No se pudo abrir JupyterLab. Revisa el mensaje anterior.")
    except KeyboardInterrupt:
        # Detener el trabajador permite que su bloque finally cierre servidor y túnel.
        import signal
        if os.name == "posix":
            proceso.send_signal(signal.SIGINT)
        else:
            proceso.terminate()
        try:
            proceso.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proceso.kill()
            proceso.wait()


if __name__ == "__main__":
    principal()
