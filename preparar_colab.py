"""Instala NotaClara en un entorno aislado, sin actualizar el kernel de Colab."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv


def preparar_entorno(directorio):
    raiz = Path(__file__).resolve().parent
    requisitos = raiz / "requisitos.txt"
    if not requisitos.is_file():
        raise FileNotFoundError(f"No se encontró {requisitos}")
    directorio = Path(directorio).expanduser().resolve()
    if directorio == Path(sys.prefix).resolve():
        raise ValueError("El entorno aislado debe ser distinto del entorno activo.")
    # Algunas imágenes de Colab no incluyen un ensurepip funcional. Crear el
    # entorno sin ese paso y usar pip --python instala pip solo en el destino.
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(directorio)
    python_jupyter = directorio / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    entorno = os.environ.copy()
    # No heredar rutas de paquetes del kernel administrado de Colab.
    entorno.pop("PYTHONPATH", None)
    entorno.pop("PYTHONHOME", None)
    subprocess.run(
        [sys.executable, "-m", "pip", "--python", str(python_jupyter),
         "install", "--upgrade", "pip"],
        check=True, env=entorno,
    )
    subprocess.run(
        [str(python_jupyter), "-m", "pip", "install", "-r", str(requisitos)],
        check=True, env=entorno,
    )
    subprocess.run([str(python_jupyter), "-m", "pip", "check"], check=True, env=entorno)
    subprocess.run(
        [str(python_jupyter), "-m", "ipykernel", "install", "--prefix", str(directorio),
         "--name", "python3", "--display-name", "Python (PrediccionesNotasAlumnos)"],
        check=True, env=entorno,
    )
    print(f"Entorno de JupyterLab preparado: {python_jupyter}")
    print("Usa este ejecutable para iniciar JupyterLab, en lugar de sys.executable.")
    return python_jupyter


def principal():
    analizador = argparse.ArgumentParser(description=__doc__)
    predeterminado = "/content/notaclara_env" if Path("/content").is_dir() else ".venv-colab"
    analizador.add_argument("--entorno", default=predeterminado)
    argumentos = analizador.parse_args()
    preparar_entorno(argumentos.entorno)


if __name__ == "__main__":
    principal()
