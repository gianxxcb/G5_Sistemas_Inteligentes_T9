import json
from pathlib import Path
from pprint import pformat
raiz=Path(__file__).resolve().parent
app=json.loads((raiz/'notebooks/PrediccionesNotasAlumnos.ipynb').read_text(encoding='utf-8'))
analisis=json.loads((raiz/'documentacion/Analisis_16_Etapas.ipynb').read_text(encoding='utf-8'))
fuente='"""Regenera la entrada única de NotaClara y su documentación académica."""\nimport json\nfrom pathlib import Path\n\nCUADERNO = '+pformat(app,width=105,sort_dicts=False)+'\n\nCUADERNO_ANALISIS = '+pformat(analisis,width=105,sort_dicts=False)+'\n\ndef crear_notebook():\n    raiz = Path(__file__).resolve().parent\n    for nombre, cuaderno in [("notebooks/PrediccionesNotasAlumnos.ipynb", CUADERNO), ("documentacion/Analisis_16_Etapas.ipynb", CUADERNO_ANALISIS)]:\n        ruta = raiz / nombre\n        ruta.parent.mkdir(parents=True, exist_ok=True)\n        ruta.write_text(json.dumps(cuaderno, ensure_ascii=False, indent=1), encoding="utf-8")\n        print(f"Cuaderno creado: {ruta}")\n\nif __name__ == "__main__":\n    crear_notebook()\n'
(raiz/'crear_notebook.py').write_text(fuente,encoding='utf-8')
