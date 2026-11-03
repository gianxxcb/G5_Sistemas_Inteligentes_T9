"""Comprueba el menú, las predicciones y la entrada única de NotaClara."""
from pathlib import Path
import ast,json,math,runpy,sys
import pandas as pd
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import src.interfaz as interfaz
from src.data_loader import cargar_datos
from src.predict import cargar_pipeline_entrenado
raiz=Path(__file__).resolve().parent
for archivo in [*raiz.glob('*.py'),*raiz.glob('src/*.py')]:
    ast.parse(archivo.read_text(encoding='utf-8'))
assert [p.name for p in (raiz/'notebooks').glob('*.ipynb')]==['PrediccionesNotasAlumnos.ipynb']
for ruta in [raiz/'notebooks/PrediccionesNotasAlumnos.ipynb',raiz/'documentacion/Analisis_16_Etapas.ipynb']:
    nbformat.validate(nbformat.read(ruta,as_version=4))
salidas=[]
interfaz.display=salidas.append
predictores=cargar_datos().drop(columns=['nota_final'])
modelo=cargar_pipeline_entrenado()
campos,calcular,resultado=interfaz.construir_panel(modelo,predictores)
assert len(campos)==32
assert all(c.layout.height=='42px' and c.layout.overflow=='visible' for c in campos.values())
panel=salidas[-1]
estructura=panel.children[1]
menu,principal=estructura.children
botones=menu.children[1:5]
encabezado,cuerpo=principal.children
formulario=cuerpo.children[0]
contenido=formulario.children[0]
assert 'NotaClara' in menu.children[0].value
assert len(menu.children)==5
assert all('(actual)' not in b.description for b in botones)
valor_edad=campos['edad'].value
for i,nombre in enumerate(['Académico','Estudiante','Familia','Bienestar']):
    botones[i].click()
    assert nombre in encabezado.value
    assert 'nav-activa' in botones[i]._dom_classes
    assert botones[i].description==nombre
    assert sum('nav-activa' in b._dom_classes for b in botones)==1
    assert campos['edad'].value==valor_edad
    assert len(contenido.children)==1
registro=pd.DataFrame([{n:c.value for n,c in campos.items()}],columns=predictores.columns)
nota=float(modelo.predict(registro)[0])
calcular.click()
assert f'{nota:.2f}' in resultado.value
assert calcular.description=='Estimar nota final' and not calcular.disabled
campos['edad'].value=campos['edad'].min
assert 'Tu estimación aparecerá aquí' in resultado.value
restablecer=formulario.children[1].children[1]
restablecer.click()
assert campos['edad'].value==valor_edad
assert 'Bienestar' in encabezado.value
assert all(not b.icon for b in [*botones,calcular,restablecer])
assert 'Requiere apoyo académico' in interfaz._resultado(7)
assert 'Aprobación proyectada' in interfaz._resultado(16)
assert '&lt;script&gt;' in interfaz._resultado(error='<script>')
# Contraste WCAG de los pares principales de la interfaz.
def luminancia(hexadecimal):
    valores=[int(hexadecimal[i:i+2],16)/255 for i in (1,3,5)]
    lineales=[v/12.92 if v<=0.04045 else ((v+0.055)/1.055)**2.4 for v in valores]
    return sum(p*v for p,v in zip([.2126,.7152,.0722],lineales))
for texto,fondo in [('#111827','#FFFFFF'),('#FFFFFF','#12336A'),('#FFFFFF','#245BC0'),('#334155','#F0F5FF')]:
    a,b=sorted([luminancia(texto),luminancia(fondo)])
    assert (b+.05)/(a+.05)>=4.5,(texto,fondo)
generados=runpy.run_path(str(raiz/'crear_notebook.py'))
assert generados['CUADERNO']==json.loads((raiz/'notebooks/PrediccionesNotasAlumnos.ipynb').read_text(encoding='utf-8'))
assert generados['CUADERNO_ANALISIS']==json.loads((raiz/'documentacion/Analisis_16_Etapas.ipynb').read_text(encoding='utf-8'))
cuaderno=nbformat.read(raiz/'notebooks/PrediccionesNotasAlumnos.ipynb',as_version=4)
gestor=KernelManager(kernel_name='python3')
gestor.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(cuaderno,km=gestor,timeout=120,resources={'metadata':{'path':str(raiz/'notebooks')}}).execute()
assert any('application/vnd.jupyter.widget-view+json' in o.get('data',{}) for c in cuaderno.cells if c.cell_type=='code' for o in c.outputs)
print('CORRECTO: menú, sección activa, navegación sin pérdida de datos, predicción, restablecimiento, contraste, generador y kernel real.')
