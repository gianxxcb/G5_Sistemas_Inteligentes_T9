# Predicciones de notas de alumnos

NotaClara es una aplicación de JupyterLab para estimar la calificación final de estudiantes. El cuaderno de entrada es **notebooks/PrediccionesNotasAlumnos.ipynb**. Este README reúne toda la documentación del proyecto: instalación, Colab, ejecución local, estructura y diseño.

## Ejecutar desde Google Colab

El flujo es: **clonar el repositorio, instalar requisitos.txt en un entorno aislado y abrir JupyterLab desde el enlace generado**. El entorno aislado evita modificar ipykernel y jupyter-client del Colab administrado.

### 1. Clonar el proyecto

En una celda nueva de Colab:

```python
!git clone https://github.com/gianxcb/G5_Sistemas_Inteligentes_T9.git
%cd /content/G5_Sistemas_Inteligentes_T9
```

Si ya clonaste el repositorio en esta sesión, entra en esa carpeta y actualízalo con git pull cuando no tengas cambios locales pendientes. Los archivos de esta versión deben haberse subido a GitHub antes de clonarla.

### 2. Instalar requisitos.txt

En la siguiente celda:

```python
%run preparar_colab.py
```

Este comando crea **/content/notaclara_env**, instala dentro todas las dependencias de **requisitos.txt** y comprueba su consistencia. No necesitas activar el entorno ni instalar paquetes directamente sobre el kernel de Colab. No ejecutes además %pip install -r requisitos.txt en el kernel administrado.

El entorno se crea sin invocar ensurepip. El pip del intérprete actual utiliza
`--python` para instalar pip dentro del entorno de destino; no actualiza el pip
ni los paquetes del kernel administrado. Esto evita el fallo de ensurepip de
algunas imágenes de Colab. Si el repositorio todavía contiene la versión anterior
de preparar_colab.py, actualiza ese archivo antes de volver a ejecutarlo. No es
necesario borrar el entorno parcialmente creado.

### 3. Abrir JupyterLab

En otra celda:

```python
%run iniciar_colab.py
```

El script pide el **Authtoken de ngrok** en un campo oculto. Obtén un token en https://dashboard.ngrok.com/get-started/your-authtoken y pega únicamente el token, sin el comando ngrok config add-authtoken. No lo escribas en el cuaderno ni lo subas a GitHub.

Después aparece un enlace HTTPS. Ábrelo en otra pestaña: mostrará **JupyterLab con archivos, editor y terminal**, y abrirá notebooks/PrediccionesNotasAlumnos.ipynb. Selecciona **Run > Run All Cells** para mostrar el formulario.

Mantén la celda del servidor en ejecución mientras usas JupyterLab. Para cerrar el túnel y el servidor, interrumpe esa celda con su botón de detener. No compartas el enlace: contiene la clave de acceso al servidor. No se desactiva la autenticación de JupyterLab.

No necesitas output.enable_custom_widget_manager() en Colab para mostrar controles dentro de la pestaña externa de JupyterLab.

### Restricciones y recuperación

Google restringe las interfaces externas y el control remoto en sus entornos gratuitos sin saldo de unidades de cómputo. Usa este flujo en un entorno que lo permita: https://research.google.com/colaboratory/faq.html. El entorno virtual resuelve el aislamiento de dependencias; no cambia esas condiciones.

Si ya instalaste requisitos sobre el kernel de Colab y aparecieron conflictos, descarga primero cualquier archivo que hayas modificado. Después usa **Entorno de ejecución > Desconectar y eliminar entorno de ejecución**, reconecta y repite el flujo desde la clonación. Reiniciar solamente el kernel no restaura los paquetes instalados.

Si el servidor no inicia, revisa **.colab/jupyterlab.log**. Si falta el cuaderno o los scripts, comprueba que clonaste la versión actualizada del repositorio. Los registros y entornos virtuales están excluidos por .gitignore.

## Ejecutar localmente en VS Code o JupyterLab

Usa Python 3.11 o 3.12. Desde la carpeta que contiene requisitos.txt, ejecuta en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requisitos.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Abre **notebooks/PrediccionesNotasAlumnos.ipynb** y usa **Run > Run All Cells**. Si ya estaba abierto, usa **Kernel > Restart Kernel and Run All Cells** para recargar la interfaz.

En VS Code instala las extensiones Python y Jupyter de Microsoft, abre el mismo cuaderno y selecciona .venv como kernel. Los comandos usan directamente el ejecutable del entorno; no requieren activar scripts de PowerShell.

## Usar el formulario

Completa las secciones **Académico**, **Estudiante**, **Familia** y **Bienestar** desde el menú izquierdo. La sección activa tiene un fondo resaltado y su nombre aparece en el encabezado. Puedes cambiar de sección sin perder los valores.

Pulsa **Estimar nota final** para mostrar el resultado. Al editar un dato, la predicción anterior se limpia para que puedas recalcularla. **Restablecer perfil** recupera los valores iniciales y conserva la sección seleccionada. La nota es una estimación educativa y debe interpretarse considerando el contexto del estudiante.

## Estructura del proyecto

- **notebooks/PrediccionesNotasAlumnos.ipynb**: única entrada de la aplicación.
- **documentacion/Analisis_16_Etapas.ipynb**: desarrollo académico completo.
- **documentacion/respaldos_jupyter**: respaldos anteriores.
- **src**: carga de datos, preprocesamiento, modelos, evaluación, predicción e interfaz.
- **data/raw**: archivos de datos originales, sin modificar.
- **data/processed**: índices de entrenamiento y prueba.
- **models**: modelos entrenados.
- **reports/tables** y **reports/figures**: tablas y gráficos generados.
- **requisitos.txt**: dependencias del proyecto.
- **preparar_colab.py**: instala requisitos en un entorno independiente.
- **iniciar_colab.py**: inicia JupyterLab y el acceso externo autenticado.
- **crear_notebook.py**: regenera el cuaderno de entrada y el análisis académico.

Jupyter puede volver a crear **.ipynb_checkpoints** como respaldo automático; no es otra aplicación. El explorador de VS Code oculta esa carpeta.

## Entrenar y regenerar los cuadernos

Para entrenar el modelo y generar las tablas y figuras:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

Para reconstruir el cuaderno de entrada y el documento académico con sus fuentes iniciales:

```powershell
.\.venv\Scripts\python.exe crear_notebook.py
```

La regeneración reemplaza las ediciones manuales de esos dos cuadernos. No crea nuevamente el archivo NotaClara.ipynb.

Las columnas y variables del proyecto están en español. Los CSV originales conservan sus columnas y se traducen al cargarlos. Las opciones descriptivas del formulario mantienen los códigos esperados por el modelo. Ejecuta el entrenamiento antes de utilizar un modelo antiguo que todavía espere nombres de columnas en inglés.

## Alcance del producto

Aplicación educativa de JupyterLab para estimar la nota final de estudiantes. El usuario completa 32 variables, recorre Académico, Estudiante, Familia y Bienestar y solicita una predicción. El modelo y la codificación de datos existentes se conservan.

El usuario solicitó una entrada única, menú lateral izquierdo, sección activa visible, blanco como color principal, azules secundarios, texto negro o blanco con contraste y ausencia de iconos en textos. La aplicación se llama NotaClara, con el subtítulo Estimación del rendimiento académico.

## Diseño de la interfaz

Modo: Operate. Referencia aportada: portal académico con navegación lateral.

Superficie principal blanca #FFFFFF; navegación azul oscuro #12336A; acciones azules #245BC0; texto principal #111827; texto secundario #334155. La opción activa del menú es blanca con texto azul oscuro. Encabezado con el nombre de la sección.

Menú a la izquierda, formulario al centro, estimación a la derecha en panel azul muy claro #F0F5FF. En anchos menores el resultado pasa debajo del formulario y el menú permanece a la izquierda. Tipografía Segoe UI y fuentes sans serif locales para uso sin conexión. Etiquetas sobre los controles, foco visible, botones sin iconos. Ningún dato se pierde al cambiar de sección.

Las opciones descriptivas respetan las categorías numéricas originales. Editar un dato invalida visualmente el resultado anterior. El restablecimiento conserva la sección activa.

## Skill de diseño

La skill Impeccable se conserva en **.agents/skills/impeccable**. Sus archivos Markdown pertenecen a la herramienta de diseño, no a la documentación del proyecto; no se combinan con este README.

## Alcance de las comprobaciones

El formulario se comprobó con navegación entre secciones, preservación de datos, predicción, restablecimiento, contraste de la paleta y carga en un kernel real. Los lanzadores de Colab mantienen las dependencias fuera del kernel administrado. Una conexión real con ngrok requiere tu token y una sesión de Colab; no se incluye ninguna credencial en el proyecto.
