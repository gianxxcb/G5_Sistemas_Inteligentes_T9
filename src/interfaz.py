"""Interfaz del cuaderno y simulador académico para JupyterLab."""
from html import escape
import math
import pandas as pd
import ipywidgets as controles
from IPython.display import HTML, display

ESTILOS = '''<style>
.panel-academico,.simulador-notas{font-family:'Segoe UI',Arial,sans-serif;font-size:14px;line-height:1.6;color:#111827;box-sizing:border-box}
.panel-academico *,.simulador-notas *{box-sizing:border-box}
.panel-academico{background:white;border:1px solid #dbe3ef;border-radius:12px;padding:28px;margin:16px 0}
.panel-academico h1,.panel-academico h2{color:#111827;line-height:1.2;margin:8px 0 14px}.panel-academico h1{font-size:32px}.panel-academico p{color:#334155}.panel-academico a{color:#1749a5}
.indicadores-academicos{display:flex;flex-wrap:wrap;gap:28px;margin-top:22px}.indicador-academico{padding:10px 20px 10px 0}.indicador-academico b{display:block;font-size:26px;color:#163a72}.indicador-academico span{font-size:12px;color:#334155}
.seccion-cuaderno{padding:20px 0;border-bottom:1px solid #dbe3ef;margin:24px 0 16px}.seccion-cuaderno h2{color:#153b77;font-size:24px}
.simulador-notas{width:100%;max-width:1440px;background:white;border:1px solid #dbe3ef;border-radius:12px;overflow:hidden;margin:12px 0;container-type:inline-size}
.simulador-notas ::selection{background:#cfe0ff;color:#102e60}
.simulador-notas .widget-html-content{width:100%}
.simulador-notas .estructura-aplicacion{display:grid!important;grid-template-columns:205px minmax(0,1fr);gap:0;align-items:stretch;width:100%}
.simulador-notas .barra-lateral{background:#12336a;color:white;min-width:0;width:100%;padding:30px 14px 24px;gap:6px}
.simulador-notas .marca-app{padding:0 12px 28px}.simulador-notas .marca-app h1{font-size:26px;font-weight:750;letter-spacing:-.7px;color:white;margin:0 0 6px;line-height:1.2}
.simulador-notas .marca-app p{font-size:12px;line-height:1.6;color:#d8e7ff;margin:0}
.simulador-notas .enlace-menu{width:100%!important;min-height:46px;height:auto!important;padding:12px 14px!important;margin:2px 0!important;text-align:left!important;background:transparent!important;color:#e1ecff!important;border:1px solid transparent!important;border-radius:6px!important;font-family:inherit!important;font-size:13px!important;font-weight:600!important;box-shadow:none!important;white-space:normal!important}
.simulador-notas .enlace-menu:hover{background:#20477f!important;color:white!important}
.simulador-notas .enlace-menu.nav-activa{background:#fff!important;color:#12336a!important;font-weight:750!important;border-color:#fff!important}

.simulador-notas .zona-principal{background:white;padding:30px;gap:24px;min-width:0;width:100%}
.simulador-notas .cabecera-seccion{padding-bottom:20px;border-bottom:1px solid #dbe3ef}
.simulador-notas .cabecera-seccion h2{font-size:28px;line-height:1.2;font-weight:700;color:#111827;margin:0 0 8px;letter-spacing:-.6px}
.simulador-notas p{color:#334155;font-size:13px;line-height:1.6;margin:6px 0}
.simulador-notas .cuerpo-simulador{display:grid!important;grid-template-columns:minmax(0,1fr) 250px;gap:28px;align-items:start;width:100%}
.simulador-notas .formulario-perfil,.simulador-notas .pagina-perfil{width:100%;min-width:0}
.simulador-notas .introduccion-grupo h3{font-size:17px;color:#153b77;margin:0 0 6px;font-weight:650}
.simulador-notas .introduccion-grupo p{font-size:12px;margin:0 0 22px;color:#334155}
.simulador-notas .rejilla-perfil{display:grid!important;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));column-gap:22px;row-gap:22px;width:100%}
.simulador-notas .campo-perfil{width:100%;min-width:0;gap:4px;padding:0;background:white}
.simulador-notas .rotulo-campo{font-size:12px;font-weight:650;color:#111827;margin-bottom:4px;min-height:20px}
.simulador-notas .control-entrada{width:100%!important;min-width:0;margin:0!important}
.simulador-notas .control-entrada>.widget-label{position:absolute!important;width:1px!important;height:1px!important;padding:0!important;overflow:hidden!important;clip-path:inset(50%);white-space:nowrap}
.simulador-notas select,.simulador-notas input[type=number]{background:#fff!important;color:#111827!important;border:1px solid #9baec8!important;border-radius:6px!important;font-family:inherit!important;font-size:13px;min-height:38px;padding:7px 9px;caret-color:#1749a5}
.simulador-notas .widget-readout{color:#163e83!important;font-weight:700!important;font-variant-numeric:tabular-nums}
.simulador-notas .noUi-connect{background:#245bc0!important}.simulador-notas .noUi-handle{border-color:#245bc0!important}
.simulador-notas .acciones-perfil{display:flex;flex-wrap:wrap;gap:10px;padding:26px 0 0;margin-top:24px;border-top:1px solid #e0e7f1}
.simulador-notas .boton-principal,.simulador-notas .boton-secundario{height:42px!important;width:auto!important;padding:0 18px!important;border-radius:6px!important;font-family:inherit!important;font-size:13px!important;font-weight:650!important;box-shadow:none!important}
.simulador-notas .boton-principal{background:#245bc0!important;color:white!important;border:1px solid #245bc0!important}.simulador-notas .boton-principal:hover{background:#16449a!important}.simulador-notas .boton-principal:disabled{background:#6d83a7!important;cursor:wait}
.simulador-notas .boton-secundario{background:white!important;color:#153b77!important;border:1px solid #9baec8!important}
.simulador-notas button:focus-visible,.simulador-notas select:focus-visible,.simulador-notas input:focus-visible{outline:3px solid #78a8ff!important;outline-offset:3px}
.simulador-notas .columna-resultado{width:100%;min-width:0;gap:18px;background:#f0f5ff;padding:22px;border-radius:10px}
.simulador-notas .tarjeta-resultado h3{font-size:17px;color:#153b77;margin:0;font-weight:650}
.simulador-notas .numero-nota{font-size:54px;line-height:1.15;color:#1749a5;font-weight:750;letter-spacing:-1.5px;margin:20px 0 6px;font-variant-numeric:tabular-nums}
.simulador-notas .numero-nota small{font-size:18px;color:#334155;font-weight:500;letter-spacing:0}
.simulador-notas .estado-nota{display:block;color:#153b77;font-size:13px;font-weight:700;padding:10px 0;margin:8px 0}
.simulador-notas .estado-riesgo{color:#111827;text-decoration:underline;text-underline-offset:4px}
.simulador-notas .barra-nota{height:6px;background:#c9d8ef;border-radius:3px;overflow:hidden;margin:16px 0 8px}.simulador-notas .barra-nota div{height:100%;background:#245bc0}
.simulador-notas .escala-nota{display:flex;justify-content:space-between;font-size:11px;color:#334155}
.simulador-notas .nota-ayuda{padding-top:18px;border-top:1px solid #c9d8ef}.simulador-notas .nota-ayuda b{color:#153b77}
.simulador-notas .aviso-perfil{font-size:11px;color:#334155;padding:12px 0}
@container(max-width:1100px){.simulador-notas .cuerpo-simulador{grid-template-columns:minmax(0,1fr)!important}.simulador-notas .columna-resultado{max-width:none}.simulador-notas .numero-nota{font-size:44px}}
@container(max-width:650px){.simulador-notas .estructura-aplicacion{grid-template-columns:148px minmax(0,1fr)}.simulador-notas .zona-principal{padding:20px 16px}.simulador-notas .barra-lateral{padding:24px 8px}.simulador-notas .marca-app{padding:0 7px 22px}.simulador-notas .marca-app h1{font-size:22px}.simulador-notas .enlace-menu{padding:12px 9px!important;font-size:12px!important}.simulador-notas .rejilla-perfil{grid-template-columns:1fr}}
@container(max-width:400px){.simulador-notas .estructura-aplicacion{grid-template-columns:112px minmax(0,1fr)}.simulador-notas .marca-app h1{font-size:18px}.simulador-notas .zona-principal{padding:16px 10px}.simulador-notas .cabecera-seccion h2{font-size:22px}}

/* Contenedores con altura natural: evita scroll interno y controles recortados. */
.simulador-notas .widget-html{height:auto!important;max-height:none!important;overflow:visible!important;flex:0 0 auto;font-family:inherit!important}
.simulador-notas .widget-html-content,.simulador-notas .widget-html-content *{font-family:inherit!important}
.simulador-notas .widget-html-content{overflow:visible!important;max-height:none!important}
.simulador-notas .pagina-activa,.simulador-notas .pagina-perfil,.simulador-notas .formulario-perfil{height:auto!important;max-height:none!important;overflow:visible!important;flex:0 0 auto}
.simulador-notas .rejilla-perfil{grid-template-columns:repeat(auto-fit,minmax(min(100%,225px),1fr));grid-auto-rows:max-content;align-items:start;align-content:start;column-gap:26px;row-gap:24px;overflow:visible!important}
.simulador-notas .campo-perfil{height:auto!important;min-height:82px;overflow:visible!important;align-self:start;gap:6px}
.simulador-notas .rotulo-campo{font-family:'Segoe UI',Arial,sans-serif;font-size:12px;font-weight:600;line-height:1.5;white-space:normal;color:#26364c;margin:0;min-height:20px}
.simulador-notas .control-entrada{height:42px!important;min-height:42px!important;max-height:none!important;overflow:visible!important;flex:0 0 auto;align-items:center}
.simulador-notas .control-entrada select{height:40px!important;min-height:40px!important;max-height:none!important;line-height:normal;width:100%;margin:0!important;flex:1 1 auto;border-color:#bdcce0!important;background:#fbfdff!important;transition:border-color .15s,background .15s}
.simulador-notas .control-entrada select:hover{border-color:#5785c7!important;background:#fff!important}
.simulador-notas .control-entrada .widget-slider{height:40px;min-width:0}
.simulador-notas .introduccion-grupo h3{margin:0 0 22px;color:#173e79;font-size:17px}
.simulador-notas .zona-principal{padding:32px 34px;gap:26px}
.simulador-notas .cabecera-seccion h2{font-size:29px;font-weight:700;letter-spacing:-.6px}
.simulador-notas .cabecera-seccion p{max-width:65ch;color:#475569;margin-top:9px}
.simulador-notas .barra-lateral{padding:32px 16px;gap:10px}
.simulador-notas .marca-app{padding:0 10px 28px}
.simulador-notas .enlace-menu{border-radius:8px!important;min-height:48px;padding:13px 14px!important;transition:background .15s,color .15s}
.simulador-notas .enlace-menu.nav-activa{background:#e7f0ff!important;border-color:#e7f0ff!important;color:#12336a!important}
.simulador-notas .acciones-perfil{margin-top:28px;padding-top:24px}
.simulador-notas .aviso-perfil{color:#475569;font-size:12px;padding-top:10px}
.simulador-notas .boton-principal:disabled{background:#526a90!important}
@container(max-width:650px){.simulador-notas .zona-principal{padding:22px 18px}.simulador-notas .barra-lateral{padding:26px 10px}.simulador-notas .rejilla-perfil{grid-template-columns:minmax(0,1fr);row-gap:20px}.simulador-notas .enlace-menu{padding:12px 10px!important}}
@media(prefers-reduced-motion:reduce){.simulador-notas button,.simulador-notas select{transition:none!important}}
</style>'''

def configurar_estilo():
    display(HTML(ESTILOS))

def mostrar_resumen(datos):
    display(HTML(f'''<div class="panel-academico">
    <h2>El punto de partida del análisis</h2><p>Información académica y contexto de estudiantes de secundaria.</p>
    <div class="indicadores-academicos">
    <div class="indicador-academico"><b>{len(datos):,}</b><span>Estudiantes registrados</span></div>
    <div class="indicador-academico"><b>{datos.shape[1]-1}</b><span>Variables predictoras</span></div>
    <div class="indicador-academico"><b>{datos['nota_final'].mean():.2f}</b><span>Nota media sobre 20</span></div>
    <div class="indicador-academico"><b>2</b><span>Escenarios de predicción</span></div></div></div>'''))

ETIQUETAS = {
    'escuela':'Escuela','sexo':'Sexo','edad':'Edad','zona_residencia':'Zona de residencia',
    'tamano_familia':'Tamaño de la familia','convivencia_padres':'Convivencia de los padres',
    'educacion_madre':'Educación de la madre','educacion_padre':'Educación del padre',
    'ocupacion_madre':'Ocupación de la madre','ocupacion_padre':'Ocupación del padre',
    'motivo_escuela':'Motivo de elección de escuela','tutor':'Tutor del estudiante',
    'tiempo_viaje':'Tiempo de viaje a la escuela','tiempo_estudio':'Estudio semanal',
    'asignaturas_reprobadas':'Asignaturas reprobadas','apoyo_escolar':'Apoyo educativo escolar',
    'apoyo_familiar':'Apoyo educativo familiar','clases_pagadas':'Clases particulares',
    'actividades':'Actividades extracurriculares','educacion_inicial':'Asistió a educación inicial',
    'aspira_educacion_superior':'Aspira a educación superior','acceso_internet':'Internet en el hogar',
    'relacion_sentimental':'Relación sentimental','relaciones_familiares':'Relaciones familiares',
    'tiempo_libre':'Tiempo libre','salidas_amigos':'Salidas con amigos',
    'alcohol_laboral':'Alcohol en días laborales','alcohol_fin_semana':'Alcohol en fin de semana',
    'salud':'Estado de salud','ausencias':'Ausencias escolares',
    'nota_primer_periodo':'Nota del primer periodo','nota_segundo_periodo':'Nota del segundo periodo'}
ESCALAS = {
    'tiempo_estudio':[('Menos de 2 horas',1),('Entre 2 y 5 horas',2),('Entre 5 y 10 horas',3),('Más de 10 horas',4)],
    'tiempo_viaje':[('Menos de 15 minutos',1),('Entre 15 y 30 minutos',2),('Entre 30 y 60 minutos',3),('Más de 60 minutos',4)],
    'educacion_madre':[('Sin educación formal',0),('Educación primaria',1),('De 5.º a 9.º grado',2),('Educación secundaria',3),('Educación superior',4)],
    'educacion_padre':[('Sin educación formal',0),('Educación primaria',1),('De 5.º a 9.º grado',2),('Educación secundaria',3),('Educación superior',4)],
    'salud':[('1 · Muy mala',1),('2 · Mala',2),('3 · Regular',3),('4 · Buena',4),('5 · Muy buena',5)],
    'relaciones_familiares':[('1 · Muy malas',1),('2 · Malas',2),('3 · Regulares',3),('4 · Buenas',4),('5 · Excelentes',5)]}
for _nombre in ['tiempo_libre','salidas_amigos','alcohol_laboral','alcohol_fin_semana']:
    ESCALAS[_nombre] = [(f'{numero} · {nivel}',numero) for numero,nivel in enumerate(['Muy bajo','Bajo','Moderado','Alto','Muy alto'],1)]

def _resultado(nota=None,error=None):
    if error:
        return f'<div class="tarjeta-resultado" role="alert"><h3>No se pudo calcular la nota</h3><p>{escape(str(error))}</p></div>'
    if nota is None:
        return '''<div class="tarjeta-resultado" aria-live="polite"><h3>Resultado de la estimación</h3>
        <div class="numero-nota">— <small>/ 20</small></div><p>Tu estimación aparecerá aquí.</p>
        <div class="barra-nota"><div style="width:0%"></div></div><div class="escala-nota"><span>0</span><span>20 puntos</span></div>
        <p style="margin-top:22px">Completa el perfil y selecciona <b>Estimar nota final</b> para conocer el resultado.</p></div>'''
    aprobado = nota >= 10
    estado = 'Aprobación proyectada' if aprobado else 'Requiere apoyo académico'
    recomendacion = 'Mantener el acompañamiento y los hábitos de estudio.' if aprobado else 'Revisar las dificultades y planificar apoyo educativo.'
    porcentaje = min(100,max(0,nota / 20 * 100))
    aviso = '<p>La estimación está fuera de la escala habitual de 0 a 20.</p>' if not 0 <= nota <= 20 else ''
    clase = '' if aprobado else 'estado-riesgo'
    return f'''<div class="tarjeta-resultado" aria-live="polite"><h3>Nota final estimada</h3>
    <div class="numero-nota">{nota:.2f} <small>/ 20</small></div><div class="estado-nota {clase}">{estado}</div>
    <div class="barra-nota"><div style="width:{porcentaje:.1f}%"></div></div><div class="escala-nota"><span>0</span><span>20 puntos</span></div>
    <p style="margin-top:20px">{recomendacion}</p>{aviso}<p style="font-size:11px;margin-top:16px">Umbral de aprobación: 10 puntos.</p></div>'''

def construir_panel(modelo,datos_predictores):
    """Formulario adaptable con etiquetas descriptivas y codificación del modelo intacta."""
    traducciones = {'yes':'Sí','no':'No','teacher':'Docencia','health':'Salud','services':'Servicios',
        'at_home':'Trabajo en el hogar','other':'Otro','mother':'Madre','father':'Padre','home':'Cercanía al hogar',
        'course':'Oferta académica','reputation':'Reputación','U':'Urbana','R':'Rural','F':'Femenino','M':'Masculino',
        'T':'Juntos','A':'Separados','LE3':'Hasta 3 personas','GT3':'Más de 3 personas','GP':'Gabriel Pereira','MS':'Mousinho da Silveira'}
    campos,tarjetas = {},{}
    for columna in datos_predictores.columns:
        serie = datos_predictores[columna]
        etiqueta = ETIQUETAS.get(columna,columna.replace('_',' ').capitalize())
        opciones = dict(description=etiqueta,layout=controles.Layout(
            width='100%',min_width='0',height='42px',min_height='42px',
            flex='0 0 auto',overflow='visible'))
        if columna in ESCALAS:
            campo = controles.Dropdown(options=ESCALAS[columna],value=int(serie.median()),**opciones)
        elif pd.api.types.is_numeric_dtype(serie):
            minimo,maximo = (0,20) if columna.startswith('nota_') else (int(serie.min()),int(serie.max()))
            campo = controles.IntSlider(value=int(serie.median()),min=minimo,max=maximo,continuous_update=False,**opciones)
        else:
            campo = controles.Dropdown(options=[(traducciones.get(str(valor),str(valor)),valor) for valor in sorted(serie.dropna().unique())],value=serie.mode().iloc[0],**opciones)
        campo.add_class('control-entrada')
        campos[columna] = campo
        rotulo = controles.HTML(
            f'<div class="rotulo-campo">{escape(etiqueta)}</div>',
            layout=controles.Layout(width='100%',height='auto',min_height='24px',
                                   flex='0 0 auto',overflow='visible'))
        tarjetas[columna] = controles.VBox([rotulo,campo],layout=controles.Layout(
            width='100%',min_width='0',height='auto',min_height='82px',
            overflow='visible')).add_class('campo-perfil')
    grupos = [
        ('Académico','Rendimiento y acompañamiento','Notas previas, hábitos de estudio y apoyos disponibles.',
         ['nota_primer_periodo','nota_segundo_periodo','tiempo_estudio','asignaturas_reprobadas','ausencias','apoyo_escolar','apoyo_familiar','clases_pagadas','aspira_educacion_superior']),
        ('Estudiante','Perfil del estudiante','Información personal y condiciones de acceso a la escuela.',
         ['escuela','edad','sexo','zona_residencia','tiempo_viaje','motivo_escuela','tutor','educacion_inicial']),
        ('Familia','Entorno familiar','Composición del hogar, educación y ocupación de los padres.',
         ['tamano_familia','convivencia_padres','educacion_madre','educacion_padre','ocupacion_madre','ocupacion_padre','relaciones_familiares','acceso_internet']),
        ('Bienestar','Rutina y bienestar','Actividades, relaciones y hábitos fuera del aula.',
         ['actividades','tiempo_libre','salidas_amigos','relacion_sentimental','alcohol_laboral','alcohol_fin_semana','salud'])]
    paginas,incluidos = [],set()
    for _,titulo,descripcion,columnas in grupos:
        seleccion = [nombre for nombre in columnas if nombre in campos]
        incluidos.update(seleccion)
        introduccion = controles.HTML(f'<div class="introduccion-grupo"><h3>{titulo}</h3></div>')
        rejilla = controles.GridBox([tarjetas[nombre] for nombre in seleccion]).add_class('rejilla-perfil')
        paginas.append(controles.VBox([introduccion,rejilla]).add_class('pagina-perfil'))
    adicionales = [tarjetas[nombre] for nombre in campos if nombre not in incluidos]
    if adicionales:
        paginas[-1].children += (controles.GridBox(adicionales).add_class('rejilla-perfil'),)
    contenido_seccion = controles.VBox([paginas[0]]).add_class('pagina-activa')
    encabezado = controles.HTML()
    botones_menu = []

    def seleccionar_seccion(indice):
        nombre, titulo, descripcion, _ = grupos[indice]
        contenido_seccion.children = (paginas[indice],)
        encabezado.value = (f'<div class="cabecera-seccion"><h2>{escape(nombre)}</h2>'
                            f'<p>{escape(descripcion)}</p></div>')
        for posicion, enlace in enumerate(botones_menu):
            activo = posicion == indice
            enlace.description = grupos[posicion][0]
            enlace.tooltip = ('Sección actual: ' if activo else 'Abrir sección: ') + grupos[posicion][0]
            if activo:
                enlace.add_class('nav-activa')
            else:
                enlace.remove_class('nav-activa')

    for indice, (nombre, *_) in enumerate(grupos):
        enlace = controles.Button(description=nombre).add_class('enlace-menu')
        enlace.on_click(lambda evento, posicion=indice: seleccionar_seccion(posicion))
        botones_menu.append(enlace)
    seleccionar_seccion(0)
    boton = controles.Button(description='Estimar nota final').add_class('boton-principal')
    restablecer = controles.Button(description='Restablecer perfil').add_class('boton-secundario')
    salida = controles.HTML(value=_resultado())
    valores_iniciales = {nombre:campo.value for nombre,campo in campos.items()}
    def marcar_pendiente(cambio):
        if cambio['new'] != cambio['old']:
            salida.value = _resultado()
    def predecir(evento):
        boton.disabled = True
        boton.description = 'Calculando nota...'
        try:
            registro = pd.DataFrame([{nombre:campo.value for nombre,campo in campos.items()}],columns=datos_predictores.columns)
            nota = float(modelo.predict(registro)[0])
            if not math.isfinite(nota):
                raise ValueError('El modelo devolvió un valor no válido.')
            salida.value = _resultado(nota)
        except Exception as error:
            salida.value = _resultado(error=error)
        finally:
            boton.disabled = False
            boton.description = 'Estimar nota final'
    def reiniciar(evento):
        for nombre,campo in campos.items():
            campo.value = valores_iniciales[nombre]
        salida.value = _resultado()
    for campo in campos.values():
        campo.observe(marcar_pendiente,names='value')
    boton.on_click(predecir)
    restablecer.on_click(reiniciar)
    acciones = controles.HBox([boton,restablecer]).add_class('acciones-perfil')
    aviso = controles.HTML('<div class="aviso-perfil">Ajusta el perfil y selecciona Estimar nota final.</div>')
    formulario = controles.VBox([contenido_seccion,acciones,aviso]).add_class('formulario-perfil')
    ayuda = controles.HTML('<div class="nota-ayuda"><b>Cómo usar el simulador</b><p>1. Revisa las secciones del menú lateral.</p><p>2. Ajusta los datos del estudiante.</p><p>3. Selecciona Estimar nota final.</p><p>Considera el contexto del estudiante al interpretar esta estimación.</p></div>')
    resultado = controles.VBox([salida,ayuda]).add_class('columna-resultado')
    marca = controles.HTML('<div class="marca-app"><h1>NotaClara</h1>'
        '<p>Estimación del rendimiento académico</p></div>')
    barra_lateral = controles.VBox([marca,*botones_menu]).add_class('barra-lateral')
    cuerpo = controles.HBox([formulario,resultado]).add_class('cuerpo-simulador')
    principal = controles.VBox([encabezado,cuerpo]).add_class('zona-principal')
    estructura = controles.HBox([barra_lateral,principal]).add_class('estructura-aplicacion')
    panel = controles.VBox([controles.HTML(ESTILOS),estructura]).add_class('simulador-notas')
    display(panel)
    return campos,boton,salida
