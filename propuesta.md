Propuesta de proyecto

Título: Brechas educativas y factores asociados en PISA 2022

1) Definición del dominio (Domain Problem)

PISA 2022 es una evaluación internacional de la OCDE aplicada a estudiantes de 15 años, enfocada en lectura, matemáticas y ciencias. El conjunto de datos oficial incluye respuestas de estudiantes, directores, docentes y padres, lo que permite estudiar no solo resultados académicos, sino también contexto escolar y familiar.

Problema:

Los resultados PISA suelen presentarse como rankings o promedios nacionales, pero eso oculta diferencias importantes entre países, escuelas y grupos de estudiantes. La propuesta busca revelar patrones de desempeño y equidad, relacionando resultados académicos con variables de contexto.

Audiencia objetivo:

Estudiantes de posgrado, docentes, analistas de política educativa y tomadores de decisión interesados en comparar sistemas educativos y detectar brechas.

Necesidad que motiva la visualización:

Explorar de forma clara y comparativa qué factores se asocian con mejores resultados, dónde aparecen brechas de equidad y cómo varían los resultados entre países y subgrupos.

2) Abstracción de datos (What?)

Fuente principal:

Base de datos PISA 2022 de la OCDE.

Datos a visualizar:

- Puntajes de desempeño en lectura, matemáticas y ciencias.
- Indicadores de contexto: sexo (ST004D01T), idioma hablado en casa (LANGN), índice socioeconómico (ESCS), calidad de escuela (PQSCHOOL), y variables de ambiente escolar (BELONG).
- Se utilizarán datos individuales para explorar diferencias entre estudiantes y escuelas, y agregaciones por país para realizar comparaciones internacionales.

Tipos de datos:

Cuantitativos: puntajes, índice socioeconómico, calidad de escuela, sentido de pertenencia o ambiente escolar.

Categóricos: país, sexo, idioma.

Justificación de variables:

No se mostrará todo el dataset, sino un subconjunto que permita responder preguntas de equidad y rendimiento, especialmente ya que el dataset cuenta con más de 1.2k columnas que dificulta bastante la visualización de todas las características.

3) Abstracción de tareas (Why?)

Preguntas analíticas:

- ¿Qué países muestran mayor desempeño promedio y cuáles mayor desigualdad interna?
- ¿Qué relación hay entre contexto socioeconómico y resultados?
- ¿Cómo cambian los resultados según sexo, idioma en casa o rangos de calidad de la escuela?
- ¿Qué sistemas combinan alto rendimiento con mayor equidad?

Tareas del usuario:

- Comparar países y subgrupos.
- Explorar distribuciones y detectar brechas.
- Localizar casos extremos o anomalías.
- Resumir patrones por región o perfil de estudiantes.

4) Propuesta de visualización (How?)

Idea general del diseño:

Un dashboard interactivo de tres paneles:

- Panel 1. Vista comparativa global: Mapa coroplético para ver distribución de puntaje cross países o diagrama de barras para identificar fácilmente el top de países por puntaje promedio.
- Panel 2. Vista de relaciones: Dispersión o boxplots para relacionar desempeño con rangos de índice socioeconómico, rangos de calidad de escuela, sexo o idioma en casa. 
- Panel 3. Vista de detalle: Tablas para explorar un país o grupo específico por dominios y subgrupos.

Interacciones previstas:

- Filtro por país, región, sexo, idioma en casa y tipo de escuela.
- Tooltip con valores exactos.
- Selección cruzada entre gráficos.
- Ordenamiento por puntaje, brecha o dispersión.

Justificación de diseño:

El mapa ofrece una visión global inicial, mientras que los boxplots y las visualizaciones por subgrupos permiten analizar la variabilidad y las brechas internas de cada país.

5) Factibilidad del proyecto

Alcance realista:

El proyecto puede desarrollarse con un subconjunto del dataset PISA 2022 y un número limitado de variables, evitando sobrecargar la interfaz.

Cronograma sugerido:

- Semana 1: 
    - Limpieza y selección de variables.
    - Análisis exploratorio y definición de métricas.
- Semana 2:
    - Wireframes y prototipo.
    - Implementación del dashboard.
    - Pruebas, ajustes y presentación final.

Riesgos y mitigación:
- Dataset muy grande: usar muestras o agregados.
- Variables con muchos faltantes: documentar y excluir cuidadosamente.
- Exceso de complejidad visual: priorizar las 3 o 4 visualizaciones más importantes.