# Proyecto Final: Sommelier Algorítmico - Control de Calidad Automatizado para Viñedos
**Materia:** Teoría de la Computación  
**Estudiantes:**
- Leonardo Quiroga
- Kimberly Moreno
- Linda Ortiz

**Link del repositorio:**
https://github.com/leoquiroga1/proyectoFinal_SommelierAlgoritmico.git

---

## 1. Contexto y Delimitación del Problema

Basándonos en los datasets proporcionados para este proyecto, encontramos información detallada sobre los atributos fisicoquímicos de diferentes variantes de productos, como el vino. A partir de la experiencia en la industria alimentaria y del conocimiento de otros sectores que desarrollan productos con largos periodos de maduración o procesos extensos de fabricación, como la industria del cuidado personal en la producción de jabones y extractos, es sabido que al finalizar el proceso se realizan pruebas para determinar la calidad del producto. Sin embargo, a pesar del tiempo, los recursos y el esfuerzo invertidos, no siempre es posible garantizar que el resultado final cumpla con los estándares establecidos.

Nuestro problema surge precisamente de esta situación. En un mercado globalizado, las empresas deben cumplir con los tiempos exigidos por los clientes sin generar sobrecostos. Cuando un producto no supera las pruebas de calidad al final de su ciclo de producción, es necesario reiniciar el proceso desde cero. Esto implica la pérdida de materias primas, tiempo y recursos, además del riesgo de perder clientes y oportunidades de venta. En el caso del vino, también pueden presentarse afectaciones a la reputación de la empresa debido a retrasos en la entrega o incumplimientos logísticos.

* **Descripción del conjunto de datos:** Se utilizará el *UCI Wine Quality Dataset*, el cual contiene atributos fisicoquímicos de diferentes variantes de vino (Alcohol, Acidez, pH, Sulfatos, etc.).
* **Objetivo:** Clasificar el vino en proceso de maduración como "Premium" o "Estándar" empleando un algoritmo de Machine Learning para predecir la calidad que tendrá de forma automatizada (metodología que también se puede emplear en otros productos, pero hemos decidido enfocarnos en uno). Esto permite anticipar y reducir las fallas durante la producción y así tomar acciones correctivas antes de completar el tiempo de añejamiento.
* **Cambio de procesos:** Se incorpora una herramienta de apoyo a la toma de decisiones basada en Machine Learning que permite realizar una clasificación preliminar de la calidad del vino utilizando únicamente atributos fisicoquímicos. La evaluación sensorial realizada por expertos se mantiene como etapa final de validación del producto (en la industria química de cuidado personal, se mantendrían los análisis típicos que certifican el material como “apto” o de “calidad”).
* **Alcance:** Procesamiento de las 11 variables químicas provistas por el análisis químico, entrenamiento del modelo, evaluación de métricas de Machine Learning y exposición a través de una API local.
* **Exclusiones:** No se procesarán variables sensoriales u organolépticas como lo son el sabor, el color o los aromas. El sistema no hará sugerencias para arreglar el vino ni determinará el precio en el mercado. Tampoco incluye despliegue en un entorno cloud de producción. El modelo no garantiza una precisión alta en la detección de vinos Premium, dado que la calidad sensorial depende también de factores que no están capturados en los datos fisicoquímicos, por lo que el modelo no reeemplaza el análisis sensorial por un sommelier.
* **Criterio de éxito:** Alcanzar una precisión superior al 80% en el conjunto de pruebas, y lograr que la API responda a nuevas consultas de forma correcta.
* **Supuestos:** Los instrumentos que proveen los datos químicos de entrada deben estar correctamente calibrados. Los estándares históricos de la calidad del vino continúan vigentes, representando las preferencias actuales en el mercado.
* **Resultado final del proyecto:** Un modelo de Machine Learning de tipo Árbol de Decisión de clasificación binaria expuesto a través de una API que recibe parámetros químicos y devuelve una categoría de calidad.

> **Meta del Proyecto:** Desarrollar una API de Machine Learning que permita a los laboratorios vitivinícolas predecir de forma inmediata la viabilidad de la calidad de un lote de vino clasificándolo como “Premium” o “Estándar” basándose exclusivamente en sus atributos fisicoquímicos, excluyendo por completo el análisis sensorial.

### 1.1 Identificación de Interesados

| Interesado | Resultado Esperado |
|---|---|
| **Patrocinador** | Prioridad alta en la reducción de costos operativos por lotes deficientes y en agilizar la clasificación inicial para la reposición de producto. También alta prioridad en la precisión para evitar descartar vinos de buena calidad por error. |
| **Usuario final** | El enólogo o técnico del laboratorio debe ingresar los resultados de los tests (11 parámetros) en una interfaz y recibir la predicción al instante de la calidad del vino. |
| **Personal de Operación TI** | Requiere despliegue mediante una API de fácil ejecución y un respaldo del archivo del modelo para recuperación inmediata ante fallas de la API. |
| **Equipo de Seguridad y Datos** | Requiere protección del histórico de calidad de la bodega. Solo el personal de laboratorio autorizado debe poder acceder al sistema para evitar alteraciones o fugas de información. |
| **Equipo de Desarrollo** | El código debe ser modular para permitir reemplazar el modelo entrenado con nuevas versiones sin reescribir la API completa (ejemplo: nuevas cepas de uva, nuevas tendencias del mercado, etc.). |

---

## 2. Definición de la Arquitectura

La arquitectura del Sommelier Algorítmico determina las decisiones de alto impacto y organiza las responsabilidades técnicas. A continuación, se definen los parámetros base aplicados a nuestro proyecto:

| Elemento | Definición en el Proyecto (Sommelier Algorítmico) | Entregable Asociado |
|---|---|---|
| **Contexto** | El sistema clasifica la calidad del vino mediante parámetros químicos. Quedan fuera el análisis sensorial y el despliegue cloud. | Diagrama de contexto (C4 Nivel 1) y alcance |
| **Drivers** | Alta precisión (>80% Accuracy) para evitar pérdidas, baja latencia en respuestas (<1s) y alta modificabilidad. | Lista priorizada de drivers técnicos y riesgos |
| **Estructura** | Frontend ligero (SPA), Backend orquestador (FastAPI), Modelo ML serializado (.pkl) y Base de Datos (SQL). | Diagrama de contenedores (C4 Nivel 2) |
| **Decisiones** | Implementación en Python, uso de Árbol de Decisión por su interpretabilidad gráfica y almacenamiento local. | 3 Registros ADR (Architecture Decision Records) |
| **Evidencia** | Separación de datos Train/Test, evaluación mediante F1-Score/Accuracy, y análisis de Falsos Positivos/Negativos. | Métricas de evaluación de ML y Matriz de Confusión |

---

## 3. Diagramas C4

### Diagrama de Contexto C4

```mermaid
flowchart LR
    %% Definición de Nodos (Formas)
    Usuario((Técnico / Enólogo))
    Sistema[Sommelier Algorítmico]
    Dataset[(Dataset UCI / Histórico)]

    %% Relaciones (Flechas)
    Usuario -- "Consulta la calidad (HTTPS)" --> Sistema
    Sistema -- "Se alimenta del histórico (CSV/DB)" --> Dataset

    %% Estilos de colores oficiales de C4
    classDef persona fill:#08427b,stroke:#052e56,color:#ffffff,stroke-width:2px;
    classDef sistema fill:#1168bd,stroke:#0b4884,color:#ffffff,stroke-width:2px;
    classDef externo fill:#999999,stroke:#666666,color:#ffffff,stroke-width:2px;

    class Usuario persona;
    class Sistema sistema;
    class Dataset externo;
```

### Diagrama de Contenedores C4 (Nivel 2)

```mermaid
flowchart LR
    %% Definición de Nodos
    Usuario((Técnico / Enólogo))

    subgraph Sistema ["Sommelier Algorítmico"]
        Frontend[Interfaz de Consulta <br/> SPA / Web]
        API[API de Predicción <br/> Python / FastAPI]
        Modelo[Modelo Predictivo <br/> Scikit-Learn .pkl]
        BD[(Base de Datos <br/> SQL / Logs)]
    end

    Dataset[(Dataset UCI / Histórico)]

    %% Relaciones
    Usuario -- "Ingresa 11 parámetros (HTTPS)" --> Frontend
    Frontend -- "Solicita predicción (JSON/REST)" --> API
    API -- "Ejecuta inferencia" --> Modelo
    API -- "Guarda registros y logs" --> BD
    API -. "Entrenamiento previo" .-> Dataset

    %% Estilos de colores oficiales de C4
    classDef persona fill:#08427b,stroke:#052e56,color:#ffffff,stroke-width:2px;
    classDef contenedor fill:#1168bd,stroke:#0b4884,color:#ffffff,stroke-width:2px;
    classDef baseDatos fill:#1168bd,stroke:#0b4884,color:#ffffff,stroke-width:2px;
    classDef externo fill:#999999,stroke:#666666,color:#ffffff,stroke-width:2px;

    class Usuario persona;
    class Frontend,API,Modelo contenedor;
    class BD,Dataset baseDatos;
```

---

## 4. Drivers Arquitectónicos

En la identificación de los Drivers Arquitectónicos, el principal driver surge de la necesidad de anticipar problemas de calidad durante el proceso de producción del vino, reduciendo costos asociados al descarte de lotes defectuosos o a la sobreproducción del mismo con la finalidad de cumplir con los pedidos recibidos por los clientes. Así que podríamos decir que el atributo más importante es la precisión del modelo predictivo, estableciendo como criterio de éxito una exactitud superior al 80% en el conjunto de pruebas. Con esto queremos lograr que el modelo detecte al menos el 80% de los vinos Premium reales analizados con la métrica de Recall, manteniendo un F1-Score superior al 50%.

Igualmente, como segundo driver, la solución debe ofrecer respuestas rápidas para facilitar la toma de decisiones en el laboratorio, puesto que las pruebas químicas podrían tomar días para obtener un resultado, adicionando retrasos en los tiempos.

En cambio, otra de las variables surge como una restricción del proyecto, que sería el uso exclusivo de variables fisicoquímicas (descritas anteriormente) y la implementación mediante una API local (no se desplegará en la nube para este proyecto). Asimismo, como última variable, es importante tener en cuenta que el software debe tener la capacidad de reemplazar el modelo inicial de Machine Learning o extenderse a otros tipos de productos (tipo de vinos, uvas, nuevas tendencias, etc.) sin modificar la arquitectura existente.

---

## 5. Atributos de Calidad

* **Driver 1 - Precisión del modelo:** Cuando el laboratorio envía los datos de un lote que terminará siendo Premium, en el conjunto de prueba el sistema lo clasifica correctamente como Premium, con un acierto de al menos 70% sobre los vinos Premium reales.
* **Driver 2 - Rapidez en arrojar resultados:** Cuando el técnico envía los 11 parámetros de una muestra en el uso normal del laboratorio, el sistema devuelve la clasificación con un tiempo de respuesta menor a 5 segundos.
* **Driver 3 - Restricciones:** Cuando se ejecuta el sistema, este recibe solo las 11 variables fisicoquímicas definidas, con 0 dependencias de servicios en la nube.
* **Driver 4 - Modelo reemplazable:** Cuando el equipo necesite reemplazar el modelo por uno nuevo, el sistema acepta el nuevo modelo sin cambios en el código de la API (0 líneas de la API modificadas).

---

## 6. Diseño de la Arquitectura Lógica

### Componentes Identificados
| Componente | Responsabilidad |
|---|---|
| **Interfaz de Consulta** | Permitir al técnico ingresar los parámetros químicos y visualizar el resultado. |
| **Motor de Clasificación** | Procesar los datos y determinar si el vino es Premium o Estándar. |
| **Modelo Predictivo** | Contener el conocimiento entrenado y ejecutar la inferencia. |

### Asignación de Capacidades
| Capacidad | Componente Responsable |
|---|---|
| Ingreso de parámetros químicos | Interfaz de Consulta |
| Validación de datos de entrada | Motor de Clasificación |
| Solicitud de predicción | Motor de Clasificación |
| Clasificación Premium / Estándar | Modelo Predictivo |
| Presentación de resultados | Interfaz de Consulta |

### Cohesión
Los componentes propuestos presentan una alta cohesión debido a que cada uno tiene una única responsabilidad claramente definida:
* **Captura de Parámetros Químicos:** Solo recibe información.
* **Validación de Muestras:** Solo verifica los datos.
* **Clasificación de Calidad:** Únicamente toma decisiones sobre la categoría del vino.
* **Gestión del Modelo Predictivo:** Administra el modelo entrenado.
* **Presentación de Resultados:** Solamente comunica la predicción al usuario.

### Acoplamiento
El sistema presenta un bajo acoplamiento debido a que cada componente depende únicamente de la información proporcionada por el componente anterior.

**Flujo lógico:**  
`Captura de Parámetros → Validación de Muestras → Clasificación de Calidad → Gestión del Modelo → Presentación de Resultados`

```mermaid
flowchart LR
A[Captura de Parámetros Químicos] --> B[Validación de Muestras]
B --> C[Clasificación de Calidad]
C --> D[Gestión del Modelo Predictivo]
D --> E[Presentación de Resultados]
```

### Asignación de Drivers Arquitectónicos
| Componente | Responsabilidad Principal |
|---|---|
| **Interfaz de Consulta** | Capturar información del análisis químico. |
| **API de Predicción** | Coordinar el flujo de validación y clasificación. |
| **Modelo Predictivo** | Generar la predicción de calidad. |

---

## 7. Registros de Decisiones de Arquitectura (ADR)

### Arquitectura Seleccionada
Se seleccionó una arquitectura en capas porque permite separar la interacción con el usuario, el procesamiento de las solicitudes y la lógica del modelo de Machine Learning, facilitando el mantenimiento por un equipo y la evolución del sistema por partes sin afectar otras áreas. La arquitectura se organiza de la siguiente manera:

1. **Capa de Presentación:** Recibe los parámetros ingresados por el usuario y muestra los resultados.
2. **Capa de Aplicación:** Valida los datos que recibe, gestiona la ejecución de la predicción y coordina la comunicación entre el modelo y la capa de presentación.
3. **Capa de Inteligencia Artificial:** Ejecuta el modelo de Árbol de Decisión ya entrenado y genera el resultado de la clasificación Premium o Estándar.

```mermaid
flowchart TD
A["Capa de Presentación<br/>Usuario / Cliente"] --> B["Capa de Aplicación<br/>API FastAPI"]
B --> C["Capa de Inteligencia Artificial<br/>Modelo Árbol de Decisión"]
C --> B
```

---

## 8. Selección de Tácticas para los Escenarios Prioritarios

| Escenario | Táctica | Efecto y Costo |
|---|---|---|
| Alcanzar una exactitud superior al 80% en la clasificación de vinos | Entrenamiento y validación del modelo utilizando conjuntos separados de entrenamiento y prueba. | Incrementa la confiabilidad de las predicciones; requiere tiempo adicional de experimentación y evaluación. |
| Responder consultas en menos de 5 segundos | Cargar el modelo entrenado en memoria al iniciar la aplicación. | Reduce los tiempos de respuesta; incrementa ligeramente el consumo de memoria. |
| Evitar errores causados por datos inválidos | Validación de los once parámetros fisicoquímicos antes de ejecutar la predicción. | Mejora la robustez del sistema, evita errores y agrega lógica de validación. |
| Facilitar el reemplazo del modelo predictivo | Separar la lógica de la API del archivo del modelo (.pkl). | Facilita las actualizaciones del modelo. |
| Garantizar la correcta operación de la API | Registro de las solicitudes hechas y errores. | Facilita el diagnóstico futuro y mantenimiento; genera archivos adicionales. |

---

## 9. Registro de Decisiones y Trade-offs (ADR)

Para nuestro registro de decisiones se toman dos que son esenciales para nuestro proyecto: el primero es la adopción de una arquitectura en capas, y el segundo la selección de un Árbol de Decisión como modelo de clasificación. Además, se documentan detalles de implementación técnica, selección de ecosistema y almacenamiento.

| Campo del ADR | Contenido |
|---|---|
| **Título y estado** | **ADR-01: Adopción de una arquitectura en capas para el sistema Sommelier Algorítmico.** (Aceptada) |
| **Contexto** | El proyecto requiere clasificar vinos como Premium o Estándar mediante un modelo de Machine Learning expuesto a través de una API local. Los drivers principales son la precisión del modelo, la rapidez de respuesta y la mantenibilidad del sistema. Entre las alternativas consideradas se evaluaron microservicios, arquitectura orientada a eventos y monolito modular. |
| **Decisión** | Se adopta una arquitectura en capas compuesta por una capa de presentación, una capa de aplicación y una capa de Machine Learning. La API actuará como intermediaria entre el usuario y el modelo predictivo. |
| **Consecuencias** | **Beneficios:** Implementación sencilla, fácil mantenimiento, menor complejidad y facilidad para reemplazar el modelo en futuras versiones. **Costos o trade-offs:** Menor capacidad de escalamiento independiente en comparación con microservicios y mayor dependencia entre componentes ejecutados en el mismo entorno. |
| **Validación** | La decisión se considerará adecuada si la arquitectura permite alcanzar una exactitud superior al 80% en las pruebas del modelo, un recall superior al 80% sobre la clase Premium, responder consultas en menos de 5 segundos y permitir la sustitución del modelo sin modificar la interfaz de la API. |

| Campo del ADR | Contenido |
|---|---|
| **Título y estado** | **ADR-02: Utilización de un Árbol de Decisión como modelo de clasificación.** (Aceptada) |
| **Contexto** | El sistema debe clasificar vinos utilizando únicamente variables fisicoquímicas y generar resultados comprensibles para los interesados del proyecto. |
| **Decisión** | Utilizar un algoritmo de Árbol de Decisión entrenado con el *UCI Wine Quality Dataset* para la clasificación binaria Premium/Estándar. |
| **Consecuencias** | **Beneficios:** Fácil interpretación, tiempos de inferencia bajos y simplicidad de implementación. **Trade-offs:** Puede ofrecer menor precisión que modelos más complejos como Random Forest o XGBoost. |
| **Validación** | Comparar la precisión, recall y F1-score obtenidos durante la fase de evaluación del modelo. |

| Campo del ADR | Contenido |
|---|---|
| **Título y estado** | **ADR-03: Enfoque de Implementación (Python, FastAPI y Scikit-Learn)** (Aceptada) |
| **Contexto** | Se requiere un ecosistema con soporte nativo para EDA (Análisis Exploratorio de Datos) y despliegue rápido de endpoints. |
| **Decisión** | Python fue seleccionado sobre otros lenguajes debido a su madurez en Machine Learning y su integración ágil. |
| **Consecuencias** | Aumenta la velocidad de desarrollo, aunque exige un estricto control de versiones de paquetes y dependencias. |
| **Validación** | Construcción de una API funcional probada localmente. |

| Campo del ADR | Contenido |
|---|---|
| **Título y estado** | **ADR-04: Almacenamiento y Despliegue (Serialización .pkl)** (Aceptada) |
| **Contexto** | Se requiere que el sistema prediga resultados rápidamente sin tener que re-entrenar el modelo con cada solicitud HTTP entrante. |
| **Decisión** | Exportar el modelo a un archivo binario `.pkl` (Pickle). |
| **Consecuencias** | Optimiza el tiempo de respuesta (latencia), pero crea el riesgo de incompatibilidades si la versión de Scikit-Learn cambia en producción. |
| **Validación** | Ejecución de predicciones en menos de 1 segundo de latencia. |

---

## 10. Riesgos Técnicos y Mitigación

Para la implementación de este proyecto identificamos tres riesgos principales que pueden afectar el rendimiento del modelo, junto con la forma en que los vamos a manejar en el código:

1. **Datos Desbalanceados:** Revisando el dataset de la UCI, la gran mayoría de los vinos tienen una calificación media (5 o 6), y hay muy pocos ejemplos de vinos "Premium" reales. El riesgo aquí es que el modelo se vuelva perezoso y asuma probabilísticamente que casi todo es calidad estándar. Para mitigarlo, podemos usar técnicas de sobremuestreo como SMOTE para balancear las clases, o utilizar el parámetro `class_weight` para que el modelo penalice más equivocarse con un vino Premium.
2. **Sobreajuste (Overfitting):** Los árboles de decisión tienden a crecer demasiado y aprenderse los datos de entrenamiento de memoria. Si esto pasa, el modelo funcionaría perfecto en nuestro entorno local, pero fallaría al evaluar nuevos vinos en un laboratorio real. La mitigación más directa es limitar qué tanto puede crecer el árbol, ajustando el hiperparámetro de profundidad máxima (por ejemplo, `max_depth = 5`) al momento de entrenarlo en Scikit-Learn.
3. **Fuga de Información (Data Leakage):** Si por error mezclamos los datos y el modelo llega a ver la información de prueba durante su entrenamiento, las métricas nos van a mentir diciendo que el modelo es casi perfecto. Para evitar este engaño, desde la etapa de preparación de datos aplicamos un `train_test_split` estricto (separando el 20% de los datos solo para evaluación final) y nos aseguramos de no hacer ninguna transformación global antes de dividir el dataset.
4. **Limitaciones de las variables fisicoquimicas:** los 11 parámetros químicos tienen una correlación débil con la calidad percibida por los catadores, lo que puede limitar la precisión que puede alcanzarel modelo sin importar qué tan bien se ajusten los parámetros. Se documenta como una limitación conocida del dataset que igual se necesita intervención humana, pero no se documenta como un error de implementación.

---

## 11. Calidad y Evaluación del Modelo

Para evaluar qué tan bien funciona nuestro modelo de clasificación, decidimos no quedarnos solo con el *Accuracy* (Exactitud), ya que como mencionamos antes, los datos están desbalanceados (hay muchos más vinos normales que excelentes). Por eso, nos trazamos la meta de superar el 80% en cuatro métricas clave para asegurar que el modelo sirva en la vida real:

* **Accuracy (Exactitud):** Nos dice el porcentaje total de veces que nuestro modelo acertó, ya sea prediciendo correctamente un vino Premium o uno Estándar.
* **Precision (Precisión):** De todos los vinos que nuestro modelo etiquetó como "Premium", ¿cuántos lo eran realmente? Esto es clave para no embotellar un vino malo haciéndolo pasar por uno caro.
* **Recall (Exhaustividad):** De todos los vinos que *realmente* eran Premium en el viñedo, ¿cuántos logró detectar nuestro modelo? La idea es que no se nos escape ninguno bueno.
* **F1-Score:** Es un promedio armónico entre Precision y Recall. Es la métrica en la que más nos fijamos para saber si el modelo es estable y no nos está mintiendo por culpa del desbalance de datos.

**¿Cómo interpretamos la Matriz de Confusión en este proyecto?**
Para entender los errores y aciertos del algoritmo desde el punto de vista del negocio de la bodega, planteamos la matriz así:
* **Verdadero Positivo (VP):** Nuestro algoritmo dice que es un vino Premium, y efectivamente lo es. ¡Perfecto para vender a buen precio!
* **Falso Positivo (FP):** Nuestro algoritmo predice que es Premium, pero en realidad el vino es de mala calidad. Este es un riesgo gravísimo porque destruiría el prestigio de la bodega si un cliente exigente lo compra.
* **Verdadero Negativo (VN):** El algoritmo descarta el vino como "Estándar" y acierta, ya que efectivamente era de calidad media o baja.
* **Falso Negativo (FN):** El algoritmo rechaza el vino diciendo que es Estándar, pero en la vida real era un Premium excelente. Aquí la bodega pierde plata al vender un producto de alta gama a precio de supermercado.

---

## 12. Análisis Ético y Conclusiones

Como equipo de trabajo, decidimos no cerrar el proyecto únicamente mostrando código y diagramas, sino también analizando el impacto real que tendría nuestro *Sommelier Algorítmico* si una bodega vitivinícola lo implementara en la vida real. Tras nuestra evaluación, llegamos a las siguientes conclusiones:

* **Balance Técnico y de Negocio:** A nivel de ingeniería, logramos diseñar una arquitectura que cumple con los objetivos principales: es modular, responde rápido y ayuda al viñedo a predecir la calidad de un lote para evitar perder tiempo y dinero en el añejamiento de productos defectuosos. Sin embargo, reconocemos que las métricas de Machine Learning son solo una parte del éxito del sistema.
* **Equidad y el Sesgo Industrial:** Nos dimos cuenta de que depender exclusivamente de variables fisicoquímicas crea un sesgo fuerte hacia la "estandarización". Si usamos el modelo a ciegas, terminaríamos castigando a los vinos artesanales o ecológicos, que por naturaleza presentan más variaciones en su pH o acidez. El sistema podría etiquetarlos erróneamente como de baja calidad solo porque no encajan en la fórmula matemática perfecta de una fábrica grande.
* **Impacto Social y Cultural:** Debatimos sobre lo perjudicial que sería intentar automatizar o reemplazar el trabajo de los catadores con nuestro algoritmo. Reducir la calidad de un vino a simples datos estadísticos le quita todo el valor humano y tradicional a la enología, y podría empujar al mercado a producir vinos genéricos solo para "complacer a la máquina".
* **Reflexión y Recomendación Final:** Por todo lo anterior, concluimos que nuestra herramienta debe catalogarse estrictamente como un sistema de apoyo a la toma de decisiones. El objetivo principal del software es ahorrarle costos de laboratorio a la empresa detectando de forma temprana lo que evidentemente está dañado; pero **nunca** debe reemplazar el paladar humano. La cata final y la última palabra sobre el producto siempre debe tenerla el experto enólogo.

## 13. Desarrollo del proyecto

Inicialmente creamos el repositorio en GitHub y tomamos el link que usaremos para la configuración en VS Code que será el IDE que usaremos para el montaje y ejecución del modelo. 

![Configuración del repositorio](Pantallazos/1.png)

Luego de que ya tenemos conectado nuestro repositorio y hemos creado las ramas del proyecto acorde a las exigencias del proyecto, nos disponemos a montar un "Virtual environment" y a instalar las herramientas que usaremos, en nuestro caso pandas para cargar, limpiar y manipular los datos en el CSV del vino tinto y blanco, scikit-learn para crear el modelo de árbol de decisión, matplotlib para crear gráficos y visualizaciones, fastapi que nos permite convertir el modelo en una API, y uvicorn que es un servidor que ejecuta aplicaciones FastAPI.

![Instalación de librerías en entorno virtual](Pantallazos/2.png)

Después que teníamos instaladas las librerías y de intentar la ejecución de un intento inicial de código, nos dimos cuenta que el código entero se ejecutaría sin permitirnos ver el paso por paso de como está avanzando el proyecto, los resultados y que cambios hay que aplicar, aunque los resultados del entrenamiento inicial nos arrojó que nuestro modelo ya superaba las métricas de Accuracy y Recall que tenían exactitud del 82% y exhaustividad del 86%, para estas 2 métricas ya habíamos fijado una meta mínima de 80% y 70% respectivamente, lo que podemos anotar como un éxito hasta ahora:

![Ejecución inicial del código](Pantallazos/3.png)

![Resultados de las métricas iniciales](Pantallazos/4.png)

Sin embargo las métricas de Precision y F1-Score nos dieron porcentajes de 42% y 56% respectivamente, por lo que tuvimos que acudir a nuestra matriz de confusión, porque sencillamente la métrica de precisión, nos dice cuantos de los vinos que el modelo etiquetó como Premium lo eran de verdad, resultando 42%, lo que significa que el modelo es un poco optimista en termino de etiquetar un vino de calidad “Premium” dejando pasar unos de calidad estándar dentro de este grupo, pero no permitió que los vinos premium se perdieran como estándar. Así ratificamos con este modelo inicial que sigue siendo necesario un enólogo, sommelier o técnico vitivinícola para evitar hayan esos falsos positivos antes de embotellar, sin embargo, no se perdieron vinos de calidad Premium debido a la precisión en este modelo. 

![Matriz de confusión](Pantallazos/5.png)

Como parte del proceso es mejorar a medida que se van dando los resultados que se van guardando en la base de datos, se espera mejorar estos valores en futuras generaciones del modelo en el archivo .pkl que es el reemplazable.
Igualmente como se puede evidenciar en los archivos, se creó el modelo base para la API de nombre “wine_quality_model.pkl” tal y como se detalló en la parte inicial de este proyecto.

![Generación del archivo .pkl](Pantallazos/6.png)

Ahora, decidimos irnos con la instalación de Jupyter Notebook, así podríamos ver cada parte del proceso para sustentarlo. La única actualización que no genera ningún cambio en lo comprometido en el proyecto es que vamos a pasar de tener un archivo .py a uno .ipynb.

En este archivo se explica paso a paso el funcionamiento de nuestro modelo predictivo, y empezamos por la carga del modelo de datos, y una muestra de los datos a analizar y procesar:

![Carga y exploración de datos](Pantallazos/7.png)

Luego pasamos a la definición de distribución y etiquetado de la calidad:

![Etiquetado de la variable de calidad](Pantallazos/8.png)

Separamos los datos y definimos qué será usado para el entrenamiento y qué se irá para el testeo del modelo:

![Separación de datos](Pantallazos/9.png)

Seguimos con el entrenamiento del modelo y la generación del archivo .pkl que va a ser el cerebro de nuestra API

Analizamos los resultados basados en la matriz de confusión, como lo hicimos anteriormente, y finalmente hacemos pruebas de nuestro modelo simulando una API que es la que entregaremos en entregas futuras del proyecto. Después de este segundo análisis nos dimos cuenta que aunque propusimos superar el 80% en las cuatro métricas, nos encontramos que esto no es alcanzable con este dataset en el que solo el 20% de los vinos son Premium, y las variables fisicoquímicas no explican completamente la calidad sensorial del vino. Por eso después de eso priorizamos el Recall (86,05%), para no dejar pasar vinos Premium por error, aceptando una Precision más baja (41,57%) como un trade-off consciente y documentado del proyecto.

## 13.1 Implementación interfaz web y modelo de recomendación de vinos

Descripción

Se realiza el aporte correspondiente al desarrollo de la aplicación Sommelier Algorítmico, orientada a la clasificación y recomendación de vinos mediante técnicas de Machine Learning.

Codigo implementado:

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

1.	Cargarmos del dataset local y utilizamos sep=';' porque el dataset que descargamos tiene separador punto y coma, igualmente decidimos iniciar con los vinos rojos.

data = pd.read_csv('winequality-red.csv', sep=';')

2.	Definimos la etiqueta binaria de clasificación de los vinos 

(Premium >= 7, Estándar < 7)
data['Premium'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

3.	Separarmos las 11 variables de entrada (X) y la etiqueta (y)

X = data.drop(['quality', 'Premium'], axis=1)
y = data['Premium']

4.	Hacemos un split estricto antes de la transformación (asi verificamos tener aparte los datos para test y entrenamiento).

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

5.	Entrenarmos el modelo a traves del arbol de decision (Decision Tree Classifier) con profundidad máxima de 5 y balanceando las clases
modelo = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
modelo.fit(X_train, y_train)
6.	Ejecutamos las predicciones sobre el conjunto de prueba
y_pred = modelo.predict(X_test)
7.	Imprimimos las métricas que definimos inicialmente en la propuesta de proyecto (Accuracy, Precision, Recall y F1-Score)
print("--- MÉTRICAS DE EVALUACIÓN (Clase Positiva: Premium) ---")
print(f"Accuracy (Exactitud): {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision (Precisión): {precision_score(y_test, y_pred):.2f}")
print(f"Recall (Exhaustividad): {recall_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
8.	Exportamos el modelo para usarlo en la API
with open('wine_quality_model.pkl', 'wb') as f:
pickle.dump(modelo, f)
print("\nModelo guardado exitosamente como 'wine_quality_model.pkl'")
9.	Mostramos la gráfica de la Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Estándar (0)', 'Premium (1)'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - Sommelier Algorítmico")
plt.show()

Interfaz visual:
 <img width="1242" height="749" alt="interfaz" src="https://github.com/user-attachments/assets/ed387e0c-9a8e-4364-b803-d3c64a4beea0" />

Resultado esperado de la prueba
<img width="850" height="197" alt="prueba 1" src="https://github.com/user-attachments/assets/90f43475-12b9-4705-a138-4e02fe4f5259" />

Cambios realizados

En este aporte se desarrolló la interfaz web de la aplicación Sommelier Algorítmico, permitiendo al usuario ingresar las características fisicoquímicas de un vino para su análisis. Además, se integró el modelo de Machine Learning previamente entrenado para realizar la predicción y clasificación de la calidad del vino. Como complemento, se incorporó una sección que presenta recomendaciones y observaciones basadas en los resultados obtenidos. También se realizaron mejoras en el diseño visual de la aplicación para facilitar la interacción del usuario, junto con ajustes de organización y optimización del código HTML y CSS para mejorar su estructura, mantenimiento y legibilidad.

Index ajustado:
<title>🍷 Sommelier Algorítmico</title>
<style>
    body {
        font-family: Arial, sans-serif;
        background: #f4f4f4;
        margin: 0;
        padding: 30px;
    }

    .contenedor {
        max-width: 800px;
        margin: auto;
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 5px 20px rgba(0, 0, 0, 0.15);
    }

    h1 {
        text-align: center;
        color: #722F37;
    }

    .subtitulo {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }

    .grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
    }

    .campo {
        display: flex;
        flex-direction: column;
    }

    label {
        font-weight: bold;
        margin-bottom: 5px;
    }

    input {
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 6px;
    }

    button {
        width: 100%;
        margin-top: 25px;
        padding: 15px;
        background: #722F37;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
    }

    button:hover {
        background: #5A252B;
    }

    .resultado {
        margin-top: 30px;
        padding: 25px;
        border-left: 6px solid #722F37;
        background: #fafafa;
        border-radius: 10px;
    }

    .vino {
        font-size: 28px;
        font-weight: bold;
        color: #722F37;
    }

    .recomendacion {
        margin-top: 15px;
        font-size: 16px;
        line-height: 1.6;
    }

    .recomendacion h3 {
        color: #722F37;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    .recomendacion ul {
        text-align: left;
        display: inline-block;
        padding-left: 20px;
    }

    .recomendacion li {
        margin-bottom: 8px;
    }
</style>
<h1>🍷 Sommelier Algorítmico</h1>

<p class="subtitulo">
    Clasificación inteligente de vinos mediante Machine Learning
</p>

<form action="/predecir" method="POST">

    <div class="grid">

        <div class="campo">
            <label>Acidez fija</label>
            <input type="number" step="0.01" name="fixed_acidity" required>
        </div>

        <div class="campo">
            <label>Acidez volátil</label>
            <input type="number" step="0.01" name="volatile_acidity" required>
        </div>

        <div class="campo">
            <label>Ácido cítrico</label>
            <input type="number" step="0.01" name="citric_acid" required>
        </div>

        <div class="campo">
            <label>Azúcar residual</label>
            <input type="number" step="0.01" name="residual_sugar" required>
        </div>

        <div class="campo">
            <label>Cloruros</label>
            <input type="number" step="0.001" name="chlorides" required>
        </div>

        <div class="campo">
            <label>Dióxido de azufre libre</label>
            <input type="number" step="0.01" name="free_sulfur_dioxide" required>
        </div>

        <div class="campo">
            <label>Dióxido de azufre total</label>
            <input type="number" step="0.01" name="total_sulfur_dioxide" required>
        </div>

        <div class="campo">
            <label>Densidad</label>
            <input type="number" step="0.00001" name="density" required>
        </div>

        <div class="campo">
            <label>pH</label>
            <input type="number" step="0.01" name="ph" required>
        </div>

        <div class="campo">
            <label>Sulfatos</label>
            <input type="number" step="0.01" name="sulphates" required>
        </div>

        <div class="campo">
            <label>Alcohol (%)</label>
            <input type="number" step="0.1" name="alcohol" required>
        </div>

    </div>

    <button type="submit">
        🍷 Analizar Calidad del Vino
    </button>

</form>

{% if resultado %}

<div class="resultado">

    <div class="vino">
        {{ resultado }}
    </div>

    {% if comentarios %}
    <div class="recomendacion">

        <h3>🔍 Características detectadas</h3>

        <ul>
            {% for comentario in comentarios %}
            <li>{{ comentario }}</li>
            {% endfor %}
        </ul>

    </div>
    {% endif %}

    <div class="recomendacion">

        <h3>📋 Recomendación</h3>

        <p>{{ recomendacion }}</p>

    </div>

</div>

{% endif %}
::::::::

Resultado esperado de la prueba
<img width="814" height="450" alt="mejora 1" src="https://github.com/user-attachments/assets/54e2d7d2-cbd0-468a-a065-9868c88f033a" />

Tecnologías utilizadas:

Python
Flask
HTML5
CSS3
Scikit-Learn
Machine Learning


Clasificación de calidad.
Características detectadas.
Recomendaciones asociadas al resultado obtenido.

Rama origen
horjuela81

Rama destino
dev_equiroga92

Rama principal Develop

---

## 14. Glosario

Para facilitar la lectura del proyecto a personas que no están familiarizadas con Machine Learning o con la industria vitivinícola, definimos los términos técnicos que usamos a lo largo del documento:

### 14.1 Términos de Machine Learning

| Término | Definición |
|---|---|
| **Machine Learning (Aprendizaje Automático)** | Rama de la inteligencia artificial en la que un algoritmo aprende patrones a partir de datos históricos para hacer predicciones sobre datos nuevos, sin ser programado con reglas explícitas. |
| **Clasificación binaria** | Tipo de problema en el que el modelo debe asignar cada registro a una de dos categorías. En nuestro caso: **Premium (1)** o **Estándar (0)**. |
| **Árbol de Decisión (Decision Tree)** | Algoritmo de clasificación que divide los datos mediante una secuencia de preguntas del tipo "¿el alcohol es mayor a 11,5%?", formando una estructura de ramas y hojas. Es fácil de interpretar porque cada predicción puede explicarse siguiendo el camino del árbol. |
| **Variable de entrada (Feature)** | Cada una de las características que recibe el modelo para hacer su predicción. En este proyecto son las 11 variables fisicoquímicas del vino. |
| **Variable objetivo (Target / Etiqueta)** | Lo que el modelo intenta predecir. En nuestro caso es la columna `Premium`, creada a partir de la columna `quality` (1 si la calidad es ≥ 7, 0 si es menor). |
| **Dataset** | Conjunto de datos organizado en filas (registros, en este caso vinos) y columnas (variables). |
| **Conjunto de entrenamiento (Train)** | Porción de los datos (80%) con la que el modelo aprende los patrones. |
| **Conjunto de prueba (Test)** | Porción de los datos (20%) que se reserva y que el modelo nunca ve durante el entrenamiento. Se usa únicamente para medir su desempeño real. |
| **`train_test_split`** | Función de Scikit-Learn que separa el dataset en conjuntos de entrenamiento y prueba. Usamos el parámetro `stratify` para que ambos conjuntos conserven la misma proporción de vinos Premium y Estándar. |
| **Datos desbalanceados** | Situación en la que una clase tiene muchos más registros que la otra. En nuestro dataset la gran mayoría de vinos son Estándar y muy pocos son Premium. |
| **`class_weight='balanced'`** | Parámetro del modelo que le da más peso a los errores cometidos sobre la clase minoritaria (Premium), para compensar el desbalance de los datos. |
| **SMOTE** | Técnica de sobremuestreo que genera ejemplos sintéticos de la clase minoritaria para balancear el dataset. Se consideró como alternativa de mitigación. |
| **Hiperparámetro** | Configuración del modelo que se define antes de entrenarlo y que no se aprende de los datos, por ejemplo `max_depth`. |
| **`max_depth` (Profundidad máxima)** | Hiperparámetro que limita cuántos niveles de preguntas puede tener el árbol. Lo fijamos en 5 para evitar el sobreajuste. |
| **Sobreajuste (Overfitting)** | Cuando el modelo se "aprende de memoria" los datos de entrenamiento y por eso funciona muy bien con ellos, pero falla con datos nuevos. |
| **Fuga de información (Data Leakage)** | Error en el que información del conjunto de prueba se filtra durante el entrenamiento, haciendo que las métricas parezcan mejores de lo que realmente son. |
| **Modelo de referencia (Baseline)** | Modelo muy simple (por ejemplo, uno que siempre predice "Estándar") que sirve como punto de comparación para demostrar que nuestro modelo realmente aporta valor. |
| **Inferencia** | Proceso de usar un modelo ya entrenado para predecir la categoría de un dato nuevo. |
| **Sesgo (Bias)** | Tendencia sistemática del modelo o de los datos a favorecer ciertos resultados. Por ejemplo, favorecer vinos con características "industriales" por sobre vinos artesanales. |

### 14.2 Métricas de evaluación

| Término | Definición |
|---|---|
| **Matriz de Confusión** | Tabla que compara las predicciones del modelo contra los valores reales, mostrando los aciertos y errores por cada clase. |
| **Verdadero Positivo (VP)** | El modelo predice Premium y el vino realmente es Premium. |
| **Falso Positivo (FP)** | El modelo predice Premium pero el vino realmente es Estándar. |
| **Verdadero Negativo (VN)** | El modelo predice Estándar y el vino realmente es Estándar. |
| **Falso Negativo (FN)** | El modelo predice Estándar pero el vino realmente es Premium. |
| **Accuracy (Exactitud)** | Porcentaje total de predicciones correctas: (VP + VN) / Total. |
| **Precision (Precisión)** | De todos los vinos que el modelo etiquetó como Premium, qué porcentaje realmente lo era: VP / (VP + FP). |
| **Recall (Exhaustividad / Sensibilidad)** | De todos los vinos que realmente eran Premium, qué porcentaje logró detectar el modelo: VP / (VP + FN). |
| **F1-Score** | Media armónica entre Precision y Recall. Resume en un solo valor el equilibrio entre ambas métricas y es especialmente útil con datos desbalanceados. |
| **Trade-off** | Compromiso en el que mejorar un aspecto implica empeorar otro. En nuestro proyecto priorizamos el Recall aceptando una Precision más baja. |

### 14.3 Términos de arquitectura y software

| Término | Definición |
|---|---|
| **API (Application Programming Interface)** | Interfaz que permite que otros programas o usuarios se comuniquen con el sistema enviando datos y recibiendo respuestas. Nuestra API recibe las 11 variables y devuelve la clasificación del vino. |
| **Endpoint** | Dirección específica de la API que atiende un tipo de solicitud, por ejemplo `/predecir`. |
| **JSON** | Formato de texto estándar para intercambiar datos entre sistemas, basado en pares clave-valor. |
| **Flask** | Framework ligero de Python para construir aplicaciones web y APIs. |
| **FastAPI** | Framework moderno de Python para construir APIs de alto rendimiento. |
| **Pickle (`.pkl`)** | Formato de Python para serializar (guardar en un archivo) objetos, como un modelo ya entrenado, y cargarlos después sin tener que reentrenarlo. |
| **Serialización** | Proceso de convertir un objeto en memoria (como el modelo) en un archivo que puede almacenarse y reutilizarse. |
| **Modelo C4** | Método para documentar la arquitectura de software mediante diagramas en cuatro niveles de detalle: Contexto, Contenedores, Componentes y Código. |
| **ADR (Architecture Decision Record)** | Registro breve que documenta una decisión de arquitectura importante: su contexto, la decisión tomada, sus consecuencias y cómo se validará. |
| **Driver arquitectónico** | Requisito o restricción que tiene mayor influencia en el diseño de la arquitectura, como la precisión, la latencia o la modificabilidad. |
| **Latencia** | Tiempo que tarda el sistema en responder desde que recibe una solicitud. |
| **Arquitectura en capas** | Estilo arquitectónico que organiza el sistema en niveles con responsabilidades separadas (presentación, aplicación y modelo de ML). |
| **Entorno virtual (venv)** | Carpeta aislada que contiene las librerías de Python del proyecto, evitando conflictos con otras instalaciones del equipo. |
| **Jupyter Notebook (`.ipynb`)** | Documento interactivo que combina código, resultados y texto explicativo, ejecutado por celdas. |

### 14.4 Variables fisicoquímicas del vino

| Variable (dataset) | Nombre en español | Definición |
|---|---|---|
| **fixed acidity** | Acidez fija | Ácidos que no se evaporan fácilmente (principalmente ácido tartárico). Expresada en g/dm³. |
| **volatile acidity** | Acidez volátil | Cantidad de ácido acético. En niveles altos produce un sabor desagradable a vinagre. Expresada en g/dm³. |
| **citric acid** | Ácido cítrico | Presente en pequeñas cantidades, aporta frescura al vino. Expresado en g/dm³. |
| **residual sugar** | Azúcar residual | Azúcar que queda después de terminar la fermentación. Expresada en g/dm³. |
| **chlorides** | Cloruros | Cantidad de sal en el vino. Expresada en g/dm³. |
| **free sulfur dioxide** | Dióxido de azufre libre | Forma libre del SO₂ que previene el crecimiento microbiano y la oxidación. Expresado en mg/dm³. |
| **total sulfur dioxide** | Dióxido de azufre total | Suma del SO₂ libre y combinado. En concentraciones altas se percibe en el olor y sabor. Expresado en mg/dm³. |
| **density** | Densidad | Depende principalmente del contenido de alcohol y azúcar. Expresada en g/cm³. |
| **pH** | pH | Nivel de acidez en una escala de 0 a 14. La mayoría de los vinos están entre 3 y 4. |
| **sulphates** | Sulfatos | Aditivo que contribuye a los niveles de SO₂ y actúa como antimicrobiano y antioxidante. Expresado en g/dm³. |
| **alcohol** | Alcohol | Porcentaje de alcohol por volumen (% vol.). |
| **quality** | Calidad | Puntuación sensorial de 0 a 10 asignada por catadores expertos (mediana de al menos 3 evaluaciones). Es la base para construir la etiqueta Premium/Estándar. |

---

## 15. Referencias

### 15.1 Fuente del dataset

* Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547-553. https://doi.org/10.1016/j.dss.2009.05.016
* Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/186/wine+quality

### 15.2 Herramientas y librerías

* **Python Software Foundation.** *Python Language Reference*. https://www.python.org
* **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830. https://scikit-learn.org
* **McKinney, W. (2010).** *Data Structures for Statistical Computing in Python*. Proceedings of the 9th Python in Science Conference, 56-61. https://pandas.pydata.org
* **Hunter, J. D. (2007).** *Matplotlib: A 2D Graphics Environment*. Computing in Science & Engineering, 9(3), 90-95. https://matplotlib.org
* **Pallets Projects.** *Flask Documentation*. https://flask.palletsprojects.com
* **Ramírez, S.** *FastAPI Documentation*. https://fastapi.tiangolo.com
* **Project Jupyter.** *Jupyter Notebook*. https://jupyter.org
* **Python Software Foundation.** *pickle — Python object serialization*. https://docs.python.org/3/library/pickle.html
* **Mermaid.** *Mermaid Diagramming and Charting Tool*. https://mermaid.js.org

### 15.3 Decisiones técnicas y metodología

* **Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984).** *Classification and Regression Trees*. Wadsworth. (Fundamento del algoritmo de Árbol de Decisión utilizado).
* **Scikit-learn developers.** *Decision Trees*. https://scikit-learn.org/stable/modules/tree.html
* **Scikit-learn developers.** *Metrics and scoring: quantifying the quality of predictions*. https://scikit-learn.org/stable/modules/model_evaluation.html
* **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).** *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321-357. https://doi.org/10.1613/jair.953
* **Brown, S.** *The C4 model for visualising software architecture*. https://c4model.com
* **Nygard, M. (2011).** *Documenting Architecture Decisions*. https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
