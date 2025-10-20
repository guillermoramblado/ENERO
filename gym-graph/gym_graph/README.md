Para generar los diferentes entornos de trabajo personalizados, y usando la librería 'gym', se sigue el siguente procedimiento:

1. Se crea un subdirectorio 'gym_graph', que contiene:
    * Directorio /envs, que contiene:
        - Directorio 'envs' con los diferentes entornos de trabajo personalizados
        - _init_.py donde importamos los diferentes entornos de trabajo
    * _init_ --> usamos función 'register' del módulo 'gym.envs' para registrar los diferentes entornos creados. Para cada entorno a registrar, se debe indicar:
        - ID de dicho entorno
        - Ruta de la clase que implementa dicho entorno personalizado

2. Dentro de gym_graph (raíz), se define 'setup.py', donde especificamos las dependencias requeridas (como Gym, numpy...) para lanzar nuestros entornos personalizados. La estructura del setup.py es la siguiente:
    * Se debe importar la función 'setup' del módulo 'setuptools'
    * Usando 'setup', se debe indicar:
        - Nombre del paquete que contiene los entornos personalizados. En este caso, el subdirectorio 'gym_graph' (dentro del cual hemos definido el _init_ y los /envs). 
        - Versión
        - Dependencias requeridas en el parámetro 'install_requires' de la función 'setup'


----------- ENTRENAMIENTO --------------

Se entrena el agente DRL usando train_Enero_3top.py:

Primero será necesario usar convert_dataset.py para cada una de las topologías que se usarán para entrenar el agente, obteniendo así dos carpetas (para cada topología) con las matrices de tráfico
usadas para entrenamiento y para vadalición, respectivamente.
Se ejecuta el script 'train_Enero_3top.py', que ejecuta, para cada episodio, el script 'train_Enero_3top_script.py'

---- Obtener gráficos sobre la evaluación del modelo en ciertas topologías -------

Para cada topología....
1. Ejecutar convert_dataset.py, encargado de dividir las matrices de tráfico de la topología en TRAIN y VAL.
2. Ejecutar el archivo eval_on_single_topology.py pasando la topología sobre la que queremos evaluar el modelo entrenado. En dicho archivo, se creará la carpeta evalRes_NEW_<Topologia> donde se guardarán
   los resultados de evaluar el modelo en las matrices de validación de dicha topología.
   * Internamente, se ejecutará script_eval_on_single_topology.py para cada una de las matrices de validación, usando Enero (agente DRL entrenado + Local Search)

Una vez hecho eso para todas las topologías --> Generamos el gráfico ejecutando figures_5_and_6.py



-------- ¿Cómo se actualiza el actor/crítico? -----------


Tras recopilar experiencia a lo largo de una trayectoria:
1. Se calcula la ventaja conseguida en cada uno de los estados de la trayectoria, a excepción del último estado alcanzado
2. Se procede a actualizar recorriendo 'PPO_EPOCHS' veces la misma trayectoria, tal que para cada iteración, se procesa por lotes dicha trayectoria. 
   * Cada lote se procesa usando '_train_step_combined', que permite actualizar usando dicho lote:
      - Internamente, dicho método realiza lo siguiente para cada estado del lote:
         - Invoca _actor_step y _critic_step sobre dicho estado, volviendo a aplicar el actor/crítico para trabajar con la nueva política y poder así calcular la pérdida del actor/crítico.
      - Tras procesar la pérdida en los diferentes estados del lote:
            * Se calcula la pérdida media, y se usa esa pérdida media (similar a calcular la media de los gradientes) para calcular los gradientes de los parámetros del modelo entrenables (tanto del actor como del crítico) usando tape.gradient
            * Una vez calculados dichos gradientes, se actualizan los parámetros del modelo usando dichos gradientes, cuya forma de actualizar dependerá del optimizador empleado.


---------------------  DEPENDENCIAS INSTALADAS ------------------
* He instalado la última versión disponible de las diferentes dependencias del proyecto, a excepción de la librería gym, ya que la ultima versión me obligaba a definir el espacio de acciones y observaciones. Por tanto, he instalado una versión más antiguo.

* Tb he eliminado el uso del paquete 'resource' en train_Enero_3top.py

