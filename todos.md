# TODOS

## Métricas

- Tener en cuenta la normalización respecto del tamaño del dedo cuando hacemos
  las gráficas de desplazamiento (quizás dividiendo la posición normalizada
  entre el tamaño que ocupa el dedo en la imagen?)
  - Resultado: Decidimos normalizar entre la raíz cuadrada del tamaño total del
    dedo, ya que el área que ocupa el dedo en la imagen es una dimensión
    cuadrática, mientras que los desplazamientos que estamos calculando son
    dimensiones lineales.
- Estudiar cómo podemos extraer la velocidad de movimiento del dedo a partir de
  la posición del centro del dedo
  - Resultado: Utilizando la transformada de Fourier de la trayectoria del
    centro del dedo, podemos calcular la frecuencia de oscilación del dedo en el
    vídeo. Una vez calculada la transformada de Fourier, el valor máximo nos da
    información de la amplitud del movimiento, y el índice de dicho valor nos da
    información sobre la frecuencia de oscilación.
- Estudiar si PCA o LinReg nos dan la dirección principal del dedo como una
  recta para estudiar el movimiento del dedo en distintos puntos (inicio vs
  punta del dedo)
  - Resultado: PCA nos da las direcciones principales del dedo, y nos permite
    localizar los puntos extremos
- Calcular el coeficiente de amplitud en ventanas de tiempo, para mitigar los
  casos en los que el dedo se mueve solamente al principio o al final del vídeo
  (durante la preparación), pero se mantiene quieto en la mayor parte de la
  grabación.
  - Resultado: Se disminuye el ruido provocado por desplazamientos en el vídeo,
    obteniendo mejores resultados

## Modelos de segmentación

- Estudiar el funcionamiento de STCN para propagación de máscaras
  - Nos quedamos con S2M basado en MobilenetV2 y STCN para propagación

## Videos

- Los vídeos que Paco tiene descargados no coinciden en nomenclatura con los de
  Laura. Revisar quién tiene los vídeos correctos. Unificar la terminología de
  vídeos (probablemente, Paciente/Visita/Prueba)
  - Los vídeos se han cotejado, encontrando que el paciente P2 tiene vídeos mal
    etiquetados. Se ha creado una nueva carpeta, Videos_filtrados, en la que
    está la información organizada.
  - El sistema de nomenclatura que se ha utilizado para marcar a los pacientes
	es PX/Visita_N/prueba, donde la prueba puede ser:
	- Dedos_enfrentados
	- D-N_der,
	- D-N_izq
	- Extension
	- Reposo
	- Suero_der
	- Suero_izq
	- Perfil_der
	- Perfil_izq

	Los perfiles no están disponibles para todos los pacientes
- Comprobar la mejor resolución a la que pueden segmentarse los vídeos
  - Actualmente, la mejor resolución que somos capaces de utilizar en nuestros
    ordenadores es de 400px. El resultado con esta resolución es razonablemente
    bueno. El principal problema que tiene la interfaz gráfica es que carga
    el vídeo completo desde el principio, por lo que si los requerimientos de
    memoria del mismo son grandes (e.g. el vídeo es muy largo o la resolución
    muy alta), el programa revienta por falta de memoria. Se debería estudiar
    si puede optimizarse la interfaz para no cargar el vídeo completo en memoria
    todo el tiempo de ejecución, si no que se fuera cargando a medida que los
    fotogramas van siendo necesarios.
    - Primera idea: Si no necesitamos propagar la máscara hacia atrás, y nos
      quedamos con la segmentación inicial, podemos cargar un fotograma en cada
      vuelta del algoritmo de segmentación.
- En el vídeo `P1/Visita_2_ON/Reposo`, el paciente no lleva puestos los dedines,
  pero la segmentación de las manos es perfecta. Es posible estudiar este
  problema con guantes completos, o directamente segmentando las manos si los
  pacientes llevan manga larga?
  - Resultado: Se van a grabar vídeos en los que se coloquen distintos elementos
    para segmentar (dedines, guantes completos, manos desnudas con prendas de
    manga larga), y se estudiará si los resultados obtenidos son adecuados, y en
    caso positivo, comparables entre sí independientemente del objeto
    segmentado.

## Problemas detectados

- La posición de los pacientes en los vídeos no es estándar (por ejemplo, en el
  vídeo de reposo se colocan las manos apoyadas sobre las palmas o sobre el
  dorso indistintamente). Creemos que las medidas estadísticas sobre el
  movimiento de los dedos en estas dos posiciones no son comparables.
- ¿Segmentar ambos dedos en todos los vídeos aunque la prueba en cuestión esté centrada solo en el derecho/izquierdo? Cuando hay dos dedos se toma el derecho como el primero y el izquierdo como el segundo, pero cuando hay solo uno se toma ese como el primero, independientemente de si es derecho o izquierdo. Esto es problemático en la extracción automática de resultados, ya que, en ese caso, se guardan todas las estadísticas como si fueran del primer dedo (derecho).
  - Al etiquetar los dos dedos en todos los vídeos, los resultados son consistentes y se pueden estudiar los dedos por separado.
- Valores raros del índice de movimiento en los vídeos dedo-nariz y gráficas de movimiento poco informativas.
- Hay dos funciones plot_movements, una en plotting.py y otra en statistics.py. La que usamos es la de plotting: ¿eliminar la otra?
