# TODOS

## Métricas

- Tener en cuenta la normalización respecto del tamaño del dedo cuando hacemos
  las gráficas de desplazamiento (quizás dividiendo la posición normalizada
  entre el tamaño que ocupa el dedo en la imagen?)
- Estudiar cómo podemos extraer la velocidad de movimiento del dedo a partir de
  la posición del centro del dedo
- Estudiar si PCA o LinReg nos dan la dirección principal del dedo como una
  recta para estudiar el movimiento del dedo en distintos puntos (inicio vs
  punta del dedo)
  - Resultado: PCA nos da las direcciones principales del dedo, y nos permite
    localizar los puntos extremos
- Calcular el coeficiente de amplitud en ventanas de tiempo, para mitigar los
  casos en los que el dedo se mueve solamente al principio o al final del vídeo
  (durante la preparación), pero se mantiene quieto en la mayor parte de la
  grabación

## Modelos de segmentación

- Estudiar el funcionamiento de STCN para propagación de máscaras

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
