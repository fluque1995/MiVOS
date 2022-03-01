# Resultados obtenidos

## Métricas implementadas

- Índice de movimiento: Cociente entre el espacio total ocupado por el dedo en
  el vídeo y el tamaño medio en cada fotograma. Se puede calcular por ventanas
  temporales. Un valor más alto implica un mayor movimiento.
- Centro de los dedos: Utilizando la media de los píxeles ocupados por las
  máscaras de segmentación en cada fotograma, extraemos el punto central del
  dedo, lo cual nos permite medir desplazamientos completos a partir de la
  trayectoria de dicho punto.
- Gráficas de desplazamiento con respecto a la posición de equilibrio:
  Utilizando el centro de los dedos en cada fotograma, se puede calcular la
  posición de equilibrio del centro del dedo en el vídeo (media de las
  posiciones en el vídeo completo), y representar gráficamente los
  desplazamientos dicho punto respecto del equilibrio en ambos ejes. El
  resultado es una gráfica cercana a la función constante si el dedo está más o
  menos quieto, y una onda oscilante si el dedo presenta mucho movimiento.
- Frecuencia y amplitud de desplazamiento del dedo: Utilizando la trayectoria
  del centro de los dedos durante todo el vídeo, y aplicando la transformada de
  Fourier a dicha onda, podemos calcular la amplitud de movimiento del dedo
  durante el vídeo, así como la frecuencia de oscilación. Estos dos valores nos
  dan una idea de la gravedad del movimiento.
- Extracción de puntos extremos utilizando PCA: Considerando los píxeles que
  ocupa la máscara en el fotograma como una distribución de puntos, podemos
  extraer la dirección de máximo cambio (es decir, la dirección principal del
  dedo) utilizando el algoritmo PCA. Con esta información, podemos hacer el
  análisis de movimiento que hemos realizado anteriormente para el centro de los
  dedos con los puntos extremos del mismo. Esto nos permite extraer información
  de movimientos en los que el centro se mantiene más o menos fijo, pero los
  puntos extremos se mueven (movimientos pendulares).
- Normalizaciones respecto del tamaño del dedo: Para aquellos estadísticos que
  se extraen como desplazamientos del dedo, resulta interesante estandarizar la
  medida para que el resultado sea comparable independientemente de la distancia
  entre el individuo y la cámara. Para ello, se ha considerado que una medida de
  la distancia entre el individuo y la cámara viene determinada por el tamaño
  medio que ocupa el dedo en el vídeo. Dividiendo las distancias por la raíz
  cuadrada de este valor (se toma la raíz cuadrada para convertir una magnitud
  espacial en una magnitud lineal), se obtiene un valor de movimiento
  normalizado respecto de la distancia a la cámara.
