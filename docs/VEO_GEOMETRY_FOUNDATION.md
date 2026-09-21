# VEO Geometry Foundation

> Fundamentos técnicos de la reconstrucción geométrica de Veo: desde los recursos y la calibración de cámara hasta la conversión de píxeles en coordenadas físicas del terreno.

## Índice

- [1. Del problema inicial al modelo geométrico de Veo](#capitulo-1)
  - [1.1. Objetivo, evolución de la solución y enfoques descartados](#sec-1-1)
  - [1.2. Primera aproximación: homografía imagen-campo](#sec-1-2)
  - [1.3. Primer obstáculo: la cámara no era una única homografía estable](#sec-1-3)
  - [1.4. HUMAN anchors y Camera Atlas](#sec-1-4)
  - [1.5. Propagación y refinamiento de homografías](#sec-1-5)
  - [1.6. El problema de soporte y extrapolación](#sec-1-6)
  - [1.7. Segundo obstáculo: la distorsión de las lentes](#sec-1-7)
  - [1.8. Intento de estimar la óptica desde las imágenes](#sec-1-8)
  - [1.9. Descubrimiento del vídeo de las dos lentes](#sec-1-9)
  - [1.10. Intento de stitching clásico LEFT/RIGHT](#sec-1-10)
  - [1.11. El cambio decisivo: reconstruir la cámara de Veo](#sec-1-11)
  - [1.12. Qué aprendimos de los enfoques descartados](#sec-1-12)
  - [1.13. Timeline resumida](#sec-1-13)
  - [1.14. Punto de partida de la solución actual](#sec-1-14)
- [2. Datos y calibración obtenidos de Veo](#capitulo-2)
  - [2.1. Qué información obtenemos de Veo y qué significa realmente](#sec-2-1)
  - [2.2. Cómo localizamos los recursos de un partido](#sec-2-2)
  - [2.3. El vídeo de Seguimiento de balón](#sec-2-3)
  - [2.4. El vídeo Dual Lens](#sec-2-4)
  - [2.5. `camera_alignment.veo`: la pieza principal de calibración](#sec-2-5)
  - [2.6. Parámetros intrínsecos: cómo una dirección termina en un píxel](#sec-2-6)
  - [2.7. Modelo de distorsión de las lentes](#sec-2-7)
  - [2.8. Transformaciones 4×4: qué significan y por qué aparecen](#sec-2-8)
  - [2.9. `lens_left2camera` y `lens_right2camera`](#sec-2-9)
  - [2.10. `camera2world`: de la cámara al sistema físico del partido](#sec-2-10)
  - [2.11. `camera2presentation`: orientación de la vista panorámica](#sec-2-11)
  - [2.12. `camera2logic`: sistema utilizado por la cámara de Seguimiento](#sec-2-12)
  - [2.13. Dimensiones físicas del terreno](#sec-2-13)
  - [2.14. `va_camera`, `va_logic` y `va_presentation`](#sec-2-14)
  - [2.15. `selected_corners`: anclas observadas durante la calibración](#sec-2-15)
  - [2.16. `worldmap_version` y métricas adicionales de calibración](#sec-2-16)
  - [2.17. `camera_directions.det`](#sec-2-17)
  - [2.18. Los shaders: una fuente especialmente valiosa de evidencia](#sec-2-18)
  - [2.19. Cómo selecciona el shader una dirección panorámica](#sec-2-19)
  - [2.20. El remapeo vertical de la textura Dual Lens](#sec-2-20)
  - [2.21. Cómo mezcla las dos lentes](#sec-2-21)
  - [2.22. Las cuatro visualizaciones y de qué dependen](#sec-2-22)
  - [2.23. Qué información cambia entre partidos y qué parece pertenecer al sistema de cámara](#sec-2-23)
  - [2.24. Qué podemos afirmar y qué no](#sec-2-24)
  - [2.25. Resumen de dependencias geométricas](#sec-2-25)
- [3. Fundamento geométrico: de píxel a mundo](#capitulo-3)
  - [3.1. De un píxel a una posición real sobre el terreno](#sec-3-1)
  - [3.2. Sistemas de coordenadas utilizados](#sec-3-2)
  - [3.3. El modelo pinhole como punto de partida](#sec-3-3)
  - [3.4. Del píxel a un rayo, no a un punto 3D](#sec-3-4)
  - [3.5. Intersección rayo-plano: el paso que convierte imagen en metros](#sec-3-5)
  - [3.6. Casos en los que la intersección no es válida](#sec-3-6)
- [4. Doble lente: inversión óptica y proyección al campo](#capitulo-4)
  - [4.1. Doble lente: píxel RAW → rayo → campo](#sec-4-1)
  - [4.2. Identificación de la lente](#sec-4-2)
  - [4.3. Invertir el remapeo vertical del transcode](#sec-4-3)
  - [4.4. Volver al espacio 3840×2160](#sec-4-4)
  - [4.5. Por qué no basta con multiplicar por \(K^{-1}\)](#sec-4-5)
  - [4.6. Inversión numérica de la distorsión](#sec-4-6)
  - [4.7. Construcción del rayo de una lente](#sec-4-7)
  - [4.8. Por qué Doble lente no necesita una homografía del campo](#sec-4-8)
- [5. Panorámica e Interactivo](#capitulo-5)
  - [5.1. Panorámica: un píxel representa una dirección angular](#sec-5-1)
  - [5.2. Longitude y latitude → dirección 3D](#sec-5-2)
  - [5.3. Espacio de presentación → mundo](#sec-5-3)
  - [5.4. Interactivo: una cámara virtual dentro del mismo mundo](#sec-5-4)
  - [5.5. Píxel Interactivo → coordenadas del viewport](#sec-5-5)
  - [5.6. FOV y por qué aparece la tangente](#sec-5-6)
  - [5.7. Rotaciones de la cámara virtual](#sec-5-7)
  - [5.8. Por qué mover Interactivo no rompe la geometría](#sec-5-8)
- [6. Seguimiento de balón y cámara dinámica](#capitulo-6)
  - [6.1. Seguimiento de balón: el caso dinámico](#sec-6-1)
  - [6.2. Tiempo → estado de cámara](#sec-6-2)
  - [6.3. Estado de cámara Tracking → mundo](#sec-6-3)
  - [6.4. FOV de Tracking → tamaño del viewport](#sec-6-4)
  - [6.5. Píxel Tracking → dirección local](#sec-6-5)
- [7. Sistema canónico, campo 2D y validación métrica](#capitulo-7)
  - [7.1. Cuatro caminos diferentes, una misma operación final](#sec-7-1)
  - [7.2. El sistema canónico del terreno](#sec-7-2)
  - [7.3. Cómo se dibuja `(X,Z)` sobre el campo 2D](#sec-7-3)
  - [7.4. Por qué el campo 2D es una validación útil](#sec-7-4)
  - [7.5. Validación mediante landmarks conocidos](#sec-7-5)
  - [7.6. Landmarks utilizados](#sec-7-6)
  - [7.7. Estadísticos de validación](#sec-7-7)
  - [7.8. Qué significa una precisión de unos pocos metros](#sec-7-8)
  - [7.9. Diferencia entre error de geometría y error de selección](#sec-7-9)
  - [7.10. Por qué un pequeño error angular puede convertirse en metros](#sec-7-10)
  - [7.11. Por qué el punto de apoyo de un jugador será importante](#sec-7-11)
  - [7.12. La escala espacial y la escala temporal quedan unificadas](#sec-7-12)
  - [7.13. Qué parte está validada actualmente](#sec-7-13)
  - [7.14. Una asimetría pendiente: mundo → algunas vistas](#sec-7-14)
  - [7.15. Por qué no corregimos ese problema con un offset fijo](#sec-7-15)
  - [7.16. Posible corrección residual futura](#sec-7-16)
  - [7.17. Resumen matemático de la conversión](#sec-7-17)
  - [7.18. Por qué podemos confiar razonablemente en esta construcción](#sec-7-18)
  - [7.19. Resultado de esta etapa](#sec-7-19)
- [8. Evidencia experimental, limitaciones y estado actual](#capitulo-8)
  - [8.1. Validación experimental: cómo comprobamos que la reconstrucción funciona](#sec-8-1)
  - [8.2. Primera validación fuerte: reproducir la panorámica de Veo](#sec-8-2)
  - [8.3. El caso `camera2presentation`: un error de convención que la validación detectó](#sec-8-3)
  - [8.4. Validación con dos partidos diferentes](#sec-8-4)
  - [8.5. Comparación CPU frente al renderer WebGL](#sec-8-5)
  - [8.6. Qué valida realmente la coincidencia de panorama](#sec-8-6)
  - [8.7. Validación de la geometría RAW mediante inversión del mismo modelo](#sec-8-7)
  - [8.8. Por qué este round-trip es útil](#sec-8-8)
  - [8.9. Validación en las dos lentes](#sec-8-9)
  - [8.10. Validación métrica mediante landmarks del terreno](#sec-8-10)
  - [8.11. Cómo calculamos el error de un landmark](#sec-8-11)
  - [8.12. Resultados métricos observados](#sec-8-12)
  - [8.13. Por qué excluimos un clic claramente erróneo](#sec-8-13)
  - [8.14. No ajustar la calibración con los mismos puntos usados para validarla](#sec-8-14)
  - [8.15. Evolución de la validación de Tracking](#sec-8-15)
  - [8.16. Descubrimiento posterior de `camera_directions.det`](#sec-8-16)
  - [8.17. Validación temporal del `.det`](#sec-8-17)
  - [8.18. Validación funcional Tracking → campo](#sec-8-18)
  - [8.19. El problema de la proyección inversa en Tracking](#sec-8-19)
  - [8.20. Decisión: desactivar el overlay antes que ocultar el error](#sec-8-20)
  - [8.21. Experimento con biases manuales](#sec-8-21)
  - [8.22. Por qué rechazar un bias fijo es importante](#sec-8-22)
  - [8.23. Registro visual residual como experimento](#sec-8-23)
  - [8.24. Por qué la homografía residual quedó aparcada](#sec-8-24)
  - [8.25. Validación de Interactivo](#sec-8-25)
  - [8.26. Consistencia entre visualizaciones](#sec-8-26)
  - [8.27. Qué significa "funciona" en este proyecto](#sec-8-27)
  - [8.28. Tabla de estado actual](#sec-8-28)
  - [8.29. Limitaciones actuales](#sec-8-29)
  - [8.30. Niveles de confianza](#sec-8-30)
  - [8.31. Por qué no necesitamos perfección antes de empezar visión artificial](#sec-8-31)
  - [8.32. Qué queda fundamentado después de esta validación](#sec-8-32)
  - [8.33. Siguiente consecuencia técnica](#sec-8-33)
- [9. Arquitectura futura y análisis automático](#capitulo-9)
  - [9.1. Arquitectura conceptual final](#sec-9-1)
  - [9.2. La idea más importante: separar percepción de geometría](#sec-9-2)
  - [9.3. Pipeline futuro de jugadores](#sec-9-3)
  - [9.4. Del track a una posición física](#sec-9-4)
  - [9.5. Por qué conviene detectar en Dual Lens](#sec-9-5)
  - [9.6. Solape entre lentes](#sec-9-6)
  - [9.7. Clasificación de equipo y rol](#sec-9-7)
  - [9.8. Identidad de jugador](#sec-9-8)
  - [9.9. Balón](#sec-9-9)
  - [9.10. Tiempo como segundo sistema canónico](#sec-9-10)
  - [9.11. Eventos como información temporal](#sec-9-11)
  - [9.12. Qué NO concluimos de los highlights](#sec-9-12)
  - [9.13. Diferencia entre evento y vídeo de highlight](#sec-9-13)
  - [9.14. Otros datasets que puede exponer la plataforma](#sec-9-14)
  - [9.15. Qué información de Veo forma parte de la base del proyecto](#sec-9-15)
  - [9.16. Coexistencia entre datos de Veo y análisis propio](#sec-9-16)
  - [9.17. Correcciones manuales sin destruir resultados automáticos](#sec-9-17)
  - [9.18. Métricas derivables cuando tengamos tracks físicos](#sec-9-18)
  - [9.19. Heatmaps](#sec-9-19)
  - [9.20. Qué no forma parte todavía del sistema](#sec-9-20)
- [10. Reproducibilidad y cierre técnico](#capitulo-10)
  - [10.1. Reproducibilidad de la geometría](#sec-10-1)
  - [10.2. Reproducción mínima de la panorámica](#sec-10-2)
  - [10.3. Reproducción mínima de pixel → campo](#sec-10-3)
  - [10.4. Qué datos deben pertenecer al mismo partido](#sec-10-4)
  - [10.5. Por qué utilizamos más de un partido para validar](#sec-10-5)
  - [10.6. Dependencia de versiones](#sec-10-6)
  - [10.7. Qué deberíamos validar en un partido nuevo](#sec-10-7)
  - [10.8. Posibles mejoras futuras de la calibración](#sec-10-8)
  - [10.9. Riesgos que debemos evitar en el futuro](#sec-10-9)
  - [10.10. Qué hemos construido realmente](#sec-10-10)
  - [10.11. Por qué esta base es útil para el proyecto del Real Jaén](#sec-10-11)
  - [10.12. Resumen final](#sec-10-12)
  - [10.13. Conclusión](#sec-10-13)

<a id="capitulo-1"></a>

## 1. Del problema inicial al modelo geométrico de Veo

<a id="sec-1-1"></a>

### 1.1. Objetivo, evolución de la solución y enfoques descartados

> Este documento describe la base técnica que permite transformar información visual procedente de una grabación de Veo en coordenadas físicas sobre un terreno de juego. No documenta la aplicación web utilizada durante el desarrollo; la aplicación es únicamente una herramienta para inspeccionar, validar y experimentar con esta geometría.

#### 1.1.1. El problema que queríamos resolver

El objetivo inicial puede formularse de manera muy sencilla:

> Dado un punto visible en un vídeo de fútbol, queremos saber dónde se encuentra físicamente ese punto sobre el terreno de juego.

En una imagen o en un vídeo, una posición se expresa normalmente mediante coordenadas de píxel:

\[
(u,v)
\]

donde \(u\) indica la posición horizontal en la imagen y \(v\) la posición vertical.

Estas coordenadas son útiles para tareas de visión artificial, pero por sí solas no tienen significado físico. Por ejemplo, saber que el centro inferior de la caja de un jugador se encuentra en el píxel

\[
(1350,720)
\]

no nos dice si ese jugador está a cinco metros del centro del campo, dentro del área o junto a la banda.

Para poder realizar análisis deportivo necesitamos terminar en otro sistema de coordenadas:

\[
(X,Z)
\]

donde \(X\) y \(Z\) representan una posición sobre el terreno medida en metros.

Ésta es la transformación fundamental del proyecto:

\[
\boxed{
\text{píxel de imagen}
\longrightarrow
\text{posición física en el campo}
}
\]

Una vez obtenida, muchas tareas que inicialmente parecen problemas independientes pasan a compartir la misma base. Por ejemplo, una futura detección de jugadores podría seguir el pipeline:

```text
detección de jugador
        ↓
bounding box
        ↓
punto de apoyo aproximado sobre el césped
        ↓
píxel
        ↓
posición (X,Z) en metros
```

Y si repetimos el proceso a lo largo del tiempo:

```text
track_id
+
[(t0, X0, Z0), (t1, X1, Z1), ...]
```

podremos construir trayectorias, distancias recorridas, velocidades, ocupación espacial, heatmaps y estructuras tácticas.

Por tanto, antes de centrarnos en detectar o seguir jugadores, necesitábamos resolver de forma fiable el problema geométrico.

---

#### 1.1.2. Por qué no basta con detectar jugadores

Los detectores modernos de objetos trabajan principalmente en espacio de imagen. Su salida típica es una caja:

\[
B=(x,y,w,h)
\]

junto con una clase y una confianza.

Por ejemplo:

```text
person
confidence = 0.94
bbox = (1180, 410, 75, 190)
```

Esto permite dibujar correctamente la detección sobre el vídeo, pero no permite comparar directamente posiciones entre cámaras o estados de cámara diferentes.

Un jugador puede permanecer quieto sobre el césped y, sin embargo, desplazarse cientos de píxeles en la imagen si la cámara gira o cambia de zoom.

De forma inversa, dos jugadores que están separados muchos metros pueden aparecer muy próximos en la imagen debido a la perspectiva.

La proyección de una escena tridimensional sobre una imagen pierde información de profundidad. Por eso era necesario reconstruir la geometría de la cámara.

La pregunta real dejó de ser:

> ¿Dónde está el jugador en la imagen?

y pasó a ser:

> ¿Qué rayo del espacio tridimensional corresponde a ese píxel y dónde corta ese rayo al plano del terreno de juego?

Ésta es la idea que finalmente sustenta la solución actual, aunque llegar hasta ella requirió varios enfoques previos.

<a id="sec-1-2"></a>

### 1.2. Primera aproximación: homografía imagen-campo

La primera estrategia fue tratar el terreno de juego como un plano y buscar una transformación proyectiva entre la imagen y un modelo 2D del campo.

Es una aproximación clásica y razonable para este problema.

#### 1.2.1. Qué es una homografía

Una homografía es una transformación proyectiva entre dos planos.

Si el terreno de juego se considera plano, un punto del campo puede expresarse como:

\[
P_f=
\begin{bmatrix}
X\\
Z\\
1
\end{bmatrix}
\]

y el punto correspondiente de la imagen como:

\[
p=
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
\]

La relación entre ambos puede escribirse como:

\[
\lambda p = H P_f
\]

es decir:

\[
\lambda
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
=
\begin{bmatrix}
h_{11} & h_{12} & h_{13}\\
h_{21} & h_{22} & h_{23}\\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
\begin{bmatrix}
X\\
Z\\
1
\end{bmatrix}
\]

donde:

- \(H\) es una matriz \(3\times3\);
- \(\lambda\) es un factor de escala debido al uso de coordenadas homogéneas;
- \((X,Z)\) representa un punto físico del campo;
- \((u,v)\) representa su proyección en la imagen.

Como una matriz homogénea está definida salvo una escala global, una homografía posee ocho grados de libertad efectivos.

En su forma explícita:

\[
u=
\frac{
h_{11}X+h_{12}Z+h_{13}
}{
h_{31}X+h_{32}Z+h_{33}
}
\]

\[
v=
\frac{
h_{21}X+h_{22}Z+h_{23}
}{
h_{31}X+h_{32}Z+h_{33}
}
\]

Con suficientes correspondencias entre puntos conocidos del campo y puntos marcados en la imagen es posible estimar \(H\).

En el caso mínimo ideal hacen falta cuatro correspondencias de puntos no degeneradas. En la práctica es preferible disponer de más información y resolver el sistema mediante mínimos cuadrados, DLT normalizado o métodos robustos como RANSAC.

#### 1.2.2. Por qué esta idea tenía sentido

Un campo de fútbol es casi plano.

Además, contiene geometría conocida:

- esquinas;
- línea central;
- puntos de penalti;
- límites de las áreas;
- círculo central;
- intersecciones entre líneas.

Esto permite asociar puntos visibles de la imagen con posiciones conocidas en metros.

Por ejemplo:

```text
esquina inferior izquierda del campo
imagen: (u1, v1)
mundo:  (-52.5, -34.0)

centro del campo
imagen: (u2, v2)
mundo:  (0.0, 0.0)
```

Con suficientes correspondencias podríamos estimar una homografía y, utilizando su inversa, transformar cualquier píxel perteneciente al césped a coordenadas del campo:

\[
P_f \sim H^{-1}p
\]

Después se normaliza la coordenada homogénea:

\[
X=\frac{P_{f,x}}{P_{f,w}}
\qquad
Z=\frac{P_{f,z}}{P_{f,w}}
\]

En una cámara fija y con óptica suficientemente aproximable mediante un modelo proyectivo ideal, esta solución puede ser excelente.

El problema era que nuestra situación real era bastante más complicada.

<a id="sec-1-3"></a>

### 1.3. Primer obstáculo: la cámara no era una única homografía estable

El vídeo de seguimiento de Veo no es una cámara fija convencional.

La vista cambia a lo largo del partido para seguir la acción:

- desplazamiento horizontal;
- desplazamiento vertical;
- variación de apertura/zoom;
- cambios de encuadre.

Esto significa que una homografía obtenida en un frame no puede reutilizarse directamente durante todo el vídeo.

Podríamos escribir:

\[
H=H(t)
\]

para indicar que la transformación depende del instante.

El problema se convertía entonces en estimar la cámara para cada momento del partido.

<a id="sec-1-4"></a>

### 1.4. HUMAN anchors y Camera Atlas

Para evitar calibrar manualmente miles de frames, exploramos una estrategia basada en un conjunto reducido de vistas de referencia.

La idea era construir una especie de atlas de estados de cámara.

#### 1.4.1. Idea general

Se seleccionaban frames representativos de distintas vistas del terreno.

Sobre esos frames se realizaba una calibración manual más fiable. Estos estados anotados se denominaban de forma conceptual **HUMAN anchors**.

El sistema podía representarse así:

```text
vídeo completo
     ↓
selección de vistas representativas
     ↓
HUMAN anchors
     ↓
Camera Atlas
     ↓
buscar el estado más parecido para cada frame
     ↓
transportar/refinar su geometría
```

El objetivo era reducir el trabajo humano:

> En lugar de calibrar todos los frames, calibramos unas pocas vistas suficientemente representativas e intentamos transferir su geometría al resto del vídeo.

#### 1.4.2. Selección automática de vistas diferentes

Para elegir los anchors no queríamos seleccionar frames aleatoriamente.

Se experimentó con descriptores visuales orientados al fútbol que trataban de caracterizar:

- cantidad y distribución del césped visible;
- envolvente superior del terreno;
- estructura de líneas;
- densidad de bordes;
- composición general de la vista.

También se utilizaron características locales ORB para comparar de forma más precisa un conjunto reducido de candidatos.

ORB (*Oriented FAST and Rotated BRIEF*) combina:

1. detección de puntos de interés mediante FAST;
2. asignación de orientación;
3. descriptores binarios BRIEF rotados.

Dos imágenes pueden compararse mediante la distancia de Hamming entre descriptores.

Para eliminar correspondencias ambiguas se puede utilizar el *ratio test*. Si \(d_1\) es la distancia del mejor candidato y \(d_2\) la del segundo:

\[
\frac{d_1}{d_2}<\tau
\]

la correspondencia se conserva únicamente cuando el mejor emparejamiento es claramente mejor que el segundo.

Posteriormente RANSAC permite estimar una transformación robusta ignorando matches incompatibles con el modelo.

Esto se utilizó para distinguir vistas prácticamente duplicadas y estimar cambios relativos de pan y escala entre candidatos.

#### 1.4.3. Reducción de dimensionalidad con PCA

También exploramos si la variación de los estados de cámara podía representarse con pocas dimensiones.

PCA (*Principal Component Analysis*) busca las direcciones que explican la mayor parte de la variabilidad de unos datos.

Si tenemos estados:

\[
x_i\in\mathbb{R}^n
\]

calculamos la media:

\[
\mu=\frac1N\sum_{i=1}^{N}x_i
\]

centramos los datos:

\[
\tilde{x}_i=x_i-\mu
\]

y calculamos su covarianza:

\[
C=\frac1{N-1}\sum_i \tilde{x}_i\tilde{x}_i^T
\]

Los autovectores de \(C\) definen las direcciones principales de variación. Si unos pocos componentes concentran casi toda la varianza, podemos aproximar:

\[
x_i\approx
\mu+
\sum_{k=1}^{m}a_{ik}v_k
\]

con \(m\ll n\).

Esto apoyaba una intuición importante: la cámara no recorre configuraciones totalmente arbitrarias, sino un espacio estructurado de vistas.

Sin embargo, describir de forma compacta un estado visual no equivale a conocer con precisión su geometría física.

<a id="sec-1-5"></a>

### 1.5. Propagación y refinamiento de homografías

A partir de los anchors estudiamos cómo transportar una calibración conocida desde una vista a otra.

También probamos correcciones pequeñas aplicadas sobre una proyección ya existente.

La filosofía era:

```text
proyección previa razonable
        +
evidencia geométrica escasa del frame actual
        ↓
pequeña corrección
```

En lugar de recalcular siempre una homografía desde cero, se evaluaron familias de corrección de complejidad creciente:

```text
traslación
→ similitud
→ afín
→ homografía
```

#### 1.5.1. Traslación

\[
T=
\begin{bmatrix}
1&0&t_x\\
0&1&t_y\\
0&0&1
\end{bmatrix}
\]

Corrige únicamente un desplazamiento.

#### 1.5.2. Transformación de similitud

Añade escala y rotación:

\[
S=
\begin{bmatrix}
s\cos\theta & -s\sin\theta & t_x\\
s\sin\theta & s\cos\theta & t_y\\
0&0&1
\end{bmatrix}
\]

Permite compensar pequeños cambios de zoom, giro y desplazamiento.

#### 1.5.3. Transformación afín

\[
A=
\begin{bmatrix}
a_{11}&a_{12}&t_x\\
a_{21}&a_{22}&t_y\\
0&0&1
\end{bmatrix}
\]

Puede representar traslación, rotación, escala no uniforme y cizallamiento.

#### 1.5.4. Corrección proyectiva

Finalmente puede utilizarse una homografía completa:

\[
Q=
\begin{bmatrix}
1+a_{11}&a_{12}&t_x\\
a_{21}&1+a_{22}&t_y\\
p_1&p_2&1
\end{bmatrix}
\]

Los parámetros \(p_1,p_2\) introducen el componente propiamente proyectivo.

En nuestros experimentos estas correcciones se regularizaban alrededor de la identidad. La idea no era permitir que el optimizador inventase una cámara completamente diferente, sino corregir una predicción previa.

Conceptualmente se minimizaba:

\[
E(\theta)
=
E_{\text{datos}}(\theta)
+
\lambda E_{\text{prior}}(\theta)
\]

donde:

- \(E_{\text{datos}}\) mide el error respecto a puntos/líneas anotados;
- \(E_{\text{prior}}\) penaliza alejarnos demasiado de la transformación identidad;
- \(\lambda\) controla cuánto confiamos en la proyección inicial.

Esta fase fue útil porque permitió estudiar de forma cuantitativa dónde fallaba la transferencia geométrica.

<a id="sec-1-6"></a>

### 1.6. El problema de soporte y extrapolación

Uno de los resultados más interesantes de los experimentos con homografías fue comprobar que el error dependía mucho de qué región del terreno estaba respaldada por las anotaciones.

Si calibramos una vista utilizando puntos concentrados en una zona del campo, la transformación puede ajustar muy bien esa zona y degradarse al extrapolar lejos de ella.

Para estudiarlo se analizaron conceptos como:

- extensión horizontal de los puntos utilizados para ajustar;
- área de soporte en coordenadas del campo;
- envolvente convexa del soporte;
- distancia de puntos de validación a dicha envolvente.

Si disponemos de puntos de calibración:

\[
P_1,P_2,\ldots,P_n
\]

podemos calcular su envolvente convexa:

\[
\mathcal{H}
=
\operatorname{ConvHull}
(P_1,\ldots,P_n)
\]

Un punto de prueba dentro de \(\mathcal{H}\) se encuentra en una región interpolada por la calibración.

Un punto muy alejado de \(\mathcal{H}\) obliga al modelo a extrapolar.

Esto no invalida matemáticamente la homografía, pero sí aumenta la sensibilidad a:

- errores de anotación;
- distorsión óptica no modelada;
- pequeñas diferencias entre vistas;
- errores de transferencia temporal.

También se realizaron pruebas de sensibilidad perturbando las anotaciones con ruido.

Si:

\[
p_i' = p_i + \epsilon_i
\]

con:

\[
\epsilon_i\sim\mathcal{N}(0,\sigma^2)
\]

podemos recalcular muchas veces la homografía y medir cuánto cambia la predicción.

Esta simulación Monte Carlo permite comprobar cuánta variación puede producir una incertidumbre de apenas uno o unos pocos píxeles en los puntos anotados.

La conclusión fue que la homografía era una herramienta útil para investigar y validar, pero no queríamos construir el sistema completo sobre calibraciones manuales y extrapolaciones difíciles de controlar.

<a id="sec-1-7"></a>

### 1.7. Segundo obstáculo: la distorsión de las lentes

Una homografía describe la proyección entre planos bajo un modelo de cámara proyectivo ideal.

Sin embargo, una lente gran angular introduce una transformación óptica adicional.

Si un punto normalizado de cámara es:

\[
(x,y)
\]

definimos:

\[
r^2=x^2+y^2
\]

y un modelo radial típico introduce términos:

\[
r^2,\qquad r^4,\qquad r^6
\]

La posición final depende así de la distancia al centro óptico de forma no lineal.

Esto es conceptualmente distinto a una homografía.

Una homografía tiene una forma racional como:

\[
u=
\frac{aX+bZ+c}{gX+hZ+i}
\]

\[
v=
\frac{dX+eZ+f}{gX+hZ+i}
\]

mientras que la distorsión óptica depende de \(r\) y puede curvar líneas que deberían ser rectas bajo un modelo pinhole ideal.

Por tanto, una única homografía puede compensar parte de esa deformación localmente, pero no describe de forma exacta la óptica en toda la imagen.

Este punto fue fundamental: algunas correcciones que estábamos intentando resolver en espacio de imagen pertenecían en realidad al modelo físico de la lente.

<a id="sec-1-8"></a>

### 1.8. Intento de estimar la óptica desde las imágenes

También estudiamos si podíamos estimar la distorsión observando las líneas del campo.

La intuición es válida:

> Una línea recta del mundo debería proyectarse como una recta bajo un modelo pinhole ideal. Si aparece curvada, esa curvatura contiene información sobre la lente.

Pero la estimación robusta planteaba problemas:

- no siempre vemos líneas largas;
- existen oclusiones;
- la perspectiva dificulta separar visualmente efectos;
- ambas lentes pueden ser diferentes;
- hacen falta suficientes observaciones.

Esta vía era técnicamente posible, pero perdió sentido al descubrir posteriormente que Veo ya almacenaba parámetros de calibración asociados a la grabación.

<a id="sec-1-9"></a>

### 1.9. Descubrimiento del vídeo de las dos lentes

Durante la investigación del reproductor localizamos un recurso diferente al vídeo de seguimiento del balón.

Tiene resolución:

\[
2048\times2048
\]

y contiene las dos lentes apiladas verticalmente:

```text
0
┌────────────────────────────┐
│      LENTE IZQUIERDA       │ 2048 × 1024
├────────────────────────────┤ y = 1024
│       LENTE DERECHA        │ 2048 × 1024
└────────────────────────────┘
2048
```

Esto fue decisivo porque disponíamos de las observaciones de ambas lentes antes de la composición panorámica final.

Ya no estábamos obligados a trabajar únicamente con una vista de seguimiento procesada.

<a id="sec-1-10"></a>

### 1.10. Intento de stitching clásico LEFT/RIGHT

La primera idea fue coser ambas imágenes mediante visión por computador convencional:

```text
LEFT
   \
    → features
    → matching
    → RANSAC
    → transformación
   /
RIGHT
```

Podemos, por ejemplo, estimar una homografía:

\[
p_L\sim H_{R\rightarrow L}p_R
\]

a partir de correspondencias visuales en la zona común.

Esto puede producir una panorámica visualmente convincente, sobre todo si la escena está lejana y el solape es limitado.

Pero nuestro objetivo no era simplemente generar una imagen agradable.

Necesitábamos una relación físicamente interpretable.

Las lentes:

- poseen ópticas propias;
- presentan distorsión;
- están montadas con una relación rígida tridimensional;
- observan direcciones diferentes.

Una homografía estimada con features no explica por separado esos fenómenos ni nos proporciona directamente una ruta robusta hacia coordenadas métricas.

Por tanto, el stitching clásico quedó como experimento y no como fundamento del sistema.

<a id="sec-1-11"></a>

### 1.11. El cambio decisivo: reconstruir la cámara de Veo

La estrategia cambió al localizar un fichero de calibración asociado a la grabación:

```text
camera_alignment-....veo
```

Este recurso contiene información relacionada con:

- parámetros intrínsecos;
- distorsión de las lentes;
- transformación de cada lente respecto a la cámara;
- transformación de la cámara respecto al mundo;
- transformaciones utilizadas para la presentación;
- dimensiones reales del terreno.

Hasta entonces estábamos resolviendo:

```text
imagen
→ estimar la cámara
→ aproximar la geometría
```

Desde ese momento el objetivo pasó a ser:

```text
calibración de Veo
→ reconstruir el modelo de cámara
→ comprobar que reproduce lo que Veo muestra
→ utilizar ese modelo para obtener geometría física
```

Este cambio también mejora la interpretabilidad.

En lugar de absorber todos los errores en una única transformación 2D, podemos separar:

```text
óptica de la lente
↓
proyección
↓
orientación de cada lente
↓
cámara
↓
mundo
↓
terreno
```

Cada paso tiene un significado geométrico concreto y puede validarse de forma independiente.

<a id="sec-1-12"></a>

### 1.12. Qué aprendimos de los enfoques descartados

Los primeros experimentos no fueron trabajo perdido.

Sirvieron para identificar qué información necesitábamos y qué errores debíamos evitar.

| Enfoque | Qué aportó | Por qué no quedó como solución principal |
|---|---|---|
| Homografía manual | Primera conversión imagen-campo y validación métrica | Dependencia de anotaciones, vista dinámica y óptica no modelada |
| HUMAN anchors / Camera Atlas | Estudio de diversidad de vistas y propagación | Seguía dependiendo de calibraciones manuales |
| PCA / reducción dimensional | Evidencia de estructura en los estados de cámara | Una representación compacta no implica geometría física exacta |
| Refinamiento proyectivo | Cuantificación del error y sensibilidad | Corrige el efecto en imagen sin explicar necesariamente su causa física |
| Estimación visual de distorsión | Identificó la óptica como fuente de error | Veo ya proporcionaba calibración |
| Stitching LEFT/RIGHT | Comprensión del solape entre lentes | Insuficiente como modelo geométrico físico |

<a id="sec-1-13"></a>

### 1.13. Timeline resumida

```text
Problema:
"¿Dónde está físicamente un punto que veo en el vídeo?"
        │
        ▼
Homografía imagen → campo
        │
        ├── útil en vistas concretas
        └── cámara dinámica + distorsión
                │
                ▼
HUMAN anchors + Camera Atlas
                │
                ├── selección de vistas
                ├── propagación
                ├── PCA / diversidad
                └── refinamiento
                        │
                        ▼
Soporte, extrapolación y sensibilidad
                        │
                        └── demasiada dependencia manual
                                │
                                ▼
Estudio de distorsión óptica
                                │
                                ▼
Descubrimiento del vídeo Dual Lens
                                │
                                ▼
Stitching LEFT/RIGHT
                                │
                                └── útil visualmente,
                                    insuficiente geométricamente
                                        │
                                        ▼
Descubrimiento de camera_alignment.veo
                                        │
                                        ▼
Cambio de estrategia:
de estimar la cámara
a reconstruir la cámara de Veo
```

<a id="sec-1-14"></a>

### 1.14. Punto de partida de la solución actual

La conclusión de esta primera etapa fue que la transformación que necesitamos no debía basarse principalmente en ajustes empíricos entre imágenes.

Debía partir del modelo físico de la cámara.

El pipeline conceptual pasó a ser:

\[
(u,v)
\longrightarrow
\text{modelo óptico}
\longrightarrow
\text{rayo 3D}
\longrightarrow
\text{mundo}
\longrightarrow
(X,Z)
\]

o visualmente:

```text
PÍXEL
  ↓
LENTE
  ↓
CÁMARA
  ↓
MUNDO 3D
  ↓
INTERSECCIÓN CON EL CÉSPED
  ↓
COORDENADAS DEL CAMPO EN METROS
```

A partir de este cambio de estrategia, la solución se desarrolla alrededor de seis elementos:

1. qué información obtenemos de Veo;
2. qué significa cada parámetro de calibración;
3. cómo reconstruimos matemáticamente las visualizaciones;
4. cómo convertimos un píxel en un rayo 3D;
5. cómo intersectamos ese rayo con el terreno;
6. cómo obtenemos finalmente \((X,Z)\) en metros.

#### 1.14.1. Nota sobre el nivel de evidencia

En el resto del documento diferenciaremos cuando sea necesario entre:

- **dato observado en Veo:** aparece directamente en recursos, metadatos, ficheros o comportamiento del reproductor;
- **reconstrucción propia:** comportamiento que hemos deducido y reproducido experimentalmente;
- **fundamento matemático estándar:** geometría proyectiva, modelos de cámara, distorsión, matrices de transformación e intersección rayo-plano.

Esta separación es importante para justificar el sistema sin tratar a Veo como una caja negra y sin atribuirle algoritmos internos concretos cuando no exista evidencia directa.



<a id="capitulo-2"></a>

## 2. Datos y calibración obtenidos de Veo

<a id="sec-2-1"></a>

### 2.1. Qué información obtenemos de Veo y qué significa realmente

El cambio de estrategia más importante del proyecto fue dejar de intentar estimar toda la cámara a partir de imágenes y empezar a reconstruir el modelo geométrico que Veo ya utiliza.

A partir de aquí conviene separar muy bien tres conceptos:

1. **los vídeos que contienen las imágenes**;
2. **los metadatos que describen la cámara y su calibración**;
3. **la lógica de visualización que transforma esos datos en las vistas que muestra Veo**.

La solución actual no depende de un único archivo "mágico". Funciona porque hemos podido relacionar varias piezas que, juntas, describen suficientemente el sistema de cámara.

De forma resumida:

```text
Vídeo Dual Lens
    │
    ├── imagen de la lente izquierda
    └── imagen de la lente derecha

camera_alignment.veo
    │
    ├── intrínsecos
    ├── distorsión
    ├── lens_left2camera
    ├── lens_right2camera
    ├── camera2world
    ├── camera2presentation
    ├── camera2logic
    ├── campo
    └── rangos angulares

camera_directions.det
    │
    └── estado dinámico de la cámara virtual
        del vídeo de Seguimiento

shaders / lógica del reproductor
    │
    └── cómo combina Veo estas piezas
```

La ventaja de disponer de esta información es que podemos evitar tratar cada visualización como una imagen independiente. Las distintas vistas pasan a ser manifestaciones diferentes de un mismo sistema geométrico.

---

<a id="sec-2-2"></a>

### 2.2. Cómo localizamos los recursos de un partido

Los recursos importantes no aparecen necesariamente todos como descargas evidentes en la interfaz de Veo.

Una parte de la investigación consistió en observar qué carga el reproductor y, sobre todo, qué información contienen las respuestas JSON que recibe.

Esto produjo una regla importante para el proyecto:

> No basta con buscar archivos por extensión en las peticiones de red. Un recurso puede estar referenciado dentro de los metadatos de otro recurso.

El ejemplo más importante es `camera_directions.det`.

#### 2.2.1. `/videos/` como fuente de asociación

En las respuestas de metadatos de vídeo aparecen objetos que describen diferentes representaciones del partido.

Entre ellos encontramos recursos con tipos como:

```text
standard
panorama
```

La representación `standard` corresponde al vídeo de seguimiento que Veo presenta inicialmente al usuario.

En ese mismo objeto puede aparecer un campo:

```text
camera_directions
```

que contiene la URL del fichero de direcciones de cámara asociado a **ese vídeo concreto**.

La asociación es importante.

No hacemos:

```text
"he encontrado un .det; probablemente será éste"
```

sino:

```text
vídeo Standard seleccionado
        ↓
buscar su objeto de metadatos
        ↓
leer SU camera_directions
```

En la implementación actual la identidad se comprueba utilizando la URL normalizada del vídeo, ignorando firmas temporales o query strings, pero exigiendo que el recurso base coincida.

Esto evita mezclar por accidente:

- un vídeo de highlights;
- un clip individual;
- otro vídeo Standard;
- unas direcciones pertenecientes a otro recurso.

---

#### 2.2.2. El vídeo inicial visible como referencia de Seguimiento

Otro problema que encontramos fue que Veo puede manejar varios MP4 etiquetados o relacionados con `standard`.

No todos representan necesariamente el partido completo siguiendo el balón.

La referencia más fiable que encontramos fue:

> El vídeo reproducido por el player principal antes de activar la vista Interactive es el vídeo de Seguimiento del partido.

Por eso el `currentSrc` del vídeo inicialmente visible tiene prioridad como fuente de Tracking.

Después utilizamos `/videos/` para encontrar los metadatos asociados exactamente a ese recurso.

Conceptualmente:

```text
player inicial de Veo
       ↓
currentSrc
       ↓
vídeo de Seguimiento
       ↓
objeto correspondiente en /videos/
       ↓
camera_directions
```

Esta decisión surgió después de comprobar que una heurística demasiado amplia podía confundir un vídeo generado para highlights con el vídeo principal del partido.

---

<a id="sec-2-3"></a>

### 2.3. El vídeo de Seguimiento de balón

La primera visualización es la más parecida a un vídeo de fútbol convencional.

En los partidos estudiados se presenta como un vídeo aproximadamente:

\[
1920\times1080
\]

con relación de aspecto:

\[
\frac{16}{9}
\]

y una tasa de frames próxima a:

\[
\frac{30000}{1001}
\approx
29.97003\ \text{fps}
\]

No es una de las lentes físicas de la cámara.

Es una **vista virtual ya renderizada**, cuyo encuadre cambia a lo largo del partido.

La cámara virtual gira y cambia su campo de visión para seguir la acción.

Por eso un píxel de Tracking no puede convertirse a metros usando una única homografía estática.

Necesitamos conocer, para cada instante:

- yaw;
- pitch;
- FOV.

Esa información es precisamente la que conseguimos recuperar mediante `camera_directions.det`.

---

<a id="sec-2-4"></a>

### 2.4. El vídeo Dual Lens

El segundo recurso fundamental es el transcode asociado al panorama.

En los partidos estudiados es un vídeo:

\[
2048\times2048
\]

pero no representa una imagen panorámica convencional.

Contiene dos imágenes de:

\[
2048\times1024
\]

apiladas verticalmente.

```text
y = 0
┌────────────────────────────┐
│                            │
│      LENTE IZQUIERDA       │
│        2048 × 1024         │
│                            │
├────────────────────────────┤ y = 1024
│                            │
│       LENTE DERECHA        │
│        2048 × 1024         │
│                            │
└────────────────────────────┘
y = 2048
```

Por tanto, un píxel global:

\[
(u,v)
\]

pertenece a la lente izquierda si:

\[
0\leq v<1024
\]

y a la derecha si:

\[
1024\leq v<2048
\]

Para trabajar en coordenadas locales de cada lente:

#### 2.4.1. Lente izquierda

\[
u_L=u
\]

\[
v_L=v
\]

#### 2.4.2. Lente derecha

\[
u_R=u
\]

\[
v_R=v-1024
\]

Esta separación parece trivial, pero es muy importante: la calibración contiene parámetros diferentes para cada lente y la transformación geométrica debe saber de cuál procede el píxel.

---

#### 2.4.3. Por qué Dual Lens es nuestra fuente preferida para análisis futuro

El vídeo de Seguimiento ya ha pasado por una cámara virtual que:

- selecciona una orientación;
- aplica un FOV;
- recorta una región;
- produce un vídeo final 16:9.

Dual Lens conserva mucha más información de la captura original.

Eso lo convierte en una fuente especialmente útil para futuras tareas de:

- detección de personas;
- tracking;
- detección del balón;
- proyección al campo.

La idea futura no es analizar únicamente lo que la cámara de seguimiento decide mostrar, sino aprovechar las dos lentes disponibles.

---

<a id="sec-2-5"></a>

### 2.5. `camera_alignment.veo`: la pieza principal de calibración

El fichero:

```text
camera_alignment-<id>.veo
```

es un JSON que contiene parámetros asociados a la calibración del partido.

En las dos grabaciones que hemos utilizado aparece:

```text
calibration_version = 6.2
```

Entre sus campos encontramos:

```text
intrinsic_left
intrinsic_right

distortion_left
distortion_right

lens_left2camera
lens_right2camera

camera2world
camera2presentation
camera2logic

field_length
field_width

va_camera
va_logic
va_presentation

selected_corners
```

No todos estos campos realizan la misma función. Para entender por qué el sistema funciona hay que distinguir:

```text
INTRÍNSECOS
cómo una lente transforma direcciones en píxeles

EXTRÍNSECOS / TRANSFORMACIONES
cómo se relacionan entre sí los sistemas de coordenadas

METADATOS DEL CAMPO
qué dimensiones físicas tiene el terreno

RANGOS DE PRESENTACIÓN
qué región angular se utiliza para construir ciertas vistas
```

---

<a id="sec-2-6"></a>

### 2.6. Parámetros intrínsecos: cómo una dirección termina en un píxel

Cada lente tiene una matriz intrínseca.

Su forma general es:

\[
K=
\begin{bmatrix}
f_x&s&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}
\]

En nuestras calibraciones el término de skew \(s\) es cero, por lo que podemos interpretarla aproximadamente como:

\[
K=
\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}
\]

Los términos tienen un significado físico/proyectivo claro.

#### 2.6.1. \(f_x,f_y\): focal expresada en píxeles

Controlan cuánto cambia la posición proyectada cuando cambia el ángulo del rayo.

Una focal grande produce una vista aparentemente más cerrada; una focal menor produce un campo de visión más amplio.

No es necesario interpretar estos valores directamente como milímetros. En la matriz de cámara están expresados en unidades de píxel del espacio de calibración.

#### 2.6.2. \(c_x,c_y\): punto principal

Representan el punto en el que aproximadamente intersecta el eje óptico con el plano de imagen.

No tiene por qué coincidir exactamente con el centro geométrico de la imagen.

---

#### 2.6.3. Ejemplo real

En una de las calibraciones observamos aproximadamente:

\[
K_L=
\begin{bmatrix}
2247.24&0&1897.56\\
0&2247.24&1101.49\\
0&0&1
\end{bmatrix}
\]

y:

\[
K_R=
\begin{bmatrix}
2244.00&0&1972.13\\
0&2244.00&1120.04\\
0&0&1
\end{bmatrix}
\]

Estos parámetros no están construidos para la textura final de \(2048\times2048\).

El shader observado indica explícitamente:

```glsl
// Intrinsic matrix is built for 4K image
coord.xy / vec2(1920.0 * 2.0, 1080.0 * 2.0);
```

es decir, la matriz intrínseca trabaja en un espacio de referencia:

\[
3840\times2160
\]

Posteriormente las coordenadas se normalizan y se remapean hacia la textura Dual Lens.

Este detalle fue esencial para poder reproducir correctamente la panorámica.

---

<a id="sec-2-7"></a>

### 2.7. Modelo de distorsión de las lentes

La matriz intrínseca por sí sola describe una cámara pinhole ideal.

Pero las lentes reales, especialmente cuando cubren ángulos amplios, introducen distorsión.

El shader de Veo contiene explícitamente ocho parámetros:

```text
k1
k2
p1
p2
k3
k4
k5
k6
```

y la operación matemática observada coincide con el **modelo racional de distorsión** utilizado habitualmente en visión artificial.

Es importante formularlo de manera precisa:

> Podemos afirmar que el shader observado implementa esta fórmula. No necesitamos asumir qué librería utilizó Veo para obtener los coeficientes durante su proceso de calibración.

---

#### 2.7.1. Coordenadas normalizadas

Antes de aplicar la distorsión trabajamos con un punto ideal:

\[
(x,y)
\]

sobre el plano normalizado de cámara.

Calculamos:

\[
r^2=x^2+y^2
\]

\[
r^4=(r^2)^2
\]

\[
r^6=(r^2)^3
\]

La distancia \(r\) indica cuánto nos alejamos del eje óptico.

---

#### 2.7.2. Componente radial racional

El factor radial observado es:

\[
R(r)=
\frac{
1+k_1r^2+k_2r^4+k_3r^6
}{
1+k_4r^2+k_5r^4+k_6r^6
}
\]

y produce inicialmente:

\[
x_r=xR(r)
\]

\[
y_r=yR(r)
\]

La razón de usar potencias pares es que la deformación radial depende de la distancia al centro, no de la dirección concreta.

---

#### 2.7.3. Componente tangencial

El shader añade:

\[
\Delta x_t
=
p_2(r^2+2x^2)+2p_1xy
\]

\[
\Delta y_t
=
p_1(r^2+2y^2)+2p_2xy
\]

Por tanto:

\[
x_d=xR(r)+\Delta x_t
\]

\[
y_d=yR(r)+\Delta y_t
\]

En las calibraciones estudiadas los coeficientes tangenciales observados son cero, pero el modelo los contempla.

---

#### 2.7.4. Paso final a píxel de calibración

La coordenada distorsionada se proyecta mediante:

\[
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
=
K
\begin{bmatrix}
x_d\\
y_d\\
1
\end{bmatrix}
\]

De esta forma la cadena es:

```text
dirección 3D en la lente
        ↓
dividir por profundidad
        ↓
(x,y) normalizado
        ↓
distorsión racional/tangencial
        ↓
K
        ↓
píxel en espacio 3840×2160
```

Éste es uno de los fundamentos matemáticos más importantes del sistema.

---

<a id="sec-2-8"></a>

### 2.8. Transformaciones 4×4: qué significan y por qué aparecen

El fichero `.veo` contiene varias matrices de \(4\times4\).

No son simples tablas de números. Representan cambios entre sistemas de coordenadas tridimensionales.

Una transformación rígida puede escribirse como:

\[
T=
\begin{bmatrix}
R&t\\
0&1
\end{bmatrix}
\]

donde:

- \(R\) es una matriz \(3\times3\) de rotación;
- \(t\) es un vector de traslación;
- la última fila permite utilizar coordenadas homogéneas.

Un punto 3D se representa como:

\[
P=
\begin{bmatrix}
x\\y\\z\\1
\end{bmatrix}
\]

y puede transformarse mediante:

\[
P'=TP
\]

Una dirección, en cambio, utiliza:

\[
d=
\begin{bmatrix}
d_x\\d_y\\d_z\\0
\end{bmatrix}
\]

El cero final hace que la traslación no afecte a un vector de dirección.

Esto explica por qué en el shader aparecen expresiones como:

```glsl
vec4(0.0, 0.0, 1.0, 0.0)
```

para representar una dirección, mientras que la posición de una cámara utiliza:

```glsl
vec4(0.0, 0.0, 0.0, 1.0)
```

---

<a id="sec-2-9"></a>

### 2.9. `lens_left2camera` y `lens_right2camera`

Cada lente posee su propio sistema de coordenadas.

El eje \(Z\) de la lente, por ejemplo, representa su dirección frontal local.

Pero las dos lentes están montadas mirando hacia regiones diferentes.

Las matrices:

```text
lens_left2camera
lens_right2camera
```

describen cómo transformar una posición o dirección desde el sistema local de cada lente hasta el sistema común de la cámara.

Podemos escribir:

\[
P_C=T_{L\rightarrow C}P_L
\]

para la izquierda, y:

\[
P_C=T_{R\rightarrow C}P_R
\]

para la derecha.

Esto es mucho más informativo que una homografía entre imágenes.

Una homografía nos dice aproximadamente cómo deformar una imagen para superponerla sobre otra.

`lens2camera`, en cambio, nos dice cómo están relacionadas las lentes **en el espacio tridimensional**.

---

#### 2.9.1. La pequeña traslación entre las lentes

Las matrices contienen además una traslación.

En una de las calibraciones aparece aproximadamente:

```text
LEFT  x ≈ -0.039 m
RIGHT x ≈ +0.039 m
```

respecto al sistema común.

La separación resultante es de aproximadamente:

\[
0.078\ \text{m}
\]

es decir, unos 7.8 cm entre los centros ópticos según esta representación.

No necesitamos utilizar ese número como una especificación física general de todas las Veo Cam. Lo importante es que la calibración modela explícitamente que ambas lentes no ocupan exactamente el mismo punto.

---

<a id="sec-2-10"></a>

### 2.10. `camera2world`: de la cámara al sistema físico del partido

Una vez tenemos una dirección en el sistema común de cámara necesitamos relacionarla con el mundo.

Ésa es la función de:

```text
camera2world
```

Conceptualmente:

\[
P_W=T_{C\rightarrow W}P_C
\]

La parte de rotación nos indica cómo están orientados los ejes de cámara respecto al mundo.

La parte de traslación nos indica dónde se encuentra el origen de cámara dentro de ese mundo.

Si:

\[
T_{C\rightarrow W}
=
\begin{bmatrix}
R&t\\
0&1
\end{bmatrix}
\]

entonces la posición del origen de cámara en coordenadas de mundo es:

\[
C_W=
T_{C\rightarrow W}
\begin{bmatrix}
0\\0\\0\\1
\end{bmatrix}
=
\begin{bmatrix}
t_x\\t_y\\t_z\\1
\end{bmatrix}
\]

Esto es exactamente lo que necesitamos para construir un rayo:

\[
P(t)=C_W+t\,d_W
\]

donde \(d_W\) es la dirección transformada al mundo.

---

#### 2.10.1. No interpretar los signos sin fijar la convención

Al mirar directamente una matriz puede resultar tentador decir:

```text
camera2world[1,3] = -4.76
→ la cámara está a -4.76 metros
```

pero esto solo tiene sentido después de fijar correctamente la convención de ejes utilizada.

En nuestra geometría validada, el terreno se modela mediante:

\[
Y=0
\]

y utilizamos la convención efectiva del renderer para calcular la intersección del rayo.

Por eso no interpretamos los signos de forma aislada: los interpretamos dentro de toda la transformación.

Esta precaución evita errores habituales al mezclar:

- matrices row-major;
- matrices column-major;
- convenciones OpenGL;
- sistemas con el eje vertical positivo en sentidos diferentes.

---

<a id="sec-2-11"></a>

### 2.11. `camera2presentation`: orientación de la vista panorámica

La panorámica no se genera mirando simplemente en la dirección frontal de `camera2world`.

Veo introduce un sistema adicional que podemos llamar **presentation space**.

Su transformación está relacionada con:

```text
camera2presentation
```

Este sistema define cómo se orienta la representación panorámica respecto a la cámara física.

Es útil pensar en tres capas:

```text
LENTE
  ↓
CÁMARA FÍSICA
  ↓
PRESENTACIÓN
  ↓
PÍXEL PANORÁMICO
```

La panorámica utiliza ángulos de longitud y latitud dentro del sistema de presentación.

---

#### 2.11.1. Un detalle crítico: almacenamiento de matriz frente a uniform real de WebGL

Aquí encontramos uno de los problemas más importantes del reverse engineering.

El shader capturado contiene conceptualmente:

```glsl
camera2world * camera2presentation * ...
```

Sin embargo, interpretar directamente la cadena del `.veo` como una matriz NumPy row-major y multiplicarla de esa manera **no reproducía correctamente la panorámica**.

Al comparar con los uniforms reales suministrados a WebGL y validar contra la imagen de Veo encontramos la convención efectiva que debíamos utilizar en nuestra implementación:

\[
T_{\text{presentation}\rightarrow\text{world}}
=
T_{\text{camera}\rightarrow\text{world}}
\,
T_{\text{camera}\rightarrow\text{presentation}}^{-1}
\]

en código:

```python
camera_to_world_presentation = (
    camera2world
    @ np.linalg.inv(camera2presentation)
)
```

Esto no significa necesariamente que el algoritmo interno de Veo "haga un inverse" en ese punto.

La diferencia está relacionada con cómo se serializa la matriz, cómo la interpretamos en Python y cómo WebGL recibe/lee sus matrices.

La afirmación que sí está validada es:

> Ésta es la convención matricial efectiva que reproduce el resultado real del renderer de Veo cuando partimos de los valores serializados del `.veo`.

Esta distinción es importante porque evita confundir una cuestión de representación matricial con una operación física distinta.

---

<a id="sec-2-12"></a>

### 2.12. `camera2logic`: sistema utilizado por la cámara de Seguimiento

El `.veo` incluye además:

```text
camera2logic
```

Esta matriz aparece en la reconstrucción del vídeo Standard/Follow.

Para calibraciones:

```text
version >= 6.2
```

la lógica observada interpreta el valor almacenado como:

\[
T_{C\rightarrow L}=camera2logic
\]

y construye:

\[
T_{L\rightarrow C}
=
T_{C\rightarrow L}^{-1}
\]

que llamamos:

```text
logic2camera
```

Después el estado dinámico de Tracking utiliza yaw y pitch para construir:

\[
T_{\text{view}\rightarrow\text{world}}
=
T_{\text{camera}\rightarrow\text{world}}
\,
T_{\text{logic}\rightarrow\text{camera}}
\,
R_y(\text{yaw})
\,
R_x(\text{pitch})
\]

Ésta será una pieza central cuando expliquemos en detalle cómo un píxel del vídeo de Seguimiento se convierte en un rayo.

Por ahora lo importante es entender su papel:

> `camera2logic` conecta el sistema físico/calibrado de la cámara con el sistema de referencia que utiliza Veo para orientar su cámara virtual de seguimiento.

---

<a id="sec-2-13"></a>

### 2.13. Dimensiones físicas del terreno

El fichero de calibración contiene:

```text
field_length
field_width
```

Por ejemplo, en una de las grabaciones:

\[
L=105\ \text{m}
\]

\[
W\approx64.706\ \text{m}
\]

y en otra:

\[
L=105\ \text{m}
\]

\[
W\approx72.470\ \text{m}
\]

Esto es muy valioso porque el sistema de mundo no queda expresado únicamente en unidades arbitrarias.

Podemos comprobar si una intersección está dentro del campo mediante:

\[
|X|\leq \frac{L}{2}
\]

\[
|Z|\leq \frac{W}{2}
\]

La geometría actual utiliza precisamente esta convención para clasificar puntos como:

```text
inside_field
outside_field
```

Por ejemplo, para un campo de 105 m:

\[
-\frac{105}{2}
\leq X \leq
\frac{105}{2}
\]

es decir:

\[
-52.5\leq X\leq52.5
\]

El ancho se trata de la misma manera.

---

<a id="sec-2-14"></a>

### 2.14. `va_camera`, `va_logic` y `va_presentation`

La calibración contiene grupos como:

```text
va_camera
va_logic
va_presentation
```

con cuatro valores:

```text
left
right
bottom
top
```

Los valores observados tienen magnitudes propias de ángulos expresados en radianes.

En el shader panorámico, los límites de `va_presentation` se utilizan directamente para convertir las coordenadas normalizadas de la panorámica en:

- longitude;
- latitude.

Si el píxel panorámico normalizado es:

\[
(u,v)\in[0,1]^2
\]

entonces:

\[
\lambda
=
u(va_{right}-va_{left})
+
va_{left}
\]

\[
\phi
=
v(va_{top}-va_{bottom})
+
va_{bottom}
\]

donde:

- \(\lambda\) es la longitud angular;
- \(\phi\) es la latitud angular.

Por tanto `va_presentation` describe, de manera práctica, **qué ventana angular de la esfera virtual aparece en la panorámica**.

No son límites de píxel.

Son límites angulares.

---

<a id="sec-2-15"></a>

### 2.15. `selected_corners`: anclas observadas durante la calibración

El `.veo` contiene además una sección:

```text
selected_corners
```

con varios identificadores de esquina y coordenadas normalizadas dentro de la textura apilada.

También aparece un tiempo de frame, por ejemplo:

```text
2401.3989 s
```

en una grabación y:

```text
1081.8807 s
```

en otra.

Esto indica que el fichero conserva evidencia de puntos seleccionados sobre un frame concreto durante el proceso de calibración.

La estructura permite saber:

- qué lente observa el punto;
- si la observación es válida;
- dónde aparece en la textura apilada.

Si una coordenada normalizada es:

\[
(s_x,s_y)
\]

en una textura:

\[
W\times H
\]

entonces:

\[
u=s_xW
\]

\[
v=s_yH
\]

y, si pertenece a la lente derecha, debemos restar la altura de la mitad superior para obtener su coordenada local.

Estos puntos han sido útiles como evidencia y para comprobaciones, pero nuestra geometría actual **no depende de volver a realizar manualmente esta selección**.

Tampoco atribuimos un significado no demostrado a cada abreviatura de identificador. Conservamos los nombres que aporta Veo sin inventar semántica adicional.

---

<a id="sec-2-16"></a>

### 2.16. `worldmap_version` y métricas adicionales de calibración

También aparecen campos como:

```text
worldmap_version
corner_selection_squareness_error_radians
```

Sabemos que forman parte del recurso de calibración.

El propio nombre de `corner_selection_squareness_error_radians` sugiere una medida relacionada con la consistencia angular de la selección de esquinas, pero no utilizamos este campo para proyectar puntos y no necesitamos asumir cómo lo calcula internamente Veo.

Ésta es una regla que mantendremos durante toda la documentación:

> Que un nombre sea intuitivo no significa que conozcamos el algoritmo interno que lo produce.

Cuando un campo es necesario para nuestro cálculo y conocemos su comportamiento, lo documentamos matemáticamente.

Cuando solamente lo hemos observado, lo describimos como evidencia sin inventar un algoritmo.

---

<a id="sec-2-17"></a>

### 2.17. `camera_directions.det`

El vídeo de Seguimiento necesita información dinámica que no está contenida únicamente en `camera_alignment.veo`.

Esa información está asociada mediante el campo:

```text
camera_directions
```

y el recurso que hemos observado utiliza el nombre:

```text
detections_constrained.det
```

A pesar de su extensión, el fichero que utilizamos puede interpretarse como filas de cuatro valores numéricos.

En uno de los partidos estudiados contiene aproximadamente:

```text
202 118 filas × 4 valores
```

La relación entre:

- número de filas;
- duración del vídeo;
- FPS;

es consistente con aproximadamente una fila por frame.

---

#### 2.17.1. Correspondencia entre tiempo y fila

La lógica reconstruida utiliza:

\[
FPS=\frac{30000}{1001}
\]

y:

\[
n=\operatorname{round}(t\cdot FPS)
\]

para seleccionar:

\[
i=\max(0,n-1)
\]

Es decir:

\[
\boxed{
i=
\max
\left(
0,
\operatorname{round}
\left(
t\frac{30000}{1001}
\right)-1
\right)
}
\]

donde:

- \(t\) es el tiempo del vídeo;
- \(i\) es la fila del `.det`.

Esta relación es importante porque conecta el tiempo común del partido con el estado concreto de la cámara virtual.

---

#### 2.17.2. Significado de las columnas utilizadas

La reconstrucción actual interpreta los cuatro valores como:

```text
yaw
pitch
fov
extra
```

Los tres primeros son suficientes para construir el viewport dinámico de Tracking.

#### 2.17.3. Yaw

Rotación horizontal de la cámara virtual.

Podemos representarla con:

\[
R_y(\theta)=
\begin{bmatrix}
\cos\theta&0&\sin\theta&0\\
0&1&0&0\\
-\sin\theta&0&\cos\theta&0\\
0&0&0&1
\end{bmatrix}
\]

#### 2.17.4. Pitch

Rotación vertical:

\[
R_x(\phi)=
\begin{bmatrix}
1&0&0&0\\
0&\cos\phi&-\sin\phi&0\\
0&\sin\phi&\cos\phi&0\\
0&0&0&1
\end{bmatrix}
\]

#### 2.17.5. FOV

Campo de visión vertical de la cámara virtual.

Si el FOV vertical es \(\alpha\), la altura normalizada del plano de proyección puede escribirse como:

\[
h=2\tan\left(\frac{\alpha}{2}\right)
\]

y para una relación de aspecto \(a\):

\[
\alpha_h
=
2\arctan
\left(
\tan\left(\frac{\alpha}{2}\right)a
\right)
\]

por lo que:

\[
w=
2\tan
\left(
\frac{\alpha_h}{2}
\right)
\]

Estos valores permiten convertir un píxel del vídeo 16:9 en una dirección dentro de la cámara virtual.

La derivación completa se explicará en la parte dedicada a Seguimiento.

---

<a id="sec-2-18"></a>

### 2.18. Los shaders: una fuente especialmente valiosa de evidencia

Uno de los avances más importantes fue obtener el shader utilizado para la panorámica.

Un shader de fragmentos describe directamente qué cálculo se realiza para producir cada píxel de salida.

Esto es mucho más informativo que intentar deducir todo únicamente mirando imágenes finales.

En el shader aparecen explícitamente uniforms como:

```text
camera2world
camera2logic
camera2presentation

lens_left2camera
lens_right2camera

field_length
field_width

intrinsic_left
intrinsic_right

distortion_left
distortion_right

va_left
va_right
va_bottom
va_top
```

Es decir, encontramos en el propio renderer muchas de las piezas que también aparecen en `camera_alignment.veo`.

Esta coincidencia fue esencial para relacionar el fichero de calibración con el comportamiento real del reproductor.

---

<a id="sec-2-19"></a>

### 2.19. Cómo selecciona el shader una dirección panorámica

Para cada píxel de la panorámica, el shader recibe:

\[
(u,v)
\]

y calcula:

\[
lon=u(va_{right}-va_{left})+va_{left}
\]

\[
lat=v(va_{top}-va_{bottom})+va_{bottom}
\]

Después construye una dirección frontal y aplica rotaciones de yaw/longitude y pitch/latitude.

Nuestra forma equivalente y más directa de expresar esa dirección es:

\[
d_P=
\begin{bmatrix}
\sin(lon)\cos(lat)\\
-\sin(lat)\\
\cos(lon)\cos(lat)\\
0
\end{bmatrix}
\]

Esta expresión describe un vector unitario sobre una esfera.

Por eso la panorámica se entiende mejor como una **imagen de direcciones** que como una fotografía plana convencional.

Cada píxel responde esencialmente a:

> ¿En qué dirección del espacio estamos mirando?

Después esa dirección se transforma al mundo y se proyecta de nuevo sobre cada lente para saber qué texel del vídeo Dual Lens debe utilizarse.

---

<a id="sec-2-20"></a>

### 2.20. El remapeo vertical de la textura Dual Lens

Otro detalle que no habríamos podido adivinar fácilmente solo mirando la imagen es que Veo aplica un remapeo vertical antes de muestrear cada lente.

El shader define:

\[
r_{dest}^{top}=\frac{19}{45}
\]

\[
r_{src}^{top}=\frac{19}{32}
\]

Para la zona superior:

\[
v'=
\frac{v}{19/45}
\frac{19}{32}
\]

y para la inferior:

\[
v'
=
\frac{
v-19/45
}{
1-19/45
}
\left(1-\frac{19}{32}\right)
+
\frac{19}{32}
\]

Después:

\[
v_{stacked}=\frac{v'}{2}
\]

para la lente izquierda, y:

\[
v_{stacked}=\frac{v'}{2}+0.5
\]

para la derecha.

Este paso forma parte de la reproducción validada de la panorámica.

Es un buen ejemplo de por qué no bastaba con conocer intrínsecos y distorsión: también teníamos que reproducir cómo Veo empaqueta y muestrea sus imágenes.

---

<a id="sec-2-21"></a>

### 2.21. Cómo mezcla las dos lentes

Una misma dirección puede aparecer en la zona de solape de las dos lentes.

El renderer calcula un color para la izquierda:

\[
C_L
\]

y otro para la derecha:

\[
C_R
\]

y utiliza un factor de mezcla:

\[
b\in[0,1]
\]

para obtener:

\[
C=
(1-b)C_L+bC_R
\]

En el shader observado el factor depende de la posición horizontal de un punto de referencia cercano a la cámara dentro del espacio de presentación y de una distancia de mezcla.

El objetivo práctico es evitar una frontera abrupta entre ambas lentes.

Para nuestra conversión de píxel a campo esta mezcla visual no es el elemento fundamental; lo importante es que la reproducción completa nos permitió validar que habíamos interpretado correctamente:

- matrices;
- óptica;
- direcciones;
- selección de lentes;
- coordenadas de textura.

---

<a id="sec-2-22"></a>

### 2.22. Las cuatro visualizaciones y de qué dependen

Después de relacionar todos estos recursos podemos resumir las cuatro vistas de forma mucho más precisa.

| Vista | Imagen principal | Información adicional necesaria | Naturaleza |
|---|---|---|---|
| Seguimiento de balón | vídeo Standard 1920×1080 | `.veo` + `camera_directions.det` | cámara virtual dinámica |
| Interactivo | Dual Lens | `.veo` + estado yaw/pitch/FOV de la cámara virtual | cámara virtual controlada por usuario |
| Panorámica | Dual Lens | `.veo` + shader/modelo reconstruido | representación angular |
| Doble lente | transcode 2048×2048 | `.veo` para geometría | observación más cercana a las dos lentes |

Lo importante es que no son cuatro sistemas geométricos independientes.

Todas pueden terminar expresando una dirección en el mismo mundo.

---

<a id="sec-2-23"></a>

### 2.23. Qué información cambia entre partidos y qué parece pertenecer al sistema de cámara

Comparar dos calibraciones diferentes resultó especialmente útil.

En los dos ejemplos estudiados observamos:

- misma `calibration_version` 6.2;
- mismos intrínsecos o prácticamente los mismos valores;
- mismos coeficientes de distorsión;
- una transformación derecha muy similar/idéntica;
- pequeñas diferencias en la transformación de la lente izquierda;
- `camera2world` claramente diferente;
- `camera2presentation` diferente;
- anchura del terreno diferente;
- `selected_corners` diferentes;
- timestamp de calibración diferente.

Esto tiene sentido conceptualmente.

Hay parámetros que pueden estar ligados principalmente al hardware/óptica:

```text
intrínsecos
distorsión
relación física básica entre lentes
```

mientras que otros describen la instalación concreta del partido:

```text
camera2world
orientación de presentación
dimensiones del campo
puntos seleccionados
```

No necesitamos asumir qué parámetros recalibra exactamente Veo en cada proceso. La observación importante es que el fichero contiene una combinación de información de cámara y de la escena concreta.

---

<a id="sec-2-24"></a>

### 2.24. Qué podemos afirmar y qué no

A estas alturas podemos hacer afirmaciones bastante fuertes, pero conviene limitar cada una a la evidencia disponible.

#### 2.24.1. Podemos afirmar

- El vídeo panorámico/transcode observado contiene dos lentes apiladas.
- El `.veo` contiene matrices, intrínsecos, distorsión y dimensiones del campo.
- El shader utiliza esos tipos de parámetros para proyectar y mezclar las lentes.
- La fórmula de distorsión está presente explícitamente en el shader.
- Las intrínsecas se expresan respecto a un espacio de referencia 3840×2160.
- `camera_directions` está asociado explícitamente al vídeo Standard correspondiente.
- El `.det` contiene suficientes muestras para reconstruir un estado por frame aproximadamente.
- yaw, pitch y FOV permiten reproducir el modelo de viewport utilizado por Tracking.
- Nuestra interpretación efectiva de las matrices permite reproducir la panorámica de Veo en más de una grabación.

#### 2.24.2. No necesitamos afirmar

- qué algoritmo exacto utiliza Veo para calibrar inicialmente la cámara;
- si internamente utiliza OpenCV para optimizar los coeficientes;
- qué procedimiento exacto produce `selected_corners`;
- qué significa internamente cada métrica de calidad no utilizada;
- cómo genera Veo su lógica automática de seguimiento del balón;
- qué modelo de IA genera un highlight salvo que exista evidencia explícita.

Esta separación mantiene la documentación técnicamente sólida.

---

<a id="sec-2-25"></a>

### 2.25. Resumen de dependencias geométricas

El siguiente mapa resume las dependencias geométricas que utilizamos en el resto del documento.

```text
                            ┌────────────────────────────┐
                            │ camera_alignment.veo       │
                            │                            │
                            │ K izquierda / derecha      │
                            │ distorsión                 │
                            │ lens2camera                │
                            │ camera2world               │
                            │ camera2presentation        │
                            │ camera2logic               │
                            │ field length / width       │
                            │ rangos angulares           │
                            └─────────────┬──────────────┘
                                          │
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
            ▼                             ▼                             ▼
┌────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│ Dual Lens          │       │ Panorama            │       │ Tracking            │
│ 2048×2048          │       │ reconstruida        │       │ 1920×1080           │
│                    │       │                     │       │                     │
│ LEFT + RIGHT       │       │ shader + `.veo`     │       │ `.det` + `.veo`     │
└─────────┬──────────┘       └──────────┬──────────┘       └──────────┬──────────┘
          │                             │                             │
          └─────────────────────────────┴─────────────────────────────┘
                                        │
                                        ▼
                              DIRECCIÓN 3D EN EL MUNDO
                                        │
                                        ▼
                              INTERSECCIÓN CON Y = 0
                                        │
                                        ▼
                                    (X,Z)
                                  EN METROS
```

Con estas dependencias definidas, podemos centrarnos en la operación fundamental que comparten todas las vistas:

> Cómo pasar matemáticamente de un píxel a un rayo tridimensional y de ese rayo a una posición real sobre el terreno de juego.

Ahí entraremos con detalle en:

- modelo pinhole;
- inversión de intrínsecos;
- distorsión y su tratamiento;
- coordenadas homogéneas;
- transformación de puntos y vectores;
- origen de cámara;
- ray casting;
- intersección rayo-plano;
- sistema canónico \((X,Z)\);
- casos degenerados y condiciones de validez.



<a id="capitulo-3"></a>

## 3. Fundamento geométrico: de píxel a mundo

<a id="sec-3-1"></a>

### 3.1. De un píxel a una posición real sobre el terreno

Una vez identificados los datos que proporciona Veo y el papel que cumple cada uno, la pregunta central es:

> ¿Cómo puede un clic sobre una imagen terminar convertido en una posición del campo expresada en metros?

La respuesta se basa en una idea geométrica muy general: **cada píxel de una cámara representa una dirección de observación**.

Si conocemos:

1. dónde está la cámara;
2. hacia qué dirección apunta el rayo correspondiente al píxel;
3. dónde está el plano del terreno;

podemos calcular matemáticamente en qué punto del suelo termina ese rayo.

La cadena conceptual es:

```text
PÍXEL
  ↓
COORDENADA DE IMAGEN NORMALIZADA
  ↓
RAYO EN EL SISTEMA DE LA VISTA/LENTE
  ↓
RAYO EN EL SISTEMA DEL MUNDO
  ↓
INTERSECCIÓN CON EL PLANO DEL CÉSPED
  ↓
(X,Z) EN METROS
```

Éste es el núcleo geométrico que comparten las cuatro visualizaciones actuales.

La diferencia entre Doble lente, Panorámica, Interactivo y Seguimiento no está en la última operación. Todas terminan intersectando un rayo con el mismo terreno.

Lo que cambia es **cómo obtenemos ese rayo**.

---

<a id="sec-3-2"></a>

### 3.2. Sistemas de coordenadas utilizados

Antes de derivar las fórmulas conviene fijar qué significa cada sistema de coordenadas.

#### 3.2.1. Espacio de imagen

Un píxel se expresa como:

\[
(u_p,v_p)
\]

donde normalmente:

- \(u_p\) crece hacia la derecha;
- \(v_p\) crece hacia abajo.

Por ejemplo, en una imagen de \(1920\times1080\):

```text
(0,0) ----------------------> u
  |
  |
  |
  |
  v
```

Ésta es la convención habitual de una imagen almacenada en memoria.

---

#### 3.2.2. Coordenadas normalizadas de textura

Una imagen también puede expresarse mediante coordenadas normalizadas:

\[
(u,v)\in[0,1]^2
\]

En una conversión basada en el centro exacto del píxel utilizamos:

\[
u=\frac{u_p+0.5}{W}
\]

\[
v=\frac{v_p+0.5}{H}
\]

El término \(0.5\) es importante.

Un píxel no es matemáticamente un punto situado en su esquina superior izquierda. Representa una pequeña celda de imagen, y su muestra suele interpretarse en el centro del texel.

Por ejemplo, el primer píxel de una imagen no está en:

\[
u=0
\]

sino aproximadamente en:

\[
u=\frac{0.5}{W}
\]

Esta convención es la que utilizamos en la geometría validada de Panorámica, Doble lente e Interactivo.

---

#### 3.2.3. Espacio normalizado de cámara

En el modelo pinhole una dirección puede representarse sobre un plano situado a profundidad \(z=1\):

\[
d_C=
\begin{bmatrix}
x\\
y\\
1
\end{bmatrix}
\]

Aquí \(x\) e \(y\) ya no son píxeles.

Son coordenadas geométricas normalizadas relativas al eje óptico.

Todos los vectores:

\[
\lambda
\begin{bmatrix}
x\\y\\1
\end{bmatrix}
\qquad \lambda>0
\]

describen el mismo rayo, porque únicamente cambia su longitud, no su dirección.

---

#### 3.2.4. Espacio de lente

Cada lente de Veo tiene su sistema local.

Una dirección de la lente izquierda y otra de la derecha pueden tener las mismas coordenadas locales pero apuntar hacia zonas completamente distintas del campo, porque las lentes están orientadas de manera diferente.

Por eso posteriormente aplicamos:

\[
T_{L\rightarrow C}
\]

o:

\[
T_{R\rightarrow C}
\]

para llevarlas al sistema común de cámara.

---

#### 3.2.5. Espacio de cámara

Es el sistema común al conjunto físico de la cámara.

Nos permite expresar ambas lentes respecto a un único marco de referencia.

---

#### 3.2.6. Espacio de presentación

Es el sistema que interviene en la construcción de la Panorámica e Interactivo.

Nos resulta útil para describir una dirección mediante ángulos:

- longitude;
- latitude.

---

#### 3.2.7. Espacio del mundo

Es el sistema 3D en el que se encuentra el campo.

Representamos un punto mediante:

\[
P_W=
\begin{bmatrix}
X\\
Y\\
Z
\end{bmatrix}
\]

En nuestra geometría:

- \(X\) recorre el eje longitudinal del campo;
- \(Z\) recorre el eje transversal;
- \(Y\) representa altura.

El césped se modela mediante:

\[
\boxed{Y=0}
\]

Por tanto, una posición sobre el terreno puede almacenarse únicamente como:

\[
(X,Z)
\]

y sus unidades son metros.

---

<a id="sec-3-3"></a>

### 3.3. El modelo pinhole como punto de partida

Una cámara ideal puede modelarse mediante:

\[
s
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
=
K
\begin{bmatrix}
x\\
y\\
z
\end{bmatrix}
\]

donde:

- \((x,y,z)\) es un punto en coordenadas de cámara;
- \(K\) es la matriz intrínseca;
- \((u,v)\) es el píxel proyectado;
- \(s\) es una escala.

Para una matriz intrínseca:

\[
K=
\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}
\]

obtenemos:

\[
u=f_x\frac{x}{z}+c_x
\]

\[
v=f_y\frac{y}{z}+c_y
\]

Si queremos realizar la operación inversa y conocemos un píxel ideal:

\[
p=
\begin{bmatrix}
u\\v\\1
\end{bmatrix}
\]

podemos calcular una dirección:

\[
d_C\propto K^{-1}p
\]

Esto no recupera la profundidad del punto.

Un único píxel no nos dice a qué distancia está el objeto.

Lo que recuperamos es un **rayo**.

Ésta es una limitación fundamental de la proyección perspectiva y precisamente la razón por la que necesitamos saber que el punto pertenece al suelo.

---

<a id="sec-3-4"></a>

### 3.4. Del píxel a un rayo, no a un punto 3D

Supongamos que la cámara está situada en:

\[
C=
\begin{bmatrix}
C_x\\C_y\\C_z
\end{bmatrix}
\]

y que el píxel produce una dirección en el mundo:

\[
d=
\begin{bmatrix}
d_x\\d_y\\d_z
\end{bmatrix}
\]

Todos los puntos posibles que podrían haber generado ese píxel se encuentran sobre:

\[
\boxed{
P(t)=C+t\,d
}
\]

con:

\[
t>0
\]

si queremos mirar delante de la cámara.

Expandido:

\[
X(t)=C_x+t\,d_x
\]

\[
Y(t)=C_y+t\,d_y
\]

\[
Z(t)=C_z+t\,d_z
\]

Sin información adicional existen infinitos puntos posibles sobre ese rayo.

La información adicional en nuestro caso es:

> El punto que queremos medir está sobre el césped.

Por tanto sabemos:

\[
Y(t)=0
\]

---

<a id="sec-3-5"></a>

### 3.5. Intersección rayo-plano: el paso que convierte imagen en metros

Sustituimos la ecuación vertical del rayo:

\[
C_y+t\,d_y=0
\]

y despejamos:

\[
\boxed{
t=-\frac{C_y}{d_y}
}
\]

Siempre que:

\[
d_y\neq0
\]

Después:

\[
\boxed{
X=C_x+t\,d_x
}
\]

\[
\boxed{
Z=C_z+t\,d_z
}
\]

Éstas son probablemente las tres ecuaciones más importantes de toda la geometría actual.

Una vez conocemos:

- el origen de cámara \(C\);
- la dirección del rayo \(d\);

podemos transformar una observación de imagen en una posición física del campo.

---

#### 3.5.1. Por qué el resultado está en metros

Las transformaciones de la calibración sitúan la cámara dentro de un sistema de mundo ligado al terreno.

Las dimensiones proporcionadas:

\[
field\_length
\]

y:

\[
field\_width
\]

también están expresadas en metros.

Por tanto, si:

\[
C
\]

está expresado en ese sistema y las transformaciones rígidas conservan su escala, el punto:

\[
C+t\,d
\]

queda expresado en las mismas unidades.

El valor de \(t\) no debe interpretarse siempre como una distancia euclídea por sí solo, porque depende de si \(d\) está normalizado.

Pero las coordenadas finales:

\[
(X,Z)
\]

sí quedan correctamente expresadas en el sistema métrico del mundo.

Si queremos la distancia física real desde la cámara al punto podemos calcular:

\[
D=\|P-C\|
\]

que es independiente de la escala elegida para representar la dirección del rayo.

---

<a id="sec-3-6"></a>

### 3.6. Casos en los que la intersección no es válida

La fórmula es sencilla, pero no todos los píxeles deben generar un punto de campo.

#### 3.6.1. Rayo paralelo al terreno

Si:

\[
|d_y|\approx0
\]

el rayo es prácticamente paralelo al suelo.

Entonces:

\[
t=-\frac{C_y}{d_y}
\]

se vuelve numéricamente inestable o tiende a infinito.

En la implementación se utiliza una tolerancia pequeña, del orden de:

\[
10^{-12}
\]

para evitar realizar esta división.

---

#### 3.6.2. Intersección detrás de la cámara

Si:

\[
t\leq0
\]

la intersección matemática está detrás del origen del rayo.

No representa un punto visible delante de la cámara y se rechaza.

---

#### 3.6.3. Punto fuera del terreno

Una intersección puede ser geométricamente válida pero estar fuera del rectángulo físico del campo.

La comprobación es:

\[
|X|\leq\frac{L}{2}
\]

\[
|Z|\leq\frac{W}{2}
\]

Si no se cumple, el sistema puede clasificarla como:

```text
outside_field
```

en lugar de inventar que pertenece al terreno.

---

#### 3.6.4. Píxeles que no representan el suelo

La geometría rayo-plano no realiza segmentación semántica.

Si seleccionamos:

- el cielo;
- una grada;
- la cabeza de un jugador;
- una valla;

la recta correspondiente puede cortar matemáticamente el plano \(Y=0\), pero ese punto no describe necesariamente la superficie realmente seleccionada.

Por eso nuestra interpretación es:

> La conversión es válida para puntos que conceptualmente pertenecen al plano del terreno.

En un futuro, para un jugador, utilizaremos una estimación de su punto de apoyo sobre el césped, no el centro arbitrario de su bounding box.

---



<a id="capitulo-4"></a>

## 4. Doble lente: inversión óptica y proyección al campo

<a id="sec-4-1"></a>

### 4.1. Doble lente: píxel RAW → rayo → campo

Doble lente es el caso más cercano al modelo óptico físico de la cámara.

El pipeline utilizado es:

```text
píxel de la textura 2048×2048
        ↓
determinar LEFT / RIGHT
        ↓
coordenada local de lente
        ↓
deshacer empaquetado/remapeo vertical
        ↓
píxel en espacio de calibración 3840×2160
        ↓
invertir la distorsión
        ↓
dirección ideal de la lente
        ↓
lens2camera
        ↓
camera2world
        ↓
rayo mundial
        ↓
Y = 0
        ↓
(X,Z)
```

Cada paso merece una derivación.

---

<a id="sec-4-2"></a>

### 4.2. Identificación de la lente

Como vimos anteriormente:

\[
0\leq y<1024
\]

implica lente izquierda.

En caso contrario:

\[
1024\leq y<2048
\]

implica lente derecha.

La coordenada vertical local es:

\[
v_L=\frac{y+0.5}{1024}
\]

para la izquierda, y:

\[
v_R=
\frac{y+0.5-1024}{1024}
\]

para la derecha.

La coordenada horizontal normalizada se obtiene como:

\[
u=\frac{x+0.5}{2048}
\]

---

<a id="sec-4-3"></a>

### 4.3. Invertir el remapeo vertical del transcode

El renderer de Veo no utiliza directamente la coordenada vertical original.

En la parte anterior vimos la transformación directa:

```text
coordenada de calibración
→ compresión vertical
→ textura Dual Lens
```

Para pasar de un clic RAW hacia la cámara necesitamos hacer lo contrario.

Si llamamos \(v_s\) a la coordenada dentro de una lente en la textura y:

\[
r_s=\frac{19}{32}
\]

\[
r_d=\frac{19}{45}
\]

la función inversa implementada es:

#### 4.3.1. Zona superior

Si:

\[
v_s<\frac{19}{32}
\]

entonces:

\[
v_c=
\frac{v_s}{19/32}
\frac{19}{45}
\]

#### 4.3.2. Zona inferior

En caso contrario:

\[
v_c=
\frac{19}{45}
+
\frac{
v_s-19/32
}{
1-19/32
}
\left(
1-\frac{19}{45}
\right)
\]

Así recuperamos la coordenada vertical normalizada del espacio original de calibración.

---

<a id="sec-4-4"></a>

### 4.4. Volver al espacio 3840×2160

Las intrínsecas están construidas para:

\[
W_C=3840
\]

\[
H_C=2160
\]

Por tanto el píxel que queremos invertir es:

\[
u_C=
\frac{x+0.5}{2048}\cdot3840
\]

\[
v_C=
v_c\cdot2160
\]

Éste es el píxel en el mismo sistema en el que actúan:

- \(K\);
- los coeficientes de distorsión.

---

<a id="sec-4-5"></a>

### 4.5. Por qué no basta con multiplicar por \(K^{-1}\)

Si no existiese distorsión podríamos hacer directamente:

\[
\begin{bmatrix}
x\\y
\end{bmatrix}
\approx
K^{-1}
\begin{bmatrix}
u_C\\v_C\\1
\end{bmatrix}
\]

Pero el píxel observado ha sido deformado por el modelo racional.

Tenemos una función conocida:

\[
F(x,y)=
\begin{bmatrix}
u_C\\v_C
\end{bmatrix}
\]

que incluye:

1. distorsión radial;
2. distorsión tangencial;
3. matriz intrínseca.

La operación RAW → mundo necesita resolver:

\[
F(x,y)=p_{obs}
\]

para recuperar:

\[
(x,y)
\]

No existe una inversión algebraica sencilla del modelo racional completo que queramos utilizar directamente.

Por eso resolvemos el problema numéricamente.

---

<a id="sec-4-6"></a>

### 4.6. Inversión numérica de la distorsión

Nuestra implementación utiliza un método de Newton amortiguado.

Queremos encontrar:

\[
q=
\begin{bmatrix}
x\\y
\end{bmatrix}
\]

tal que:

\[
F(q)-p=0
\]

Definimos el residuo:

\[
r(q)=F(q)-p
\]

En Newton, una actualización ideal es:

\[
q_{n+1}
=
q_n-
J(q_n)^{-1}r(q_n)
\]

donde:

\[
J(q)=
\begin{bmatrix}
\frac{\partial F_x}{\partial x} &
\frac{\partial F_x}{\partial y}\\
\frac{\partial F_y}{\partial x} &
\frac{\partial F_y}{\partial y}
\end{bmatrix}
\]

es el Jacobiano.

---

#### 4.6.1. Jacobiano mediante diferencias centrales

En lugar de mantener una derivada simbólica de toda la función racional, calculamos numéricamente:

\[
\frac{\partial F}{\partial x}
\approx
\frac{
F(q+h e_x)-F(q-h e_x)
}{
2h
}
\]

\[
\frac{\partial F}{\partial y}
\approx
\frac{
F(q+h e_y)-F(q-h e_y)
}{
2h
}
\]

con un paso pequeño:

\[
h\approx10^{-5}
\]

Esto produce las dos columnas del Jacobiano.

---

#### 4.6.2. Amortiguación

Un Newton puro puede divergir si la estimación cae en una zona complicada del modelo racional.

Por eso no aceptamos automáticamente:

\[
q-\Delta
\]

sino que probamos:

\[
q-\alpha\Delta
\]

con:

\[
\alpha=
1,\frac12,\frac14,\frac18,\ldots
\]

hasta encontrar un candidato que reduzca el residuo:

\[
\|F(q_{new})-p\|
<
\|F(q)-p\|
\]

Este *line search* hace mucho más estable la inversión.

---

#### 4.6.3. Condición de convergencia

La solución se acepta cuando el error de reproyección es extremadamente pequeño:

\[
\|F(q)-p\|<10^{-7}
\]

en el espacio de calibración utilizado por la implementación.

La idea fundamental es importante:

> No ajustamos una nueva calibración. Estamos invirtiendo numéricamente el mismo modelo de distorsión hacia delante que ya fue validado con el renderer.

Si no converge, preferimos devolver:

```text
no ground point
```

antes que inventar una dirección.

---

<a id="sec-4-7"></a>

### 4.7. Construcción del rayo de una lente

Una vez recuperado:

\[
(x,y)
\]

la dirección ideal local puede expresarse como:

\[
d_L=
\begin{bmatrix}
x\\y\\1\\0
\end{bmatrix}
\]

para cualquiera de las lentes.

La transformación completa hacia el mundo es:

\[
T_{L\rightarrow W}
=
T_{C\rightarrow W}
T_{L\rightarrow C}
\]

o, para la derecha:

\[
T_{R\rightarrow W}
=
T_{C\rightarrow W}
T_{R\rightarrow C}
\]

El origen de la lente es:

\[
C_W=
T_{L\rightarrow W}
\begin{bmatrix}
0\\0\\0\\1
\end{bmatrix}
\]

y su dirección:

\[
d_W=
T_{L\rightarrow W}
\begin{bmatrix}
x\\y\\1\\0
\end{bmatrix}
\]

El valor homogéneo \(w=0\) de la dirección impide que la traslación de la matriz altere el vector.

A continuación aplicamos:

\[
t=-\frac{C_y}{d_y}
\]

y:

\[
P=C+t\,d
\]

para obtener:

\[
(X,Z)
\]

---

<a id="sec-4-8"></a>

### 4.8. Por qué Doble lente no necesita una homografía del campo

Ésta es una diferencia conceptual importante respecto a nuestros primeros experimentos.

No hacemos:

\[
(u,v)
\xrightarrow{H^{-1}}
(X,Z)
\]

sino:

\[
(u,v)
\rightarrow
\text{óptica}
\rightarrow
\text{rayo 3D}
\rightarrow
\text{suelo}
\]

El campo únicamente interviene al final mediante el plano:

\[
Y=0
\]

y sus dimensiones físicas.

Esto nos permite separar:

- formación de imagen;
- orientación de cámara;
- geometría del terreno.

---



<a id="capitulo-5"></a>

## 5. Panorámica e Interactivo

<a id="sec-5-1"></a>

### 5.1. Panorámica: un píxel representa una dirección angular

La Panorámica es más sencilla de invertir porque ya está construida como una representación angular.

Para un píxel:

\[
(x_p,y_p)
\]

de una panorámica:

\[
W\times H
\]

utilizamos el centro:

\[
u=\frac{x_p+0.5}{W}
\]

Como WebGL utiliza una convención vertical opuesta a una imagen convencional:

\[
v=
1-
\frac{y_p+0.5}{H}
\]

Después calculamos:

\[
lon=
u(va_{right}-va_{left})
+
va_{left}
\]

\[
lat=
v(va_{top}-va_{bottom})
+
va_{bottom}
\]

---

<a id="sec-5-2"></a>

### 5.2. Longitude y latitude → dirección 3D

La dirección en el espacio de presentación es:

\[
\boxed{
d_P=
\begin{bmatrix}
\sin(lon)\cos(lat)\\
-\sin(lat)\\
\cos(lon)\cos(lat)\\
0
\end{bmatrix}
}
\]

Podemos comprobar que tiene norma uno:

\[
\|d_P\|^2
=
\sin^2(lon)\cos^2(lat)
+
\sin^2(lat)
+
\cos^2(lon)\cos^2(lat)
\]

Agrupando:

\[
=
\cos^2(lat)
[
\sin^2(lon)+\cos^2(lon)
]
+
\sin^2(lat)
\]

\[
=
\cos^2(lat)+\sin^2(lat)=1
\]

Por tanto describe exactamente una dirección sobre la esfera unidad.

---

<a id="sec-5-3"></a>

### 5.3. Espacio de presentación → mundo

La convención efectiva que validamos en Python es:

\[
T_{P\rightarrow W}
=
T_{C\rightarrow W}
T_{C\rightarrow P}^{-1}
\]

Entonces:

\[
d_W=
T_{P\rightarrow W}d_P
\]

La posición de cámara se obtiene de:

\[
C_W=
T_{C\rightarrow W}
\begin{bmatrix}
0\\0\\0\\1
\end{bmatrix}
\]

y volvemos exactamente a la misma ecuación:

\[
P(t)=C_W+t\,d_W
\]

\[
t=-\frac{C_y}{d_y}
\]

\[
(X,Z)=
(C_x+t\,d_x,\,
 C_z+t\,d_z)
\]

Por eso el código que convierte un píxel panorámico a campo es relativamente pequeño: toda la complejidad óptica ya fue utilizada al **crear** la panorámica.

---

<a id="sec-5-4"></a>

### 5.4. Interactivo: una cámara virtual dentro del mismo mundo

La vista Interactiva puede entenderse como una cámara perspectiva virtual que mira dentro de la representación panorámica.

Su estado se describe mediante:

```text
longitude
latitude
fov
```

Además conocemos:

- anchura de la ventana;
- altura de la ventana;
- relación de aspecto.

El pipeline es:

```text
píxel Interactive
        ↓
rayo perspectiva de viewport
        ↓
rotación longitude / latitude
        ↓
presentation → world
        ↓
Y = 0
        ↓
(X,Z)
```

---

<a id="sec-5-5"></a>

### 5.5. Píxel Interactivo → coordenadas del viewport

Para una vista:

\[
W\times H
\]

calculamos:

\[
u=\frac{x+0.5}{W}
\]

\[
v=\frac{y+0.5}{H}
\]

Si el FOV vertical es:

\[
\alpha
\]

definimos:

\[
q=
\tan\left(\frac{\alpha}{2}\right)
\]

El aspecto es:

\[
a=\frac{W}{H}
\]

y el rayo local utilizado puede escribirse:

\[
d_V=
\begin{bmatrix}
(2u-1)\,q\,a\\
(2v-1)\,q\\
1\\
0
\end{bmatrix}
\]

Obsérvese que el centro:

\[
u=v=0.5
\]

produce:

\[
d_V=
\begin{bmatrix}
0\\0\\1\\0
\end{bmatrix}
\]

es decir, el eje frontal de la cámara virtual.

---

<a id="sec-5-6"></a>

### 5.6. FOV y por qué aparece la tangente

El FOV vertical total es \(\alpha\).

Si imaginamos el plano de proyección a profundidad 1, la mitad de su altura \(q\) forma un triángulo rectángulo:

```text
             borde superior
                  *
                 /|
                / |
               /  | q
              /   |
             / α/2|
 cámara  *---------*
              1
```

Por trigonometría:

\[
\tan\left(\frac{\alpha}{2}\right)
=
\frac{q}{1}
\]

por tanto:

\[
q=
\tan\left(\frac{\alpha}{2}\right)
\]

Ésta es la razón matemática por la que el FOV aparece mediante una tangente al construir el rayo.

---

<a id="sec-5-7"></a>

### 5.7. Rotaciones de la cámara virtual

La orientación de Interactivo se construye con rotaciones de longitude y latitude.

Podemos expresarlas como:

\[
R_y(\theta)=
\begin{bmatrix}
\cos\theta&0&\sin\theta&0\\
0&1&0&0\\
-\sin\theta&0&\cos\theta&0\\
0&0&0&1
\end{bmatrix}
\]

y:

\[
R_x(\phi)=
\begin{bmatrix}
1&0&0&0\\
0&\cos\phi&-\sin\phi&0\\
0&\sin\phi&\cos\phi&0\\
0&0&0&1
\end{bmatrix}
\]

La transformación efectiva es conceptualmente:

\[
T_{V\rightarrow W}
=
T_{P\rightarrow W}
R_y(longitude)
R_x(latitude)
\]

y:

\[
d_W=T_{V\rightarrow W}d_V
\]

Después repetimos:

\[
t=-\frac{C_y}{d_y}
\]

y obtenemos:

\[
(X,Z)
\]

---

<a id="sec-5-8"></a>

### 5.8. Por qué mover Interactivo no rompe la geometría

Cuando el usuario arrastra la vista no estamos alterando el campo ni recalibrando la cámara.

Únicamente cambiamos:

\[
longitude
\]

y:

\[
latitude
\]

Cuando hace zoom cambiamos:

\[
FOV
\]

La calibración física permanece exactamente igual.

Ésta es una ventaja directa de haber separado:

```text
geometría física
```

de:

```text
cámara virtual de visualización
```

---



<a id="capitulo-6"></a>

## 6. Seguimiento de balón y cámara dinámica

<a id="sec-6-1"></a>

### 6.1. Seguimiento de balón: el caso dinámico

Seguimiento es conceptualmente parecido a Interactivo, pero la orientación no la controla el usuario.

Veo proporciona una secuencia temporal de estados de cámara mediante `.det`.

Para cada tiempo \(t\):

```text
t
 ↓
fila .det
 ↓
yaw(t)
pitch(t)
fov(t)
 ↓
cámara virtual
 ↓
píxel
 ↓
rayo
 ↓
(X,Z)
```

---

<a id="sec-6-2"></a>

### 6.2. Tiempo → estado de cámara

La frecuencia utilizada es:

\[
f=
\frac{30000}{1001}
\]

Para un tiempo:

\[
t
\]

calculamos:

\[
n=
\operatorname{round}(tf)
\]

y:

\[
i=
\max(0,n-1)
\]

La fila \(i\) proporciona:

\[
yaw_i,\ pitch_i,\ fov_i
\]

Este paso convierte el tiempo del vídeo en una configuración geométrica.

Por tanto, un píxel de Tracking no puede interpretarse sin conocer además:

\[
t
\]

El mismo píxel:

\[
(960,540)
\]

puede mirar hacia zonas diferentes del campo en dos instantes distintos.

---

<a id="sec-6-3"></a>

### 6.3. Estado de cámara Tracking → mundo

Para calibraciones 6.2 construimos primero:

\[
T_{L\rightarrow C}
=
camera2logic^{-1}
\]

y después:

\[
\boxed{
T_{V\rightarrow W}
=
camera2world
\cdot
logic2camera
\cdot
R_y(yaw)
\cdot
R_x(pitch)
}
\]

Esta matriz expresa una dirección del viewport Tracking dentro del mundo.

Aquí cada factor tiene una función concreta:

#### 6.3.1. \(R_x(pitch)\)

Inclina verticalmente la cámara virtual.

#### 6.3.2. \(R_y(yaw)\)

La gira horizontalmente.

#### 6.3.3. `logic2camera`

Convierte desde el marco utilizado por la lógica de seguimiento al marco físico de la cámara.

#### 6.3.4. `camera2world`

Sitúa finalmente esa dirección en el sistema físico del terreno.

---

<a id="sec-6-4"></a>

### 6.4. FOV de Tracking → tamaño del viewport

El `.det` proporciona FOV vertical:

\[
\alpha_v
\]

Para relación de aspecto:

\[
a=\frac{16}{9}
\]

el FOV horizontal correspondiente es:

\[
\boxed{
\alpha_h
=
2\arctan
\left(
\tan\left(\frac{\alpha_v}{2}\right)a
\right)
}
\]

Definimos:

\[
h_v=
2\tan
\left(
\frac{\alpha_v}{2}
\right)
\]

\[
w_v=
2\tan
\left(
\frac{\alpha_h}{2}
\right)
\]

Matemáticamente:

\[
w_v
=
2a
\tan
\left(
\frac{\alpha_v}{2}
\right)
\]

pero mantener la formulación mediante FOV horizontal deja más clara la geometría.

---

<a id="sec-6-5"></a>

### 6.5. Píxel Tracking → dirección local

Para Tracking la convención reconstruida utiliza coordenadas normalizadas desde la esquina superior izquierda:

\[
u=\frac{x}{W}
\]

\[
v=\frac{y}{H}
\]

La dirección del viewport es:

\[
d_V=
\begin{bmatrix}
(u-0.5)w_v\\
(v-0.5)h_v\\
1\\
0
\end{bmatrix}
\]

y la dirección de mundo:

\[
d_W=
T_{V\rightarrow W}d_V
\]

El origen utilizado es la posición de la cámara:

\[
C_W=
\begin{bmatrix}
camera2world_{0,3}\\
camera2world_{1,3}\\
camera2world_{2,3}
\end{bmatrix}
\]

Finalmente:

\[
t=-\frac{C_y}{d_y}
\]

\[
P=C+t\,d
\]

y:

\[
(X,Z)=(P_x,P_z)
\]

---



<a id="capitulo-7"></a>

## 7. Sistema canónico, campo 2D y validación métrica

<a id="sec-7-1"></a>

### 7.1. Cuatro caminos diferentes, una misma operación final

Ya podemos colocar las cuatro vistas lado a lado.

#### 7.1.1. Doble lente

\[
pixel
\rightarrow
undo\ texture\ mapping
\rightarrow
undo\ distortion
\rightarrow
lens2camera
\rightarrow
camera2world
\rightarrow
ray
\]

#### 7.1.2. Panorámica

\[
pixel
\rightarrow
(lon,lat)
\rightarrow
spherical\ direction
\rightarrow
presentation2world
\rightarrow
ray
\]

#### 7.1.3. Interactivo

\[
pixel
\rightarrow
perspective\ viewport
\rightarrow
yaw/pitch/FOV
\rightarrow
presentation2world
\rightarrow
ray
\]

#### 7.1.4. Seguimiento

\[
(pixel,t)
\rightarrow
.det(t)
\rightarrow
yaw/pitch/FOV
\rightarrow
logic2camera
\rightarrow
camera2world
\rightarrow
ray
\]

Y las cuatro terminan en:

\[
\boxed{
ray
\cap
(Y=0)
\longrightarrow
(X,Z)
}
\]

Ésta es la razón por la que podemos representar puntos procedentes de visualizaciones muy distintas sobre el mismo campo 2D.

---

<a id="sec-7-2"></a>

### 7.2. El sistema canónico del terreno

Una vez calculado:

\[
(X,Z)
\]

dejamos de necesitar información sobre cómo se obtuvo.

El campo puede definirse mediante:

\[
-\frac{L}{2}
\leq
X
\leq
\frac{L}{2}
\]

\[
-\frac{W}{2}
\leq
Z
\leq
\frac{W}{2}
\]

El centro del campo es:

\[
(0,0)
\]

Los puntos de penalti están, por ejemplo, en:

\[
\left(
-\frac{L}{2}+11,\ 0
\right)
\]

y:

\[
\left(
\frac{L}{2}-11,\ 0
\right)
\]

Las esquinas son:

\[
\left(
\pm\frac{L}{2},
\pm\frac{W}{2}
\right)
\]

Este sistema es la interfaz común entre:

```text
VISIÓN
```

y:

```text
ANÁLISIS DE FÚTBOL
```

---

<a id="sec-7-3"></a>

### 7.3. Cómo se dibuja `(X,Z)` sobre el campo 2D

Nuestra visualización del campo utiliza directamente las dimensiones físicas.

Esto evita introducir una segunda transformación arbitraria.

Si el `viewBox` SVG se expresa en unidades del campo, podemos utilizar:

\[
x_{SVG}=X
\]

El eje vertical de SVG crece hacia abajo, mientras que para el terreno usamos \(Z\) con orientación matemática.

Por eso visualmente aparece una inversión:

\[
y_{SVG}=-Z
\]

Conceptualmente:

```text
MUNDO                       SVG

       +Z
        ↑
        |                       y ↓
-X -----+----- +X          -X -----+----- +X
        |
        ↓
       -Z
```

No estamos convirtiendo metros a una escala inventada y luego intentando recuperarlos.

El propio espacio del dibujo puede utilizar metros como unidades lógicas y el navegador se encarga de escalarlo visualmente al tamaño disponible.

---

<a id="sec-7-4"></a>

### 7.4. Por qué el campo 2D es una validación útil

Cuando hacemos clic sobre un punto conocido del vídeo y aparece en la posición esperada del campo estamos validando simultáneamente una cadena larga:

```text
coordenada del clic
        ↓
convención de píxel
        ↓
modelo de la vista
        ↓
matrices
        ↓
orientación
        ↓
rayo
        ↓
intersección
        ↓
sistema del campo
```

Un error en cualquiera de esas partes puede desplazar el resultado.

Por eso el campo 2D no es únicamente una visualización estética.

Es una herramienta de diagnóstico geométrico.

---

<a id="sec-7-5"></a>

### 7.5. Validación mediante landmarks conocidos

Para cuantificar el error utilizamos posiciones físicas conocidas del terreno.

Sea:

\[
P_e=
(X_e,Z_e)
\]

la posición esperada de un landmark.

Después de hacer clic obtenemos:

\[
P_m=
(X_m,Z_m)
\]

La diferencia por eje es:

\[
\Delta X=X_m-X_e
\]

\[
\Delta Z=Z_m-Z_e
\]

y el error euclídeo:

\[
\boxed{
e=
\sqrt{
(\Delta X)^2+
(\Delta Z)^2
}
}
\]

Como \(X\) y \(Z\) están en metros:

\[
e
\]

también está directamente en metros.

Ésta es una ventaja enorme respecto a validar únicamente en píxeles.

---

<a id="sec-7-6"></a>

### 7.6. Landmarks utilizados

A partir de las dimensiones del campo podemos construir referencias exactas como:

```text
centro del campo
puntos de penalti
cuatro esquinas
esquinas de las áreas
esquinas de las áreas pequeñas
```

Por ejemplo, si:

\[
L=105
\]

el extremo izquierdo está en:

\[
X=-52.5
\]

y el punto de penalti izquierdo está 11 metros hacia dentro:

\[
X=-52.5+11=-41.5
\]

con:

\[
Z=0
\]

No son coordenadas ajustadas a mano para que coincidan con la imagen.

Se derivan de las dimensiones métricas del terreno.

---

<a id="sec-7-7"></a>

### 7.7. Estadísticos de validación

Con varios puntos podemos resumir el error.

#### 7.7.1. Error medio

\[
\bar e=
\frac1N
\sum_{i=1}^{N}e_i
\]

Nos da una medida intuitiva del error típico, aunque puede verse afectada por outliers.

---

#### 7.7.2. Mediana

Ordenamos los errores y tomamos el central.

Es especialmente útil cuando uno de los clics fue claramente incorrecto, porque es más robusta frente a valores extremos.

---

#### 7.7.3. RMS

\[
\boxed{
RMS=
\sqrt{
\frac1N
\sum_{i=1}^{N}
e_i^2
}
}
\]

Penaliza más los errores grandes.

Por eso es una métrica útil para detectar una geometría que normalmente funciona pero ocasionalmente se desvía mucho.

---

#### 7.7.4. Error máximo

\[
e_{max}=\max_i(e_i)
\]

Permite conocer el peor caso observado.

---

<a id="sec-7-8"></a>

### 7.8. Qué significa una precisión de unos pocos metros

En nuestras validaciones iniciales, ignorando un clic claramente erróneo, aparecieron errores del orden aproximado de:

```text
0.5 – 3 m
```

con resultados agregados que rondaban:

```text
media ≈ 1.88 m
RMS   ≈ 2.11 m
```

Estos valores no deben interpretarse como una certificación universal de precisión.

Dependen de:

- partido;
- zona del campo;
- precisión del clic;
- vista utilizada;
- calibración.

Pero sí demuestran algo importante:

> La cadena geométrica produce posiciones coherentes en la escala física correcta sin calibrar manualmente cada frame.

Una precisión de este orden puede ser útil para:

- heatmaps;
- ocupación por zonas;
- estructuras tácticas;
- posición aproximada de jugadores;
- análisis espacial agregado.

No sería suficiente, sin una mejora adicional, para aplicaciones que exigiesen precisión centimétrica.

---

<a id="sec-7-9"></a>

### 7.9. Diferencia entre error de geometría y error de selección

Cuando validamos con un clic, el error total observado combina varias fuentes.

Podemos pensar conceptualmente:

\[
e_{observado}
=
e_{modelo}
+
e_{selección}
+
e_{representación}
+\ldots
\]

No es una suma escalar estricta, pero ayuda a entender el problema.

Por ejemplo:

#### 7.9.1. Error de modelo

La calibración o nuestra reconstrucción no representa exactamente la cámara.

#### 7.9.2. Error de clic

El usuario selecciona un punto varios píxeles desplazado de la línea real.

#### 7.9.3. Resolución

Un píxel puede representar una región física significativa en zonas lejanas.

#### 7.9.4. Ambigüedad visual

Una esquina pintada puede estar parcialmente oculta o tener varios píxeles de grosor.

Por eso un error de dos metros no implica automáticamente que la calibración esté equivocada dos metros.

La sección posterior de validación estudiará estas fuentes con más detalle.

---

<a id="sec-7-10"></a>

### 7.10. Por qué un pequeño error angular puede convertirse en metros

Este fenómeno es importante para entender la precisión.

Si una cámara está elevada a altura:

\[
h
\]

y un rayo mira hacia el suelo con un ángulo pequeño respecto al horizonte, la intersección puede estar muy lejos.

En una simplificación 2D:

\[
D\approx\frac{h}{\tan\theta}
\]

donde:

- \(D\) es distancia horizontal;
- \(\theta\) es el ángulo descendente.

La sensibilidad es:

\[
\frac{dD}{d\theta}
=
-\frac{h}{\sin^2\theta}
\]

Cuando:

\[
\theta\rightarrow0
\]

la derivada crece mucho.

Esto significa:

> Cerca del horizonte, un error angular muy pequeño puede desplazar muchos metros la intersección con el terreno.

Por eso las zonas lejanas son intrínsecamente más sensibles que las cercanas.

No es una particularidad de Veo.

Es una consecuencia de la geometría proyectiva.

---

<a id="sec-7-11"></a>

### 7.11. Por qué el punto de apoyo de un jugador será importante

En el futuro un detector producirá una caja:

\[
(x,y,w,h)
\]

Una aproximación habitual al contacto del jugador con el terreno es el centro inferior:

\[
u_f=x+\frac{w}{2}
\]

\[
v_f=y+h
\]

Ese punto puede enviarse a nuestra función:

\[
pixel\rightarrow(X,Z)
\]

Pero esto es solamente una aproximación.

La caja puede:

- incluir sombra;
- cortar los pies;
- estar inclinada;
- contener una persona saltando;
- desplazarse por una oclusión.

Por eso el futuro sistema deberá validar si bottom-center es suficiente o si conviene utilizar:

- pose estimation;
- segmentación;
- detección de pies;
- corrección temporal del track.

Lo importante es que la interfaz ya está definida:

> Cualquier método futuro que estime un punto de contacto en píxeles puede utilizar la misma geometría para obtener metros.

---

<a id="sec-7-12"></a>

### 7.12. La escala espacial y la escala temporal quedan unificadas

Con esta geometría obtenemos:

\[
P_i=(X_i,Z_i)
\]

y del vídeo conocemos:

\[
t_i
\]

Por tanto una observación futura puede expresarse como:

\[
O_i=
(t_i,X_i,Z_i)
\]

Con una identidad temporal:

\[
track\_id
\]

obtenemos una trayectoria:

\[
\mathcal{T}
=
\{
(t_0,X_0,Z_0),
(t_1,X_1,Z_1),
\dots
\}
\]

A partir de ella ya podemos definir cantidades físicas.

---

#### 7.12.1. Distancia recorrida

\[
D=
\sum_{i=0}^{N-2}
\sqrt{
(X_{i+1}-X_i)^2+
(Z_{i+1}-Z_i)^2
}
\]

---

#### 7.12.2. Velocidad

\[
v_i=
\frac{
\sqrt{
(X_{i+1}-X_i)^2+
(Z_{i+1}-Z_i)^2
}
}{
t_{i+1}-t_i
}
\]

con unidades:

\[
m/s
\]

y posteriormente:

\[
km/h=3.6\,m/s
\]

---

#### 7.12.3. Heatmap

Un heatmap puede estimar la densidad espacial de:

\[
(X_i,Z_i)
\]

por ejemplo mediante bins o un kernel:

\[
\rho(X,Z)
=
\sum_i
K_h
\left(
X-X_i,\,
Z-Z_i
\right)
\]

La calidad de estas métricas dependerá posteriormente de:

- detección;
- tracking;
- precisión geométrica;
- filtrado temporal.

Pero la geometría actual es la capa que proporciona unidades físicas comunes.

---

<a id="sec-7-13"></a>

### 7.13. Qué parte está validada actualmente

La dirección que consideramos útil y validada para experimentación es:

```text
Doble lente
pixel → (X,Z)       ✓

Panorámica
pixel → (X,Z)       ✓

Interactivo
pixel → (X,Z)       ✓

Seguimiento
pixel + tiempo
→ (X,Z)             ✓ baseline útil
```

Estos resultados pueden representarse juntos en el campo 2D.

---

<a id="sec-7-14"></a>

### 7.14. Una asimetría pendiente: mundo → algunas vistas

La existencia de una función:

\[
pixel\rightarrow(X,Z)
\]

no garantiza automáticamente que nuestra implementación inversa:

\[
(X,Z)\rightarrow pixel
\]

esté igualmente validada en todas las visualizaciones.

En pruebas de Seguimiento e Interactivo observamos que algunos marcadores reproyectados podían desplazarse claramente fuera de la posición esperada.

Por ejemplo:

```text
punto correcto en campo 2D
        ↓
reproyección sobre vídeo
        ↓
marcador demasiado alto,
fuera del césped o incluso en el cielo
```

Por ese motivo decidimos separar ambos problemas.

Actualmente confiamos en:

```text
vista → mundo
```

para selección y validación.

No utilizamos todavía como funcionalidad principal:

```text
mundo → Tracking
mundo → Interactivo
```

hasta entender completamente la discrepancia.

---

<a id="sec-7-15"></a>

### 7.15. Por qué no corregimos ese problema con un offset fijo

Una solución aparentemente sencilla sería añadir:

```text
+20 px verticales
-15 px horizontales
```

o modificar artificialmente:

```text
yaw
pitch
FOV
```

hasta hacer coincidir una captura.

De hecho, la infraestructura experimental permite representar biases:

```text
yaw_bias
pitch_bias
fov_bias
screen_offset_x
screen_offset_y
```

Pero observamos que una corrección buena en un instante no necesariamente era buena en otro.

Esto sugiere que el residuo no se explica simplemente mediante una constante global.

Por eso la configuración base mantiene:

\[
bias=0
\]

en todos estos parámetros.

La prioridad ha sido conservar un modelo interpretable antes que ocultar errores mediante números mágicos.

---

<a id="sec-7-16"></a>

### 7.16. Posible corrección residual futura

En experimentos separados comprobamos que podía registrarse visualmente una imagen generada con una observada utilizando:

```text
features
→ matching
→ RANSAC
→ homografía residual
```

El modelo futuro podría tener una forma:

\[
p_{real}
\sim
H_{residual}(t)
\,p_{geom}
\]

donde:

- \(p_{geom}\) es la predicción del modelo físico;
- \(H_{residual}(t)\) corrige únicamente un residuo visual.

Este enfoque puede ser interesante, pero no forma parte de la base geométrica actual.

La razón es metodológica:

> Primero queremos saber exactamente qué explica la cámara y qué error queda. Después podremos modelar ese residuo de manera explícita.

---

<a id="sec-7-17"></a>

### 7.17. Resumen matemático de la conversión

Todo el sistema de selección puede condensarse en dos etapas.

#### 7.17.1. Etapa A: construir el rayo

Depende de la vista:

\[
(u,v,\text{estado})
\longrightarrow
(C,d)
\]

donde:

- \(C\) es el origen de la cámara;
- \(d\) es la dirección en coordenadas de mundo.

#### 7.17.2. Etapa B: intersectar con el terreno

Es común a todas:

\[
P(t)=C+t\,d
\]

\[
P_y(t)=0
\]

\[
t=-\frac{C_y}{d_y}
\]

\[
X=C_x+t\,d_x
\]

\[
Z=C_z+t\,d_z
\]

Por tanto:

\[
\boxed{
(u,v,\text{estado de cámara})
\longrightarrow
(X,Z)
}
\]

Ésta es la interfaz geométrica fundamental sobre la que se apoya todo lo construido hasta ahora.

---

<a id="sec-7-18"></a>

### 7.18. Por qué podemos confiar razonablemente en esta construcción

No confiamos en ella únicamente porque las fórmulas sean matemáticamente correctas.

Una fórmula correcta aplicada con matrices mal interpretadas produciría un resultado incorrecto.

La confianza procede de una cadena de validaciones complementarias:

1. **Los parámetros utilizados proceden de la calibración asociada al partido.**
2. **La fórmula de distorsión está observada directamente en el shader.**
3. **La reconstrucción de la panorámica reproduce visualmente el resultado de Veo.**
4. **La reproducción funcionó con más de una calibración y partido.**
5. **La inversión RAW utiliza exactamente el modelo hacia delante validado.**
6. **Los puntos obtenidos pueden compararse contra landmarks métricos conocidos.**
7. **Las distintas vistas convergen sobre un único sistema de coordenadas del campo.**

Ninguna de estas pruebas por sí sola sería suficiente.

Juntas proporcionan una base mucho más sólida que una calibración ajustada únicamente para que "parezca correcta" en una captura.

---

<a id="sec-7-19"></a>

### 7.19. Resultado de esta etapa

Llegados a este punto ya no tratamos una imagen de Veo como una cuadrícula de píxeles sin significado físico.

Para puntos pertenecientes al terreno podemos construir:

```text
PÍXEL
  ↓
RAYO 3D
  ↓
INTERSECCIÓN CON EL CAMPO
  ↓
(X,Z)
```

y `(X,Z)` está expresado en el mismo sistema métrico en todas las vistas.

Ésta es precisamente la base que necesitábamos antes de introducir detección y tracking.

La validación experimental comprueba **cómo sabemos que toda esta reconstrucción es correcta**, incluyendo:

- reproducción de la panorámica;
- comparación entre partidos;
- landmarks;
- errores métricos;
- resultados obtenidos;
- limitaciones;
- experimentos de Tracking;
- soluciones descartadas como offsets manuales;
- qué consideramos validado y qué permanece pendiente.



<a id="capitulo-8"></a>

## 8. Evidencia experimental, limitaciones y estado actual

<a id="sec-8-1"></a>

### 8.1. Validación experimental: cómo comprobamos que la reconstrucción funciona

Una reconstrucción geométrica no queda validada únicamente porque sus fórmulas sean correctas.

Podemos tener un modelo matemáticamente impecable y, aun así, obtener resultados incorrectos por:

- interpretar una matriz en el sentido equivocado;
- mezclar row-major y column-major;
- utilizar un sistema de ejes diferente;
- olvidar un remapeo de textura;
- asociar una calibración al vídeo incorrecto;
- interpretar mal el FOV;
- desplazar un frame respecto a sus metadatos temporales.

Por eso la validación del proyecto se realizó en varias capas independientes.

La idea general fue:

```text
1. ¿Podemos reproducir visualmente lo que Veo genera?
2. ¿Podemos hacerlo con más de un partido?
3. ¿La implementación CPU coincide con el renderer WebGL?
4. ¿Podemos invertir el modelo óptico de las lentes?
5. ¿Los rayos terminan en posiciones métricas razonables?
6. ¿Coinciden con puntos físicos conocidos del campo?
7. ¿La cámara dinámica de Tracking mantiene coherencia temporal?
8. ¿Las correcciones propuestas generalizan o solo arreglan una captura?
```

Este enfoque es importante porque evita validar todo el sistema con una única prueba.

---

<a id="sec-8-2"></a>

### 8.2. Primera validación fuerte: reproducir la panorámica de Veo

El punto de inflexión del proyecto fue conseguir reproducir la panorámica que muestra Veo a partir de:

```text
Dual Lens 2048×2048
+
camera_alignment.veo
+
shader capturado
```

Si nuestra interpretación de las lentes, intrínsecos, distorsión y matrices fuese incorrecta, la panorámica resultante no podría coincidir con la original de forma consistente.

El pipeline reconstruido fue, de forma resumida:

```text
píxel de salida panorámica
        ↓
longitude / latitude
        ↓
dirección 3D de presentación
        ↓
mundo
        ↓
proyección hacia LEFT
        ↓
proyección hacia RIGHT
        ↓
distorsión de cada lente
        ↓
muestreo del Dual Lens
        ↓
blend
```

La validación visual permitió detectar varios errores de interpretación antes de considerar estable el renderer.

---

<a id="sec-8-3"></a>

### 8.3. El caso `camera2presentation`: un error de convención que la validación detectó

Uno de los problemas más importantes apareció con:

```text
camera2presentation
```

Una lectura directa del nombre y del shader parecía sugerir una multiplicación concreta.

Sin embargo, al interpretar la matriz serializada del `.veo` directamente en NumPy, la panorámica no coincidía correctamente con la observada.

Esto podía deberse a varias causas:

- disposición de memoria;
- convención row-major / column-major;
- forma en que WebGL recibe la matriz;
- sentido de la transformación almacenada.

En lugar de decidir cuál era la correcta por intuición, utilizamos el resultado observable como prueba.

La convención efectiva que terminó reproduciendo el renderer fue:

\[
T_{P\rightarrow W}
=
T_{C\rightarrow W}
\,
T_{C\rightarrow P}^{-1}
\]

es decir:

```python
camera_to_world_presentation = (
    camera2world
    @ inverse(camera2presentation)
)
```

Después de aplicar esta interpretación, la panorámica encajó con Veo.

Este resultado es importante por dos motivos.

Primero, demuestra por qué no basta con leer nombres de matrices.

Segundo, muestra nuestra metodología:

> Si una convención matricial puede interpretarse de varias formas, la validamos reproduciendo un resultado externo conocido.

---

<a id="sec-8-4"></a>

### 8.4. Validación con dos partidos diferentes

Una reconstrucción puede coincidir accidentalmente con un partido si contiene ajustes implícitos específicos de esa grabación.

Por eso repetimos el proceso con una segunda grabación y otro fichero:

```text
camera_alignment.veo
```

distinto.

Los dos partidos presentan diferencias claras en:

- `camera2world`;
- `camera2presentation`;
- anchura del campo;
- puntos de calibración;
- orientación de la cámara respecto al terreno.

A pesar de ello, el mismo algoritmo reprodujo correctamente la panorámica utilizando únicamente los datos correspondientes a cada partido.

Conceptualmente:

```text
PARTIDO A
Dual Lens A + Calibration A
        ↓
mismo renderer
        ↓
Panorama A correcta


PARTIDO B
Dual Lens B + Calibration B
        ↓
mismo renderer
        ↓
Panorama B correcta
```

No se introdujo una homografía distinta ajustada manualmente para cada panorama.

La parte variable llega de los metadatos de calibración.

Esto es una evidencia especialmente importante de generalización.

---

<a id="sec-8-5"></a>

### 8.5. Comparación CPU frente al renderer WebGL

Además de observar las imágenes visualmente, comparamos nuestra reproducción CPU con el renderer WebGL construido a partir del shader capturado.

En una validación posterior del proyecto se registraron errores RGB medios aproximados de:

```text
Partido 1: 0.12847
Partido 2: 0.09205
```

sobre una escala de canal:

\[
0\ldots255
\]

y percentiles 99 del error de aproximadamente:

```text
Partido 1: 1
Partido 2: 2
```

Estos números necesitan contexto.

Un error medio de:

\[
0.12847
\]

en una escala de 255 representa una diferencia media por canal extremadamente pequeña:

\[
\frac{0.12847}{255}
\approx
0.000504
\]

es decir, alrededor de un:

\[
0.0504\%
\]

de la escala completa.

No esperamos igualdad bit a bit entre todas las implementaciones debido a diferencias como:

- interpolación de textura;
- precisión floating point;
- GPU frente a CPU;
- redondeos;
- codificación/decodificación.

Pero estos resultados indican que ambas implementaciones están realizando esencialmente la misma transformación visual.

---

<a id="sec-8-6"></a>

### 8.6. Qué valida realmente la coincidencia de panorama

Una buena coincidencia panorámica no demuestra todavía que cada posición del campo esté perfectamente calibrada en metros.

Sí valida, sin embargo, una gran parte de la cadena anterior:

```text
va_presentation                 ✓
dirección esférica              ✓
camera2presentation efectivo    ✓
camera2world                    ✓
lens_left2camera                ✓
lens_right2camera               ✓
intrínsecos                     ✓
distorsión                      ✓
3840×2160 de calibración        ✓
remapeo vertical                ✓
LEFT / RIGHT                    ✓
blend                           ✓
muestreo de la textura          ✓
```

Por eso conseguir reproducir la panorámica fue mucho más que una mejora estética.

Fue una validación indirecta de la interpretación del modelo de cámara.

---

<a id="sec-8-7"></a>

### 8.7. Validación de la geometría RAW mediante inversión del mismo modelo

Después de validar el camino hacia delante:

```text
dirección 3D
→ lente
→ distorsión
→ píxel
```

construimos su operación inversa para Doble lente:

```text
píxel
→ invertir distorsión
→ dirección de lente
→ mundo
```

Aquí evitamos introducir un segundo modelo independiente.

La función hacia delante utilizada para comprobar la solución numérica es exactamente la misma formulación de distorsión que utiliza el renderer.

Si:

\[
q=(x,y)
\]

es nuestra solución ideal y:

\[
F(q)
\]

es la función de distorsión/proyección hacia píxel, exigimos que:

\[
F(q)\approx p_{observado}
\]

La inversión se acepta únicamente cuando el residuo de reproyección cae por debajo de un umbral muy pequeño.

Esto produce un criterio de consistencia interna:

\[
\boxed{
pixel
\xrightarrow{inverse}
ray
\xrightarrow{forward}
pixel'
}
\]

con:

\[
pixel'\approx pixel
\]

---

<a id="sec-8-8"></a>

### 8.8. Por qué este round-trip es útil

Supongamos que la inversión devuelve una dirección aparentemente razonable, pero en realidad se ha desviado hacia otra rama de la función racional.

Visualmente podríamos no detectarlo inmediatamente.

Al volver a aplicar el modelo hacia delante comprobaríamos:

\[
\|p'-p\|
\]

y el error sería grande.

Por eso nuestra inversión no acepta simplemente "la última iteración de Newton".

Exige que la solución vuelva a producir el píxel original.

Este tipo de comprobación es especialmente importante en modelos no lineales.

---

<a id="sec-8-9"></a>

### 8.9. Validación en las dos lentes

La inversión RAW fue concebida para funcionar tanto con:

```text
LEFT
```

como con:

```text
RIGHT
```

y utilizando la transformación específica de cada lente.

Esto es importante porque ambas lentes no son intercambiables.

Una implementación que accidentalmente tratase la mitad inferior como si fuese LEFT podría producir coordenadas numéricas válidas, pero físicamente absurdas.

La división explícita:

\[
y<1024
\Rightarrow LEFT
\]

\[
y\geq1024
\Rightarrow RIGHT
\]

y la utilización de:

\[
lens\_left2camera
\]

o:

\[
lens\_right2camera
\]

permiten comprobar cada mitad bajo su calibración correspondiente.

---

<a id="sec-8-10"></a>

### 8.10. Validación métrica mediante landmarks del terreno

La reproducción visual demuestra que entendemos cómo Veo forma sus imágenes.

Para demostrar que podemos obtener **metros**, necesitamos otra clase de prueba.

Utilizamos landmarks cuya posición puede derivarse directamente de las dimensiones reglamentarias del campo.

Por ejemplo:

#### 8.10.1. Centro

\[
P_e=(0,0)
\]

#### 8.10.2. Esquina

\[
P_e=
\left(
-\frac{L}{2},
-\frac{W}{2}
\right)
\]

#### 8.10.3. Punto de penalti izquierdo

\[
P_e=
\left(
-\frac{L}{2}+11,
0
\right)
\]

#### 8.10.4. Esquina del área

\[
P_e=
\left(
-\frac{L}{2}+16.5,
\pm20.16
\right)
\]

Estas posiciones no se ajustan a la imagen.

Se calculan a partir de:

\[
L=field\_length
\]

y:

\[
W=field\_width
\]

del propio partido.

---

<a id="sec-8-11"></a>

### 8.11. Cómo calculamos el error de un landmark

Si la posición conocida es:

\[
P_e=(X_e,Z_e)
\]

y nuestro clic produce:

\[
P_m=(X_m,Z_m)
\]

calculamos:

\[
\Delta X=X_m-X_e
\]

\[
\Delta Z=Z_m-Z_e
\]

y:

\[
\boxed{
e=
\sqrt{
(X_m-X_e)^2+
(Z_m-Z_e)^2
}
}
\]

Como las coordenadas están en metros:

\[
e
\]

también está en metros.

Esto nos permite responder cuantitativamente a:

> ¿Cuánto se ha desplazado nuestra predicción respecto a la posición física esperada?

---

<a id="sec-8-12"></a>

### 8.12. Resultados métricos observados

En una de las sesiones de validación manual realizadas durante el desarrollo, y descartando un clic claramente incorrecto, los errores observados quedaron aproximadamente en el rango:

```text
0.5 m – 3 m
```

con valores agregados próximos a:

\[
\bar e\approx1.88\ m
\]

y:

\[
RMS\approx2.11\ m
\]

Estos valores son resultados experimentales de esa sesión concreta, no una especificación garantizada de Veo ni un error universal del sistema.

No afirmamos:

```text
"el sistema siempre tiene 1.88 m de error"
```

sino:

> En una validación manual concreta, varios landmarks distribuidos por el campo produjeron errores del orden de pocos metros.

Para el objetivo actual, esto sirve como evidencia de que la escala, orientación y posición general del campo son coherentes.

---

<a id="sec-8-13"></a>

### 8.13. Por qué excluimos un clic claramente erróneo

Excluir un punto solo es válido si no se utiliza para ocultar un fallo del modelo.

En este caso distinguimos entre:

```text
error geométrico
```

y:

```text
error humano evidente al seleccionar el landmark
```

Una buena práctica es conservar ambos análisis:

- estadísticas con todos los puntos;
- estadísticas robustas o inspección sin un punto que sabemos seleccionado incorrectamente.

La mediana resulta especialmente útil porque reduce el impacto de un outlier sin necesidad de borrarlo matemáticamente.

La documentación final debería conservar las capturas o valores originales cuando queramos realizar una validación formal posterior.

---

<a id="sec-8-14"></a>

### 8.14. No ajustar la calibración con los mismos puntos usados para validarla

Otro principio importante es evitar una validación circular.

Si utilizásemos un landmark para modificar la cámara hasta hacerlo coincidir y después utilizásemos el mismo landmark como prueba, su error tendría poco valor.

En la geometría actual los landmarks son:

```text
VALIDACIÓN
```

no:

```text
CALIBRACIÓN OCULTA
```

La selección de un landmark calcula:

\[
expected
\]

frente a:

\[
measured
\]

pero no modifica:

- intrínsecos;
- distorsión;
- `camera2world`;
- `lens2camera`;
- homografía;
- FOV.

Esta separación es fundamental para que el error medido tenga significado.

---

<a id="sec-8-15"></a>

### 8.15. Evolución de la validación de Tracking

Seguimiento fue la vista más difícil de reconstruir.

La historia de esta parte es especialmente útil porque muestra cómo cambiamos una conclusión cuando apareció nueva evidencia.

#### 8.15.1. Fase 1: calibración estática insuficiente

Inicialmente teníamos:

- `.veo`;
- vídeo Tracking;
- renderer de Panorámica.

Pero no teníamos una trayectoria temporal de la cámara virtual.

Por eso la conclusión correcta en ese momento fue:

> No se puede convertir de forma fiable un píxel Tracking a mundo utilizando únicamente la calibración estática.

No se añadió:

- homografía fija;
- FOV inventado;
- offset manual.

La funcionalidad métrica se mantuvo desactivada.

---

<a id="sec-8-16"></a>

### 8.16. Descubrimiento posterior de `camera_directions.det`

Más adelante encontramos que el objeto de metadatos correspondiente al vídeo Standard contenía:

```text
camera_directions
```

apuntando a:

```text
detections_constrained.det
```

Esto resolvía precisamente la información que faltaba:

```text
tiempo
→ yaw
→ pitch
→ FOV
```

El descubrimiento cambió la situación.

La documentación histórica que indicaba que Tracking no podía proyectarse era correcta para la evidencia disponible en aquel momento, pero quedó superada por el descubrimiento posterior del `.det`.

Éste es un buen ejemplo de por qué conviene documentar la evolución y no únicamente el estado final.

---

<a id="sec-8-17"></a>

### 8.17. Validación temporal del `.det`

Una comprobación sencilla pero importante fue comparar:

- número de filas;
- FPS;
- duración del vídeo.

En una de las grabaciones encontramos aproximadamente:

```text
202 118 filas
```

para un vídeo de alrededor de:

```text
6744 s
```

Si calculamos:

\[
\frac{202118}{6744}
\approx29.97
\]

obtenemos prácticamente:

\[
\frac{30000}{1001}
\approx29.97003
\]

Esta coincidencia apoya la interpretación:

> El fichero contiene aproximadamente una muestra de cámara por frame del vídeo Tracking.

Además, la lógica observada de selección de fila utiliza precisamente esa frecuencia.

---

<a id="sec-8-18"></a>

### 8.18. Validación funcional Tracking → campo

Una vez reconstruida la cámara temporal hicimos la prueba importante:

```text
clic sobre una posición reconocible
en Seguimiento
        ↓
pixel + timestamp
        ↓
.det
        ↓
rayo
        ↓
(X,Z)
        ↓
campo 2D
```

Los puntos obtenidos aparecieron en posiciones coherentes sobre el campo 2D.

Esto fue suficiente para habilitar Tracking como fuente experimental de:

\[
pixel\rightarrow(X,Z)
\]

sin introducir tuning manual por defecto.

---

<a id="sec-8-19"></a>

### 8.19. El problema de la proyección inversa en Tracking

Cuando intentamos la operación contraria:

\[
(X,Z)
\rightarrow
pixel_{Tracking}
\]

aparecieron errores visuales importantes.

Un punto que en el campo 2D parecía correcto podía proyectarse:

- demasiado arriba;
- fuera del césped;
- incluso sobre el cielo.

Esto creó aparentemente una contradicción:

```text
pixel → world funciona razonablemente
world → pixel no coincide visualmente
```

En un modelo matemático ideal perfectamente consistente ambos caminos deberían ser inversos.

Que no lo sean en nuestra implementación indica que todavía existe alguna discrepancia en una de las convenciones o etapas del camino inverso.

Posibles familias de causas incluyen:

- convención vertical;
- interpretación exacta de FOV;
- cropping/resampling del vídeo Standard;
- transformación lógica;
- sincronización;
- estabilización o transformación adicional del vídeo renderizado.

No seleccionamos una causa sin evidencia.

---

<a id="sec-8-20"></a>

### 8.20. Decisión: desactivar el overlay antes que ocultar el error

Podíamos haber dejado el marcador sobre el vídeo porque "más o menos se veía".

No lo hicimos.

La decisión fue:

```text
Seguimiento:
clic → campo 2D        SÍ
campo 2D → overlay     NO por ahora
```

Esto mantiene separadas:

- una dirección que hemos validado de forma útil;
- una dirección que todavía presenta discrepancias.

La misma cautela se aplicó a Interactivo cuando su forward projection visual no quedaba suficientemente alineada.

---

<a id="sec-8-21"></a>

### 8.21. Experimento con biases manuales

Para investigar la discrepancia construimos una capa opcional:

```text
yaw_bias_rad
pitch_bias_rad
fov_bias_rad
screen_offset_x_px
screen_offset_y_px
```

Conceptualmente podríamos hacer:

\[
yaw'=yaw+\Delta yaw
\]

\[
pitch'=pitch+\Delta pitch
\]

\[
fov'=fov+\Delta fov
\]

y después:

\[
u'=u+\Delta u
\]

\[
v'=v+\Delta v
\]

Si el problema fuese un offset constante, una corrección global debería funcionar durante todo el vídeo.

Pero no ocurrió así.

Una combinación que mejoraba un timestamp no necesariamente mejoraba otro.

Por tanto, no incorporamos esos valores como calibración real.

El baseline conserva:

\[
\Delta yaw=0
\]

\[
\Delta pitch=0
\]

\[
\Delta fov=0
\]

\[
\Delta u=0
\]

\[
\Delta v=0
\]

---

<a id="sec-8-22"></a>

### 8.22. Por qué rechazar un bias fijo es importante

Un número mágico puede mejorar una captura y empeorar el modelo general.

Supongamos que encontramos:

\[
\Delta pitch=0.02\ rad
\]

que hace coincidir un punto.

Eso no demuestra que la cámara tuviese realmente ese error.

Podría estar compensando indirectamente:

- un crop vertical;
- un desfase temporal;
- un FOV mal interpretado;
- una transformación adicional.

Si lo incorporásemos como corrección física, estaríamos mezclando causas diferentes.

La decisión fue:

> No convertir un ajuste visual local en una afirmación geométrica global sin evidencia.

---

<a id="sec-8-23"></a>

### 8.23. Registro visual residual como experimento

También exploramos una estrategia diferente.

Si podemos generar una vista aproximada mediante el modelo geométrico y compararla con el frame Tracking real, podemos estimar una transformación residual:

```text
imagen geométrica
        ↓
ORB / correspondencias
        ↓
RANSAC
        ↓
H_residual
        ↓
imagen real
```

En pruebas exploratorias obtuvimos miles de correspondencias útiles y residuos del orden de subpíxel en algunos registros, con valores cercanos a:

```text
~0.8 px de error mediano
```

en un experimento con aproximadamente:

```text
~2600 correspondencias
```

Este resultado demuestra que una corrección visual residual puede ser viable.

Sin embargo, decidimos no convertirla todavía en parte del modelo principal.

---

<a id="sec-8-24"></a>

### 8.24. Por qué la homografía residual quedó aparcada

Una transformación residual puede ser muy útil si se utiliza de forma explícita:

\[
p_{real}
=
H_{residual}(t)
p_{modelo}
\]

Pero queremos evitar el siguiente escenario:

```text
modelo físico ligeramente incorrecto
        ↓
homografía residual lo tapa
        ↓
parece funcionar
        ↓
ya no sabemos qué parte era correcta
```

Por eso primero conservamos el modelo físico sin tuning.

La homografía residual queda como una capa futura claramente separada:

```text
MODELO BASE
+
CORRECCIÓN RESIDUAL OBSERVADA
```

y no como sustituto silencioso de la geometría.

---

<a id="sec-8-25"></a>

### 8.25. Validación de Interactivo

Interactivo comparte gran parte de la geometría de Panorámica.

La diferencia es que utiliza una cámara perspectiva virtual definida por:

- longitude;
- latitude;
- FOV.

La validación principal consistió en comprobar que un clic sobre el césped podía:

```text
pixel Interactive
→ rayo
→ Y=0
→ (X,Z)
→ campo 2D
```

de forma coherente.

Como en Tracking, evitamos convertir la reproyección inversa sobre la imagen en una funcionalidad considerada validada mientras existiesen discrepancias visuales.

---

<a id="sec-8-26"></a>

### 8.26. Consistencia entre visualizaciones

Una prueba conceptual muy importante es seleccionar la misma zona física desde vistas diferentes.

Idealmente:

```text
Doble lente
      \
Panorámica ----→ mismo (X,Z)
      /
Interactivo

Seguimiento ----→ aproximadamente el mismo (X,Z)
```

No necesitamos que los píxeles coincidan entre vistas.

De hecho, no deberían coincidir.

Lo que debe coincidir es la posición física final.

Ésta es la razón de haber elegido:

\[
(X,Z)
\]

como interfaz canónica.

---

<a id="sec-8-27"></a>

### 8.27. Qué significa "funciona" en este proyecto

La palabra puede resultar ambigua.

No significa:

```text
cada píxel produce una posición perfecta
con error cero
```

Tampoco significa:

```text
hemos replicado todos los algoritmos internos de Veo
```

Lo que sí podemos sostener actualmente es:

1. Podemos reproducir el renderer panorámico observado utilizando los datos de calibración.
2. La reproducción generaliza al menos a dos grabaciones/calibraciones estudiadas.
3. Podemos invertir el modelo de lente para obtener rayos desde el RAW.
4. Podemos construir rayos desde Panorámica.
5. Podemos construir rayos desde una cámara Interactiva.
6. Podemos reconstruir el estado temporal de Tracking utilizando `.det`.
7. Podemos intersectar esos rayos con el plano físico del terreno.
8. El resultado está expresado en el sistema métrico del campo.
9. Los landmarks producen errores del orden esperado para una baseline útil.
10. Conservamos separadas las operaciones que todavía no están suficientemente validadas.

---

<a id="sec-8-28"></a>

### 8.28. Tabla de estado actual

| Componente | Estado | Evidencia principal |
|---|---|---|
| Dual Lens 2048×2048 | Validado como fuente | estructura observada y utilizada por renderer |
| Parámetros `.veo` | Validados como entrada | shader + reproducción |
| Modelo de distorsión | Validado | fórmula explícita en shader |
| `lens2camera` | Validado en renderer | panorama reproducida |
| `camera2world` | Validado en renderer/ground | panorama + landmarks |
| `camera2presentation` efectivo | Validado experimentalmente | coincidencia con Veo en dos partidos |
| Remapeo vertical | Validado | necesario para coincidencia visual |
| Blend LEFT/RIGHT | Validado visualmente | reproducción panorámica |
| Panorámica → `(X,Z)` | Validado para uso actual | ray casting + landmarks |
| Dual Lens → `(X,Z)` | Validado para uso actual | inversión RAW + campo |
| Interactivo → `(X,Z)` | Útil/validado para selección | cámara virtual + campo |
| Tracking `.det` | Asociado y utilizado | `/videos/`, FPS y filas |
| Tracking → `(X,Z)` | Baseline útil | clic → campo coherente |
| `(X,Z)` → Tracking | Pendiente | overlays incorrectos |
| `(X,Z)` → Interactivo | Pendiente de validación completa | discrepancias visuales |
| Biases manuales | Rechazados como solución | no generalizan temporalmente |
| Homografía residual | Prometedora, aparcada | registro visual experimental |

---

<a id="sec-8-29"></a>

### 8.29. Limitaciones actuales

Una documentación técnica útil no debería esconder las limitaciones.

#### 8.29.1. El terreno se modela como plano

Utilizamos:

\[
Y=0
\]

Esto supone que el césped es plano.

Para un estadio convencional es una aproximación razonable para análisis táctico, pero no modela:

- desniveles;
- bombeo del césped;
- objetos sobre el terreno.

---

#### 8.29.2. Un clic sobre un objeto elevado no representa su posición 3D real

Si hacemos clic sobre la cabeza de un jugador, el rayo pasa por la cabeza.

Nuestra intersección con:

\[
Y=0
\]

continúa hasta el césped.

El resultado representa dónde corta esa línea al suelo, no la posición tridimensional de la cabeza.

Por eso para personas necesitaremos estimar el contacto con el terreno.

---

#### 8.29.3. Sensibilidad en zonas lejanas

Cerca del horizonte:

\[
d_y\rightarrow0
\]

y:

\[
t=-\frac{C_y}{d_y}
\]

crece mucho.

Un pequeño error angular puede convertirse en varios metros.

Esto limita la precisión física en las regiones más lejanas.

---

#### 8.29.4. Precisión de la calibración original

Nuestra reconstrucción puede ser fiel a la calibración de Veo y aun así heredar el error de esa calibración.

Reproducir exactamente el renderer demuestra:

```text
hemos entendido el modelo
```

pero no necesariamente:

```text
la calibración física es perfecta
```

Por eso seguimos necesitando landmarks en metros.

---

#### 8.29.5. Tracking puede incluir transformaciones que aún no hemos modelado totalmente

El `.det` aporta la pieza temporal principal, pero las discrepancias en la proyección inversa sugieren que puede existir algún detalle adicional en la generación final del vídeo Standard.

Hasta entenderlo no utilizamos esa dirección como verdad absoluta.

---

<a id="sec-8-30"></a>

### 8.30. Niveles de confianza

Para evitar tratar todos los resultados por igual podemos clasificar la evidencia.

#### 8.30.1. Alta confianza

Elementos directamente observados y reproducidos:

```text
estructura Dual Lens
shader panorámico
intrínsecos
distorsión
lens2camera
remapeo de textura
renderer panorámico
```

#### 8.30.2. Confianza alta con validación experimental

```text
convención efectiva camera2presentation
Panorama → mundo
RAW → mundo
```

porque producen resultados reproducibles en más de una grabación.

#### 8.30.3. Confianza útil pero todavía susceptible de refinamiento

```text
Tracking pixel → mundo
Interactive pixel → mundo
```

especialmente en precisión fina.

#### 8.30.4. Pendiente

```text
mundo → Tracking
mundo → Interactive
corrección residual temporal final
```

Esta clasificación no es una puntuación matemática formal.

Sirve para dejar claro qué partes consideramos base y cuáles siguen siendo investigación.

---

<a id="sec-8-31"></a>

### 8.31. Por qué no necesitamos perfección antes de empezar visión artificial

Para construir un sistema útil de análisis deportivo no necesitamos que la primera versión localice cada jugador con precisión centimétrica.

Necesitamos que:

1. el sistema de coordenadas sea estable;
2. diferentes vistas hablen el mismo idioma espacial;
3. los errores sean medibles;
4. las capas posteriores no tengan que reinventar la geometría.

Una detección futura podrá producir:

\[
pixel_{foot}
\]

y nuestra capa geométrica devolver:

\[
(X,Z)
\]

Si posteriormente mejoramos la calibración de 2 m a 0.5 m, no será necesario reescribir el detector.

Ésta es otra ventaja de separar:

```text
VISIÓN
```

de:

```text
GEOMETRÍA
```

---

<a id="sec-8-32"></a>

### 8.32. Qué queda fundamentado después de esta validación

La cadena que consideramos suficientemente fundamentada para continuar es:

```text
DATOS DE VEO
    ↓
MODELO DE CÁMARA
    ↓
RECONSTRUCCIÓN DE VISTAS
    ↓
PÍXEL
    ↓
RAYO 3D
    ↓
PLANO DEL TERRENO
    ↓
(X,Z) EN METROS
```

Esta cadena está respaldada por tres tipos de evidencia diferentes:

#### 8.32.1. Evidencia directa

Datos y shader observados en Veo.

#### 8.32.2. Evidencia matemática

Modelos de cámara, transformaciones, distorsión e intersección rayo-plano.

#### 8.32.3. Evidencia experimental

Reproducción visual, comparación CPU/WebGL, segundo partido, inversión RAW y landmarks.

Esta combinación es precisamente lo que permite utilizar la calibración de Veo sin trabajar "a ciegas".

No confiamos en una caja negra.

Conocemos qué operación matemática estamos realizando y disponemos de pruebas externas que comprueban su comportamiento.

---

<a id="sec-8-33"></a>

### 8.33. Siguiente consecuencia técnica

Una vez validada esta base, el siguiente problema ya no es:

> ¿Cómo convertimos un píxel a metros?

sino:

> ¿Qué píxel corresponde de manera estable a cada jugador o al balón en cada instante?

Eso pertenece a la siguiente capa:

```text
DETECCIÓN
    ↓
TRACKING
    ↓
PUNTO DE APOYO
    ↓
GEOMETRÍA YA VALIDADA
    ↓
(X,Z,t)
```

La documentación puede cerrar a partir de aquí la base geométrica y dedicar una última parte a:

- cómo se conectará con detección y tracking;
- qué información adicional de Veo hemos descubierto;
- qué elementos quedan deliberadamente fuera;
- estructura conceptual final;
- reproducibilidad;
- conclusiones.



<a id="capitulo-9"></a>

## 9. Arquitectura futura y análisis automático

<a id="sec-9-1"></a>

### 9.1. Arquitectura conceptual final

Después de recorrer los enfoques descartados, reconstruir el modelo de cámara y validar la conversión de píxeles a coordenadas métricas, podemos resumir la arquitectura técnica del proyecto en una idea muy simple:

> Todas las observaciones visuales deben terminar expresadas en un sistema común de espacio y tiempo.

Ese sistema común está formado por:

\[
(X,Z)
\]

para la posición sobre el terreno, y:

\[
t
\]

para el instante del partido.

Por tanto, la unidad conceptual mínima sobre la que se apoyará el análisis futuro es:

\[
\boxed{
O_i=(t_i,X_i,Z_i)
}
\]

Cuando además conozcamos la identidad temporal de una entidad:

\[
track\_id
\]

podremos construir trayectorias completas.

La arquitectura final puede representarse así:

```text
                    DATOS DE VEO
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
      Dual Lens      Calibración   Camera Directions
      2048×2048          .veo            .det
          │              │              │
          └───────┬──────┴──────┬───────┘
                  │             │
                  ▼             ▼
           MODELO DE CÁMARA   ESTADO TEMPORAL
                  │             │
                  └──────┬──────┘
                         ▼
                    RAYO 3D
                         │
                         ▼
                   PLANO Y = 0
                         │
                         ▼
                  (X,Z) EN METROS
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
         VISUALIZACIÓN          ANÁLISIS FUTURO
         CAMPO 2D               jugadores, balón,
                                trayectorias,
                                métricas, eventos
```

La aplicación de desarrollo no forma parte de esta arquitectura conceptual. Es únicamente una herramienta para inspeccionar y validar estas capas.

---

<a id="sec-9-2"></a>

### 9.2. La idea más importante: separar percepción de geometría

Una de las decisiones más útiles del proyecto es no mezclar en una única función:

- detección de personas;
- tracking;
- clasificación de equipos;
- geometría de cámara;
- proyección al campo;
- métricas.

La arquitectura correcta es por capas.

#### 9.2.1. Capa 1: percepción visual

Responde a preguntas como:

```text
¿Dónde hay una persona?
¿Dónde está el balón?
¿Qué detección pertenece al mismo jugador que antes?
¿Es del equipo local, visitante o árbitro?
```

La salida sigue estando principalmente en píxeles.

---

#### 9.2.2. Capa 2: geometría

Responde:

```text
¿Qué posición física del terreno corresponde
a ese punto de imagen?
```

Ésta es la parte ya construida y validada.

Su entrada es un punto de imagen y su salida:

\[
(X,Z)
\]

---

#### 9.2.3. Capa 3: análisis deportivo

Trabaja ya con información física:

```text
track_id
timestamp
X
Z
```

A partir de ahí puede calcular:

- velocidad;
- distancia;
- ocupación;
- zonas;
- separaciones;
- líneas defensivas;
- amplitud;
- profundidad;
- heatmaps;
- eventos.

Esto significa que un detector futuro puede cambiar sin necesidad de modificar la geometría.

De la misma forma, una mejora futura de la calibración puede aumentar la precisión métrica sin obligarnos a reentrenar el detector.

---

<a id="sec-9-3"></a>

### 9.3. Pipeline futuro de jugadores

La siguiente capa natural será la detección y seguimiento de personas.

Conceptualmente:

```text
Dual Lens
   ↓
detección de personas
   ↓
detecciones por frame
   ↓
tracking temporal
   ↓
tracks estables
   ↓
clasificación de equipo / rol
   ↓
estimación del punto de apoyo
   ↓
pixel → (X,Z)
   ↓
trayectoria física
```

Podemos expresar una detección como:

\[
D_i=
(t_i,B_i,c_i)
\]

donde:

- \(t_i\) es el timestamp;
- \(B_i\) es la bounding box;
- \(c_i\) es la confianza.

Una caja puede escribirse:

\[
B_i=(x_i,y_i,w_i,h_i)
\]

Posteriormente el tracker asocia detecciones:

\[
D_i,D_j,\ldots
\]

a una misma identidad temporal:

\[
track\_id=k
\]

y obtenemos:

\[
T_k=
\{
(t_0,B_0),
(t_1,B_1),
\dots
\}
\]

---

<a id="sec-9-4"></a>

### 9.4. Del track a una posición física

El tracker seguirá trabajando inicialmente en imagen.

Para convertirlo al campo necesitaremos estimar un punto de contacto.

Una primera aproximación es:

\[
u_f=x+\frac{w}{2}
\]

\[
v_f=y+h
\]

es decir, el centro inferior de la bounding box.

Después:

\[
(u_f,v_f)
\longrightarrow
(X,Z)
\]

mediante la geometría ya construida.

El track deja entonces de ser:

```text
track 17
frame 4201
bbox = (...)
```

y pasa a convertirse también en:

```text
track 17
t = 140.17 s
X = 12.4 m
Z = -8.1 m
```

Repitiendo esto:

\[
\mathcal{T}_{17}
=
\{
(t_0,X_0,Z_0),
\dots,
(t_n,X_n,Z_n)
\}
\]

Ésta será la representación realmente útil para análisis deportivo.

---

<a id="sec-9-5"></a>

### 9.5. Por qué conviene detectar en Dual Lens

Para nuestro análisis automático, Dual Lens es la fuente más interesante porque conserva las dos observaciones amplias de la cámara.

El vídeo de Seguimiento está pensado principalmente para visualización humana.

Una cámara virtual decide en cada instante:

- dónde mirar;
- cuánto zoom utilizar;
- qué parte del partido dejar fuera del encuadre.

Si detectásemos únicamente sobre Tracking, perderíamos personas que la cámara de seguimiento no está mostrando.

Dual Lens proporciona una cobertura espacial mucho mayor.

La estrategia prevista es:

```text
frame 2048×2048
       ↓
┌───────────────┐
│ LEFT 2048×1024│
├───────────────┤
│RIGHT 2048×1024│
└───────────────┘
       ↓
detector sobre ambas lentes
       ↓
detecciones en coordenadas RAW
```

Cada detección debería conservar también qué lente la produjo.

---

<a id="sec-9-6"></a>

### 9.6. Solape entre lentes

Existe una región donde un mismo jugador puede aparecer simultáneamente en:

```text
LEFT
```

y:

```text
RIGHT
```

Por tanto, inicialmente pueden existir dos observaciones visuales del mismo objeto físico.

No debemos eliminar automáticamente una de ellas antes de comprender su geometría.

Una estrategia futura puede aprovechar que ambas detecciones terminan en:

\[
(X,Z)
\]

Si dos tracks procedentes de lentes diferentes aparecen:

- en tiempos equivalentes;
- con posiciones físicas muy próximas;
- con apariencia compatible;

podemos estudiar su fusión.

Conceptualmente:

\[
\|P_L-P_R\|<\epsilon
\]

puede ser una señal de que ambas observaciones corresponden a la misma persona.

Pero esta fusión pertenece a una capa posterior.

No forma parte todavía de la geometría básica.

---

<a id="sec-9-7"></a>

### 9.7. Clasificación de equipo y rol

Una vez construido un track temporal, podemos añadir atributos semánticos.

Por ejemplo:

```text
team = home
role = player
```

o:

```text
team = away
role = goalkeeper
```

o:

```text
role = referee
```

Conviene separar:

```text
equipo
```

de:

```text
rol
```

porque no son la misma clasificación.

Un portero sigue perteneciendo a uno de los equipos.

Un árbitro tiene un rol diferente pero no pertenece a ninguno de los dos.

La salida futura puede modelarse conceptualmente como:

\[
C_k=
(team_k,role_k,confidence_k)
\]

asociada al track:

\[
track\_id=k
\]

---

<a id="sec-9-8"></a>

### 9.8. Identidad de jugador

Una etapa todavía posterior podría intentar asociar un track con un futbolista concreto.

Posibles fuentes:

- dorsal;
- reconocimiento visual;
- plantilla del partido;
- intervención manual.

La identidad no debería sobrescribir la trayectoria original.

Conviene mantener:

```text
track automático
+
asignación de identidad
```

como capas distintas.

Esto permite corregir una identificación errónea sin alterar detecciones o posiciones ya calculadas.

---

<a id="sec-9-9"></a>

### 9.9. Balón

El balón plantea un problema diferente a las personas:

- es mucho más pequeño;
- puede moverse a gran velocidad;
- sufre blur;
- puede quedar oculto;
- puede estar elevado respecto al suelo.

Por tanto no podremos utilizar siempre:

\[
Y=0
\]

como hacemos para un jugador apoyado en el césped.

Cuando el balón está sobre el terreno, la geometría actual puede utilizarse directamente.

Cuando está en el aire, un único rayo no determina su posición 3D.

En ese caso serían necesarias otras fuentes de información, por ejemplo:

- observación simultánea desde dos lentes;
- trayectoria temporal;
- hipótesis física;
- profundidad aproximada.

Por eso conviene tratar:

```text
player field position
```

y:

```text
ball 3D position
```

como problemas relacionados pero no idénticos.

---

<a id="sec-9-10"></a>

### 9.10. Tiempo como segundo sistema canónico

El campo proporciona un sistema espacial común:

\[
(X,Z)
\]

pero necesitamos también un sistema temporal común.

Utilizamos:

\[
t
\]

en segundos de partido o de medio sincronizado.

Esto permite que elementos procedentes de fuentes diferentes se relacionen mediante el mismo instante.

Por ejemplo:

```text
evento de Veo
start = 302 s
```

y:

```text
track de jugador
t = 302 s
```

pueden analizarse conjuntamente.

La idea conceptual es:

\[
\boxed{
\text{espacio común}=(X,Z)
}
\]

\[
\boxed{
\text{tiempo común}=t
}
\]

Estas dos referencias son suficientes para unir muchas capas futuras.

---

<a id="sec-9-11"></a>

### 9.11. Eventos como información temporal

Además de vídeos y calibración, Veo expone en algunos partidos información estructurada sobre acciones.

En una de nuestras capturas observamos:

```text
37 highlights
```

distribuidos como:

```text
11 goal
26 shot_on_goal
```

Los registros observados incluyen campos como:

```text
id
type
start
duration
thumbnail
videos
```

El dato más importante desde el punto de vista de la arquitectura es:

```text
start
```

porque sitúa una acción sobre la misma línea temporal del partido.

Esto permite representar:

```text
Gol
t = 1432 s
```

sin necesidad de reproducir un clip independiente.

---

<a id="sec-9-12"></a>

### 9.12. Qué NO concluimos de los highlights

Los registros contienen muchos campos adicionales, pero no debemos inventar semántica.

Por ejemplo, en la captura utilizada:

```text
team_association = null
```

y:

```text
involved_players = []
```

Por tanto no podemos afirmar que Veo nos haya proporcionado automáticamente:

- autor del tiro;
- equipo;
- posición de campo;
- confianza del evento.

Igualmente, un campo como:

```text
ai_resolution
```

no se interpreta automáticamente como una probabilidad o confidence score.

La regla es la misma que hemos aplicado en toda la geometría:

> Utilizamos únicamente la semántica que podemos justificar mediante los datos observados.

---

<a id="sec-9-13"></a>

### 9.13. Diferencia entre evento y vídeo de highlight

Otro descubrimiento importante fue que:

```text
evento estructurado
```

y:

```text
vídeo de highlight
```

no deben confundirse.

En la captura estudiada cada highlight podía contener referencias a vídeo, pero no encontramos evidencia suficiente de un único vídeo recopilatorio independiente que debiese considerarse "la película de highlights".

Por eso conceptualizamos un highlight principalmente como:

\[
E=
(type,start,duration,\ldots)
\]

sobre la línea temporal.

El vídeo principal puede simplemente desplazarse a:

\[
t=start
\]

para visualizar la acción.

---

<a id="sec-9-14"></a>

### 9.14. Otros datasets que puede exponer la plataforma

Durante la investigación encontramos referencias o capacidades relacionadas con información como:

```text
events
lineup
roster
player tracking
player moments
heatmaps
physical metrics
match report
shot map
possession
passing data
player stats
```

Sin embargo existe una diferencia fundamental entre:

```text
la plataforma conoce/soporta esta capacidad
```

y:

```text
este partido concreto contiene el dataset
```

No debemos confundir ambas cosas.

Una permission flag o un nombre encontrado en JavaScript no demuestra que el partido actual tenga datos.

Por eso utilizamos estados conceptuales como:

```text
observed
empty
forbidden
unavailable
not_observed
```

Por ejemplo:

```text
not_observed
```

no significa:

```text
Veo no tiene este dato
```

significa simplemente:

> En nuestra captura actual no hemos observado una respuesta que permita afirmar que el dataset está disponible.

---

<a id="sec-9-15"></a>

### 9.15. Qué información de Veo forma parte de la base del proyecto

Podemos dividir los datos descubiertos en tres categorías.

#### 9.15.1. A. Fundamentales para la geometría actual

```text
Dual Lens
camera_alignment.veo
camera_directions.det
shader panorámico / comportamiento de visualización
```

Sin ellos no tendríamos la reconstrucción actual completa.

---

#### 9.15.2. B. Útiles para navegación y análisis temporal

```text
Tracking
Highlights
Eventos
```

Nos permiten relacionar acciones con el reloj del partido.

---

#### 9.15.3. C. Datos potencialmente útiles en el futuro

```text
lineups
roster
player moments
player tracking
heatmaps
physical metrics
reports
```

Solo se utilizarán cuando estén realmente disponibles y conozcamos su formato.

No diseñamos la geometría suponiendo que esos datasets existan.

---

<a id="sec-9-16"></a>

### 9.16. Coexistencia entre datos de Veo y análisis propio

Un principio importante para el futuro es no mezclar silenciosamente:

```text
dato proporcionado por Veo
```

con:

```text
dato generado por nuestro sistema
```

Por ejemplo, si Veo proporciona un evento:

```text
goal at 1432 s
```

y nuestro sistema futuro detecta también un posible gol, ambos pueden coexistir.

De forma conceptual:

```text
source = veo
```

frente a:

```text
source = our_analysis
```

Lo mismo se aplica a:

- posiciones;
- tracks;
- métricas;
- heatmaps;
- identidades.

La procedencia es importante para poder validar y comparar resultados.

---

<a id="sec-9-17"></a>

### 9.17. Correcciones manuales sin destruir resultados automáticos

En un sistema real siempre aparecerán errores.

Por ejemplo:

```text
track 28 fue clasificado como visitante
pero era árbitro
```

La solución no debería ser modificar silenciosamente la salida original del algoritmo.

Es preferible conservar:

```text
resultado automático
+
corrección manual
```

De esta manera podemos saber:

- qué produjo el modelo;
- qué corrigió una persona;
- cuándo;
- sobre qué intervalo temporal.

Esto también será útil para crear nuevos datos de entrenamiento en el futuro.

---

<a id="sec-9-18"></a>

### 9.18. Métricas derivables cuando tengamos tracks físicos

Una vez tengamos:

\[
\mathcal{T}_k=
\{(t_i,X_i,Z_i)\}
\]

podremos calcular magnitudes directamente.

#### 9.18.1. Distancia recorrida

\[
D_k=
\sum_i
\sqrt{
(X_{i+1}-X_i)^2+
(Z_{i+1}-Z_i)^2
}
\]

---

#### 9.18.2. Velocidad instantánea aproximada

\[
v_i=
\frac{
\sqrt{
(X_{i+1}-X_i)^2+
(Z_{i+1}-Z_i)^2
}
}{
t_{i+1}-t_i
}
\]

---

#### 9.18.3. Separación entre dos jugadores

Para dos posiciones:

\[
P_a=(X_a,Z_a)
\]

\[
P_b=(X_b,Z_b)
\]

la distancia es:

\[
d(a,b)
=
\sqrt{
(X_a-X_b)^2+
(Z_a-Z_b)^2
}
\]

---

#### 9.18.4. Centroide de un equipo

Para \(N\) jugadores:

\[
\bar X=
\frac1N
\sum_{i=1}^{N}X_i
\]

\[
\bar Z=
\frac1N
\sum_{i=1}^{N}Z_i
\]

Esto puede aproximar el centro espacial de la estructura del equipo.

---

#### 9.18.5. Anchura

Una medida sencilla:

\[
W_{team}
=
\max_i(Z_i)
-
\min_i(Z_i)
\]

---

#### 9.18.6. Profundidad

\[
L_{team}
=
\max_i(X_i)
-
\min_i(X_i)
\]

Estas métricas son ejemplos de por qué era importante trabajar en metros.

Realizarlas directamente sobre píxeles no tendría significado táctico consistente.

---

<a id="sec-9-19"></a>

### 9.19. Heatmaps

Un heatmap puede construirse acumulando posiciones sobre el plano del campo.

La forma más sencilla es dividir el campo en celdas.

Si:

\[
B_{ij}
\]

es una celda espacial, contamos:

\[
N_{ij}
=
\#\{
P_k\in B_{ij}
\}
\]

Una versión suave puede utilizar una estimación kernel:

\[
\rho(X,Z)
=
\sum_k
\exp
\left(
-\frac{
(X-X_k)^2+
(Z-Z_k)^2
}{
2h^2
}
\right)
\]

donde:

\[
h
\]

controla el suavizado.

Lo importante es que el heatmap vive directamente sobre el mismo terreno métrico.

---

<a id="sec-9-20"></a>

### 9.20. Qué no forma parte todavía del sistema

Aunque la arquitectura ya permite visualizar hacia dónde vamos, actualmente no consideramos implementadas ni validadas:

```text
detección automática completa de personas
tracking de jugadores
clasificación de equipos
clasificación de árbitro/portero
detección de balón
tracking del balón
OCR de dorsales
identidad de futbolistas
métricas físicas finales
heatmaps automáticos
eventos propios
```

Esta documentación explica la base sobre la que podrán construirse.

No debe dar la impresión de que estas capas ya están terminadas.

---



<a id="capitulo-10"></a>

## 10. Reproducibilidad y cierre técnico

<a id="sec-10-1"></a>

### 10.1. Reproducibilidad de la geometría

Una característica importante del trabajo actual es que, una vez obtenidos los recursos necesarios, la geometría puede estudiarse localmente sin depender continuamente del reproductor web.

Para reproducir la base de un partido necesitamos principalmente:

```text
1. Dual Lens MP4
2. camera_alignment.veo
3. camera_directions.det   # para Tracking
```

y el conocimiento del shader/comportamiento panorámico que ya hemos reconstruido.

---

<a id="sec-10-2"></a>

### 10.2. Reproducción mínima de la panorámica

La panorámica se puede reproducir conceptualmente mediante:

```text
leer frame Dual Lens
        ↓
leer calibración
        ↓
para cada píxel panorama:
    calcular lon/lat
    construir dirección
    transformar al mundo
    proyectar sobre LEFT
    proyectar sobre RIGHT
    aplicar distorsión
    muestrear textura
    mezclar
```

No necesitamos una panorámica descargada ya terminada.

La podemos generar a partir del vídeo de las lentes y la calibración.

---

<a id="sec-10-3"></a>

### 10.3. Reproducción mínima de pixel → campo

Para Doble lente:

```text
RAW pixel
    ↓
LEFT / RIGHT
    ↓
deshacer remapeo
    ↓
invertir distorsión
    ↓
lens2camera
    ↓
camera2world
    ↓
ray
    ↓
Y = 0
    ↓
(X,Z)
```

Para Panorámica:

```text
pixel
    ↓
lon / lat
    ↓
dirección de presentación
    ↓
presentation → world
    ↓
Y = 0
    ↓
(X,Z)
```

Para Tracking:

```text
pixel + timestamp
    ↓
camera_directions.det
    ↓
yaw / pitch / fov
    ↓
logic → camera → world
    ↓
Y = 0
    ↓
(X,Z)
```

Esto representa la reproducción mínima del fundamento geométrico.

---

<a id="sec-10-4"></a>

### 10.4. Qué datos deben pertenecer al mismo partido

Una condición crítica de reproducibilidad es no mezclar recursos de grabaciones diferentes.

Debe cumplirse:

```text
Dual Lens A
+
Calibration A
+
Camera Directions A
```

No:

```text
Dual Lens A
+
Calibration B
```

Aunque dos cámaras tengan intrínsecos muy parecidos, `camera2world` depende de cómo estaba instalada la cámara respecto al terreno.

Por tanto, emparejar correctamente los recursos es parte de la validez del sistema.

---

<a id="sec-10-5"></a>

### 10.5. Por qué utilizamos más de un partido para validar

La segunda grabación cumple un papel metodológico importante.

Si todo se hubiese desarrollado únicamente sobre un partido, siempre existiría la posibilidad de haber introducido sin querer:

- valores específicos;
- signos ajustados manualmente;
- offsets;
- dimensiones concretas.

Cuando el mismo algoritmo funciona con otra:

```text
camera2world
camera2presentation
field_width
selected_corners
```

la hipótesis de que hemos reconstruido una regla general gana fuerza.

No demuestra que funcionará automáticamente con cualquier versión futura de Veo, pero sí evita una validación puramente accidental.

---

<a id="sec-10-6"></a>

### 10.6. Dependencia de versiones

Las calibraciones estudiadas utilizan:

```text
calibration_version = 6.2
```

Algunas convenciones pueden cambiar entre versiones.

De hecho, la lógica de `camera2logic` que reconstruimos contiene una condición explícita relacionada con la versión.

Por tanto:

> Una nueva versión de calibración debe verificarse antes de asumir que conserva exactamente las mismas convenciones.

La estrategia correcta es conservar:

- versión de calibración;
- fórmulas observadas;
- tests de reproducción;
- landmarks.

Así podremos detectar una ruptura futura.

---

<a id="sec-10-7"></a>

### 10.7. Qué deberíamos validar en un partido nuevo

Cuando incorporemos una grabación nueva, la validación mínima recomendable sería:

1. comprobar que Dual Lens tiene la estructura esperada;
2. parsear correctamente `.veo`;
3. comprobar versión;
4. reproducir una panorámica;
5. comparar visualmente con Veo;
6. seleccionar varios landmarks;
7. medir error en metros;
8. comprobar `.det` y duración/FPS si necesitamos Tracking;
9. seleccionar puntos Tracking y revisar el campo 2D.

Si estos pasos funcionan, podremos considerar la geometría suficientemente validada para análisis.

---

<a id="sec-10-8"></a>

### 10.8. Posibles mejoras futuras de la calibración

Aunque la baseline actual es útil, existen varias vías para mejorar precisión.

#### 10.8.1. Refinamiento mediante líneas del campo

Podemos utilizar las líneas pintadas como evidencia adicional.

La calibración de Veo sería el prior y buscaríamos únicamente una corrección pequeña.

---

#### 10.8.2. Corrección residual temporal

Especialmente para Tracking:

\[
H_{residual}(t)
\]

podría compensar pequeñas diferencias entre el modelo geométrico y el vídeo final.

---

#### 10.8.3. Optimización multiframe

En lugar de ajustar cada frame independientemente, podríamos minimizar un error conjunto con regularización temporal.

Por ejemplo:

\[
E=
\sum_t
E_{image}(t)
+
\lambda
\sum_t
\|
\theta_{t+1}-\theta_t
\|^2
\]

Esto penaliza cambios bruscos no físicos.

---

#### 10.8.4. Utilizar detecciones como evidencia geométrica

Cuando tengamos tracks estables, la coherencia entre LEFT y RIGHT puede aportar nuevas restricciones.

Pero cualquier refinamiento debe mantenerse separado de la calibración original para poder medir si realmente mejora.

---

<a id="sec-10-9"></a>

### 10.9. Riesgos que debemos evitar en el futuro

El trabajo realizado hasta ahora también nos deja varias reglas prácticas.

#### 10.9.1. No ajustar números hasta que "se vea bien"

Un offset puede ocultar una causa real.

---

#### 10.9.2. No mezclar coordenadas de redimensionado con píxeles RAW

Un detector puede trabajar internamente con:

\[
640\times640
\]

pero la geometría debe recibir coordenadas de la fuente original.

---

#### 10.9.3. No mezclar datos de partidos diferentes

La calibración depende del partido.

---

#### 10.9.4. No considerar una permission flag como un dataset existente

Capacidad de plataforma y datos reales son conceptos diferentes.

---

#### 10.9.5. No convertir la app de desarrollo en la fuente de verdad matemática

La geometría debe poder comprenderse y reproducirse independientemente de la interfaz.

---

#### 10.9.6. No sobrescribir resultados automáticos con correcciones manuales

La procedencia debe mantenerse.

---

<a id="sec-10-10"></a>

### 10.10. Qué hemos construido realmente

Es fácil describir el proyecto como:

```text
"una herramienta para ver vídeos de Veo"
```

pero técnicamente eso no representa el trabajo principal.

Lo que hemos construido hasta este punto es:

> Una reconstrucción interpretable de la geometría de cámara de Veo capaz de transformar observaciones de distintas visualizaciones a un sistema físico común del terreno.

La aplicación web solamente nos permite verlo y probarlo.

La pieza central es:

\[
\boxed{
pixel
\rightarrow
ray
\rightarrow
ground
\rightarrow
(X,Z)
}
\]

---

<a id="sec-10-11"></a>

### 10.11. Por qué esta base es útil para el proyecto del Real Jaén

Las funcionalidades deportivas que se quieren construir posteriormente requieren una representación espacial coherente.

Por ejemplo:

```text
kilómetros recorridos
```

necesita distancia en metros.

```text
heatmaps
```

necesita posiciones en un mismo terreno.

```text
distancias entre líneas
```

necesita coordenadas comparables.

```text
amplitud del equipo
```

necesita posiciones físicas.

```text
jugadas automáticas
```

necesita relacionar posiciones con tiempo.

Sin la geometría, todas estas tareas quedarían ligadas a píxeles de una cámara que se mueve.

Con:

\[
(t,X,Z)
\]

podemos empezar a hablar en términos de fútbol y no de imagen.

---

<a id="sec-10-12"></a>

### 10.12. Resumen final

La evolución completa del proyecto puede condensarse de esta manera:

```text
1. Intentamos calibrar manualmente la imagen
        ↓
2. Construimos homografías y HUMAN anchors
        ↓
3. Estudiamos diversidad de cámara y propagación
        ↓
4. Detectamos límites por distorsión y extrapolación
        ↓
5. Encontramos el vídeo Dual Lens
        ↓
6. Intentamos stitching
        ↓
7. Encontramos camera_alignment.veo
        ↓
8. Reconstruimos el modelo óptico
        ↓
9. Recuperamos el shader panorámico
        ↓
10. Reproducimos la panorámica de Veo
        ↓
11. Validamos con un segundo partido
        ↓
12. Invertimos la óptica RAW
        ↓
13. Construimos pixel → ray
        ↓
14. Intersectamos con Y = 0
        ↓
15. Obtenemos (X,Z) en metros
        ↓
16. Validamos con landmarks
        ↓
17. Encontramos camera_directions.det
        ↓
18. Reconstruimos Tracking → campo
        ↓
19. Unificamos todas las vistas en el mismo terreno
```

El resultado es un sistema donde:

```text
Doble lente ───────┐
Panorámica ────────┤
Interactivo ───────┼──→ (X,Z,t)
Seguimiento ───────┘
```

A partir de aquí, la siguiente etapa deja de ser un problema de calibración básica.

El objetivo pasa a ser obtener automáticamente qué entidad aparece en cada píxel.

El futuro pipeline será:

```text
detección
    ↓
tracking
    ↓
clasificación
    ↓
punto de apoyo
    ↓
geometría ya construida
    ↓
(X,Z,t)
    ↓
análisis deportivo
```

Ése es el fundamento técnico sobre el que se apoyará el resto del proyecto.

---

<a id="sec-10-13"></a>

### 10.13. Conclusión

La parte más importante del trabajo realizado no es haber conseguido dibujar un punto sobre un campo 2D.

Lo importante es conocer **por qué ese punto aparece ahí**.

Podemos seguir la cadena completa:

```text
recurso de Veo
→ parámetro de calibración
→ significado geométrico
→ fórmula
→ transformación
→ rayo
→ intersección
→ coordenada métrica
→ validación experimental
```

Esto evita depender de Veo como una caja negra.

Aunque la plataforma nos proporciona información esencial de calibración, somos capaces de:

- interpretar sus parámetros;
- reproducir matemáticamente su renderer;
- invertir su modelo óptico;
- transformar píxeles a rayos;
- intersectarlos con el terreno;
- obtener metros;
- medir el error;
- identificar qué partes siguen pendientes.

Por tanto, la base actual no se sostiene únicamente en que "la imagen parece correcta".

Se sostiene en una combinación de:

\[
\boxed{
\text{datos observados}
+
\text{fundamento matemático}
+
\text{reverse engineering}
+
\text{validación experimental}
}
\]

Ésa es la razón por la que podemos utilizar esta geometría como fundamento de las siguientes etapas del proyecto.
