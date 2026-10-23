# Tracking V2.0 · Team Discovery

Primer experimento offline e independiente para agrupar observaciones por dos apariencias de equipación, con una tercera salida UNKNOWN/SPECIAL. Este paquete no importa módulos de Tracking V0/V1 y no implementa trayectorias, IDs persistentes, asociación, Kalman ni ReID.

## Método inicial

- Recorta la región central superior del recorte individual y calcula histogramas de cromaticidad RGB normalizada y tono HSV ponderado por saturación.
- Usa dos-means determinista para formar dos grupos anónimos. TEAM_A y TEAM_B se ordenan por la primera muestra del grupo; el código no relaciona grupos con colores ni nombres de clubes.
- Envía recortes ausentes, con pocos píxeles, fuera de la distribución o demasiado ambiguos a UNKNOWN/SPECIAL.
- La geometría del campo solo aporta contexto y estratifica una parte de la muestra. No elimina observaciones ni modifica la característica visual.
- La galería separa predicciones de etiquetas manuales y permite exportar una hoja CSV. Las métricas solo se calculan con manual_team rellenado.

## Entrada y límites del primer experimento

El adaptador lee un índice JSON de recortes ya guardados y, de forma opcional, un JSONL de detecciones para obtener coordenadas de campo. Solo copia los píxeles de la muestra al directorio nuevo de resultados.

El conjunto de recortes disponible fue preparado por un proceso V1 que seleccionó imágenes mediante tracklets y calidad. El paquete V2 ignora los IDs y no ejecuta esa lógica, pero la selección previa introduce sesgo: no es todavía una muestra independiente de todas las detecciones del clip. La revisión local encontró algunas observaciones fuera del rectángulo calibrado entre los recortes guardados, pero no hay cobertura suficiente para afirmar que estén representados espectadores o personal fuera del campo.

El caché de candidatos de detector contiene muchas más cajas que recortes disponibles. No se vuelven a ejecutar detectores ni se procesa el vídeo completo. Para ampliar la muestra de personas no cubiertas, primero hace falta extraer un conjunto pequeño de recortes desde los frames/cajas existentes, preservando cámara, timestamp y geometría, e incluirlos como una fuente separada.

Las imágenes y hojas de evaluación son privadas y deben permanecer en la torre. No se deben añadir a GitHub.

## Ejecución

Desde la raíz del repositorio:

    .venv/bin/python -m tracking_v2.experiment \
      --crop-index /ruta/local/appearance_crops/index.json \
      --detections /ruta/local/detections/frames.jsonl \
      --match-json /ruta/local/match.json \
      --output-dir /ruta/local/tracking_v2/team_discovery/run-01 \
      --sample-count 96 \
      --outside-count 12 \
      --seed 17

El directorio de salida debe estar vacío o no existir. Se generan index.html, assignments.csv, labels.csv, summary.json y copias de los recortes seleccionados. El informe no contiene métricas de clasificación hasta introducir etiquetas de referencia.

Después de completar labels.csv:

    .venv/bin/python -m tracking_v2.metrics \
      --assignments /ruta/local/tracking_v2/team_discovery/run-01/assignments.csv \
      --labels /ruta/local/tracking_v2/team_discovery/run-01/labels.csv \
      --output /ruta/local/tracking_v2/team_discovery/run-01/metrics.json

La evaluación prueba ambas permutaciones de nombres TEAM_A/TEAM_B y conserva UNKNOWN. Informa precisión selectiva, cobertura, matriz de confusión y resultados por contexto de campo, luz, rol, cámara y tamaño del recorte.

## Pruebas

    .venv/bin/python -m pytest tests/test_tracking_v2_team_discovery.py -q
