# Auditoría de detecciones fuera del campo — resumen para revisión

## Hallazgo

La revisión del Baseline sobre artefactos de detección ya guardados confirma que, en el caso investigado, la persona sí aparece en la salida original del detector cuando está fuera del terreno. La observación se proyecta con la geometría de cámara y campo y luego queda fuera durante el filtrado espacial anterior a la actualización del tracker. Por tanto, la desaparición observada no se explica por una ausencia del detector: la observación no llega al tracker como medición.

Cuando la persona vuelve a una zona aceptada, el Baseline puede iniciar o asociar otra trayectoria según sus reglas actuales. No hay en este flujo evidencia de una identidad persistente que sobreviva automáticamente al descarte espacial. Este diagnóstico identifica dónde se pierde continuidad; por sí solo no determina si la persona pertenece a los futbolistas.

## Implicaciones para Tracking V2

- Mantener disponibles las detecciones originales y registrar por separado por qué una observación no alimentó una trayectoria.
- Tratar la posición respecto al campo como contexto para asociación y evaluación, no como razón suficiente para borrar una observación o cerrar una identidad confirmada.
- Mantener como desconocida la pertenencia cuando la evidencia no permita distinguir futbolistas, personal técnico y espectadores.
- Evaluar continuidad, fragmentación, intercambios y asociaciones falsas frente al Baseline con casos reproducibles, sin inferencia adicional durante esta auditoría.

## Alcance y reproducibilidad

El diagnóstico se obtuvo inspeccionando artefactos existentes; no se ejecutó detector, inferencia de GPU ni análisis completo de vídeo. Los identificadores del partido y clip, las coordenadas y los artefactos de origen se mantienen fuera de este documento y de la rama pública de revisión. El test ligado a esos artefactos permanece local. La prueba publicada para el selector solo verifica que Baseline sea la única opción visible en ese control.

Este resumen documenta un caso observado, pero no demuestra todavía que el mismo filtro explique todas las pérdidas de continuidad. La primera regresión de Tracking V2 debería aislar el filtrado espacial con una secuencia controlada y medir, al menos, continuidad de trayectoria y asociaciones falsas.
