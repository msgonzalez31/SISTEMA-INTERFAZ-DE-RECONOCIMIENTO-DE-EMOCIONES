# Sistema / Interfaz de Reconocimiento de Emociones con Deep Learning

Este repositorio contiene los scripts desarrollados para un sistema de reconocimiento de emociones básicas, posiblemente aplicable a contextos educativos y terapéuticos, especialmente con niños con Trastorno del Espectro Autista (TEA).

## Estructura del Proyecto

### Modelos entrenados con 7 clases de emociones:
- 😃 Alegría
- 😢 Tristeza
- 😐 Neutralidad
- 😮 Sorpresa
- 😱 Miedo
- 😠 Enojo
- 🤢 Disgusto

**Ubicación**: carpeta `modelos_7_clases/`

### Modelos entrenados con 3 clases de emociones:
- 😃 Alegría
- 😢 Tristeza
- 😐 Neutralidad

**Ubicación**: carpeta `modelos_3_clases/`

## Modelos utilizados

- **CNN** desarrollada desde cero
- **VGG16** (transfer learning)
- **ResNet50** (transfer learning)

Cada modelo tiene un script para cada configuración (3 o 7 clases).
