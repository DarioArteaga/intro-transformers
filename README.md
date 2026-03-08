# Introducción a los Transformers con PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=Jupyter&logoColor=white)

Este repositorio contiene una implementación paso a paso y desde cero de los componentes fundamentales de un **Modelo Transformer**. 

El objetivo de este proyecto es netamente educativo: desarmar la "magia" detrás de los Grandes Modelos de Lenguaje (LLMs) como GPT, Llama o Claude, y demostrar cómo las relaciones lingüísticas se reducen a operaciones matriciales y arquitecturas optimizadas.

## Sobre el proyecto

En lugar de utilizar librerías de alto nivel que ocultan la complejidad matemática, este notebook de Jupyter (`.ipynb`) construye los cimientos de la arquitectura utilizando tensores puros de PyTorch. 

Es una guía interactiva ideal para entender la mecánica interna de la Inteligencia Artificial moderna.

## Arquitectura desglosada

El código está dividido en los siguientes módulos fundamentales:

1. **Configuración de hardware:** Detección dinámica de aceleración por GPU (CUDA/MPS) o ejecución en CPU local.
2. **Multi-Head Self-Attention:** Implementación del mecanismo central donde las palabras (representadas como vectores) calculan su relevancia entre sí mediante matrices de *Query*, *Key* y *Value*.
3. **Positional encoding:** Inyección de contexto secuencial mediante funciones sinusoidales (Seno y Coseno) para que el modelo comprenda el orden posicional de los tokens.
4. **Bloque `Encoder` completo:** Integración de los mecanismos de atención con redes neuronales Feed-Forward (FFN) y capas de normalización residual (Add & Norm).

## Instalación y uso

Este proyecto utiliza `uv` como gestor de paquetes para una instalación ultra rápida. 

### 1. Clonar el repositorio

```markdown
Repositorio: https://github.com/DarioArteaga/intro-transformers
```
```bash
git clone https://github.com/DarioArteaga/intro-transformers.git
cd intro-transformers
```

### 2. Crear el entorno virtual e instalar dependencias

Asegúrate de tener [uv](https://github.com/astral-sh/uv) instalado en tu sistema.

```bash
uv venv
# Activar en Windows:
.venv\Scripts\activate
# Activar en Mac/Linux:
source .venv/bin/activate

uv pip install torch matplotlib seaborn ipykernel


```

*Nota: Si cuentas con una GPU NVIDIA, asegúrate de instalar la versión de PyTorch compilada con soporte para CUDA.*

### 3. Ejecutar

Abre el proyecto en tu editor de código (se recomienda VS Code con la extensión de Jupyter) y selecciona el kernel de Python alojado en `.venv`. Ejecuta las celdas secuencialmente para observar las visualizaciones de los mapas de calor de atención.

## Visualizaciones

El notebook utiliza `seaborn` y `matplotlib` para generar mapas de calor (heatmaps) que permiten observar en tiempo real cómo las "cabezas de atención" asignan pesos estadísticos a las relaciones entre diferentes palabras de una frase de prueba.

## Contribuciones

Si deseas mejorar las visualizaciones, agregar el bloque del *Decoder*, o implementar una *LM Head* de clasificación final, los Pull Requests son bienvenidos.