# Lab09

1. Introducción 

En el presente laboratorio se desarrolló un sistema denominado Chat-Box CUDA Smart Home, cuyo propósito fue aplicar técnicas de procesamiento paralelo mediante CUDA para optimizar el desempeño de un pipeline orientado al reconocimiento de comandos en un entorno de casa inteligente. 

El sistema integra tres componentes principales: procesamiento de lenguaje natural (NLU), análisis de datos de sensores (DATA) y un módulo de fusión y decisión (FUSE). El objetivo fue evaluar el comportamiento del sistema bajo distintas configuraciones de concurrencia (1, 2, 4 y 8 streams), analizando su impacto sobre la latencia, la variabilidad y el rendimiento global en términos de consultas por segundo (QPS). 

2. Fundamento Teórico 

CUDA (Compute Unified Device Architecture) es un modelo de programación paralelo que permite ejecutar múltiples hilos de manera simultánea en la GPU. En este proyecto, se emplearon streams para lograr concurrencia entre la ejecución de kernels y las transferencias de memoria entre CPU y GPU. 

El pipeline implementado se compone de tres etapas: 

NLU: interpreta el texto del usuario, genera representaciones vectoriales y compara con plantillas de comandos. 

DATA: procesa datos simulados de sensores, calculando estadísticas básicas (promedio y desviación). 

FUSE: fusiona los resultados anteriores y decide la acción final en función de reglas y umbrales. 

3. Flujo de ejecución (instrucciones)
# Paso 1: Compilar el código CUDA
nvcc -O3 -std=c++17 main.cu -o chatbox_cuda

# Paso 2: Ejecutar el benchmark (genera los CSV)
chatbox_cuda.exe

# Paso 3: Generar las gráficas
python generate_plots.py
