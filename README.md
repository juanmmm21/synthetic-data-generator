# Synthetic Data Generator

Un motor automatizado para la generacion, mutacion y filtrado de conjuntos de datos sinteticos (datasets) de alta calidad en formato de Ajuste Fino de Instrucciones (Instruction Tuning) y Optimizacion de Preferencia Directa (DPO - Direct Preference Optimization). Este modulo resuelve el cuello de botella de obtencion de datos limpios para el entrenamiento de modelos de lenguaje dentro de la infraestructura.

## Metodologia de Ingesta y Generacion Sintetica

El pipeline de generacion de datos opera bajo las siguientes fases:

1.  **Mutacion Semantica de Entrada (Self-Instruct Style):**
    *   Para cada topico de entrada, el sistema selecciona verbos de accion, sinonimos tecnicos e instrucciones de formato, y los combina de forma pseudo-aleatoria en plantillas estructurales diversas. Esto evita que el dataset contenga directivas homogeneas o repetitivas.
2.  **Filtro de Redundancia de Contenido (Similitud Jaccard de N-Gramas):**
    *   Cada prompt generado se contrasta contra el historico de prompts guardados computando la similitud de Jaccard basada en tri-gramas de caracteres. Si supera el umbral parametrizable (por defecto 0.45), se descarta de inmediato para forzar la diversidad tematica del dataset.
3.  **Alineacion de Respuestas DPO (Direct Preference Optimization):**
    *   Para cada instruccion aceptada, el motor genera dos variantes:
        *   **Elegida (Chosen):** Una respuesta de alta calidad tecnica, con tipado correcto y explicaciones de arquitectura detalladas en español.
        *   **Rechazada (Rejected):** Una respuesta de baja calidad o alucinada (errores logicos, explicaciones incorrectas o desestructuradas) que sirve para entrenar al modelo en que conductas evitar.

## Requisitos e Instalacion

*   Python 3.8 o superior
*   Pydantic

Para instalar las dependencias locales, ejecute:

```bash
pip install -r requirements.txt
```

## Pruebas y Verificacion

1.  **Ejecutar Suite de Pruebas Unitarias:**
    ```bash
    python -m unittest test_generator.py
    ```
2.  **Ejecutar Demostracion de Generacion:**
    ```bash
    python example.py
    ```
    El script generara 12 muestras de instrucciones y 12 pares DPO sobre arquitectura RAG, embeddings y NanoVectorDB, guardando los resultados en dos archivos JSON locales (`synthetic_instruction_dataset.json` y `synthetic_dpo_dataset.json`) para su posterior versionado.
