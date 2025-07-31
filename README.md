# Detección de emociones en grooming online mediante PLN /  
Emotion Detection in Online Grooming Using NLP

## 📘 Descripción del proyecto / Project Description

### 🇪🇸 Español

Este repositorio contiene el código fuente y los recursos desarrollados para el Trabajo de Fin de Grado titulado **"Detección de emociones en grooming online usando técnicas de Procesamiento de Lenguaje Natural"**.  
El objetivo principal es explorar la **detección temprana de conversaciones potencialmente predatorias** (grooming) a través del análisis automático de emociones presentes en el texto.

Para ello, se combinan modelos de clasificación emocional preentrenados (como RoBERTa y ModernBERT, ajustados sobre el conjunto de datos [GoEmotions](https://aclanthology.org/2020.acl-main.372)) con conversaciones del corpus [PAN12](https://pan.webis.de/clef12/pan12-web/sexual-predator-identification.html), etiquetadas binariamente como grooming/no grooming.

El sistema genera vectores de emociones multietiqueta para cada conversación y emplea técnicas de inteligencia artificial explicable (como SHAP) para identificar las emociones más relevantes en la predicción.

### 🇬🇧 English

This repository contains the source code and resources developed for the Bachelor's Thesis titled **"Emotion Detection in Online Grooming Using Natural Language Processing Techniques"**.

The main goal is to explore the **early detection of potentially predatory conversations** (grooming) through automatic analysis of emotional content in text.

The approach combines pretrained emotion classification models (such as RoBERTa and ModernBERT fine-tuned on the [GoEmotions](https://aclanthology.org/2020.acl-main.372) dataset) with binary-labeled conversations from the [PAN12](https://pan.webis.de/clef12/pan12-web/sexual-predator-identification.html) corpus (grooming / non-grooming).

The system generates multi-label emotion vectors for each conversation and applies explainable AI techniques (like SHAP) to identify the most relevant emotional features in the prediction process.

---

Este repositorio forma parte de un trabajo académico realizado en la **Universidad de León (España)**, dentro del Grado en Ingeniería Informática.

_This repository is part of an academic project developed at the **University of León (Spain)** as part of the Bachelor's Degree in Computer Engineering._

## 📂 Requisitos previos / Prerequisites

### 🇪🇸 Español

Antes de ejecutar el proyecto, asegúrate de cumplir los siguientes requisitos:

1. **Python 3.10 o superior** debe estar instalado en el sistema. Puedes comprobarlo con:

   python --version

2. Se recomienda crear un **entorno virtual** para instalar las dependencias de forma aislada:

   python -m venv venv  
   source venv/bin/activate   # En Linux/macOS  
   .\venv\Scripts\activate     # En Windows

3. Instala las dependencias necesarias ejecutando:

   pip install -r requirements.txt

4. Descarga manualmente el conjunto de datos **PAN12 Sexual Predator Identification Corpus** desde la fuente oficial:

   🔗 https://pan.webis.de/clef12/pan12-web/sexual-predator-identification.html

   Una vez descargado, coloca los siguientes archivos en la carpeta:

   model_pan12/dataset/

   Archivos requeridos:

   - pan12-sexual-predator-identification-training-corpus-2012-05-01.xml  
   - pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt

⚠️ **Importante**: El dataset PAN12 no se incluye en este repositorio por motivos legales. Es necesario solicitarlo y aceptar sus condiciones de uso para obtenerlo.

---

### 🇬🇧 English

Before running the project, make sure you meet the following requirements:

1. **Python 3.10 or higher** must be installed on your system. You can check it with:

   python --version

2. It is recommended to create a **virtual environment** to install dependencies in isolation:

   python -m venv venv  
   source venv/bin/activate   # On Linux/macOS  
   .\venv\Scripts\activate     # On Windows

3. Install the required dependencies by running:

   pip install -r requirements.txt

4. Manually download the **PAN12 Sexual Predator Identification Corpus** from the official source:

   🔗 https://pan.webis.de/clef12/pan12-web/sexual-predator-identification.html

   Once downloaded, place the following files inside the folder:

   model_pan12/dataset/

   Required files:

   - pan12-sexual-predator-identification-training-corpus-2012-05-01.xml  
   - pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt

⚠️ **Important**: The PAN12 dataset is not included in this repository due to legal and licensing restrictions. You must request it and accept the terms of use in order to obtain it.

## 🧠 Procesamiento inicial del dataset PAN12 / Initial Processing of the PAN12 Dataset

### 🇪🇸 Español

Dentro de la carpeta `model_pan12`, una vez colocado el dataset PAN12 como se indicó en los requisitos previos, se encuentra el primer script que permite convertir el corpus original en XML en un formato CSV estructurado y utilizable para los siguientes pasos del proyecto.

Este script realiza lo siguiente:

- Carga los identificadores de groomers desde el fichero `.txt`.
- Parsea el archivo XML que contiene todas las conversaciones.
- Extrae todos los mensajes de cada conversación y los concatena en un solo bloque de texto.
- Etiqueta la conversación como `1` si participa algún groomer, o como `0` en caso contrario.
- Exporta el resultado a un fichero `conversations_dataset.csv` con tres columnas:
  - `conversation_id`
  - `conversation`
  - `grooming`
- Además, genera un archivo `conversations_stats.txt` con estadísticas generales del conjunto.

Este paso es fundamental para preparar el dataset con las etiquetas necesarias de cara a la anotación emocional posterior.

---

### 🇬🇧 English

Inside the `model_pan12` folder, once the PAN12 dataset has been placed as described earlier, you will find the first processing script. Its purpose is to convert the original XML corpus into a structured CSV file, ready for emotional annotation and further modeling.

The script performs the following tasks:

- Loads groomer identifiers from the `.txt` file.
- Parses the XML file containing all chat conversations.
- Extracts and concatenates all messages from each conversation.
- Labels the conversation as `1` if it involves any known groomer, or `0` otherwise.
- Exports the results into a `conversations_dataset.csv` file with three columns:
  - `conversation_id`
  - `conversation`
  - `grooming`
- Additionally, it generates a `conversations_stats.txt` file with basic dataset statistics.

This preprocessing step is essential for creating a clean and labeled dataset to be used in the next stage: emotional feature extraction.

---

## 🎯 Generación del dataset emocional / Emotion Vector Generation

### 🇪🇸 Español

Una vez generado el archivo `conversations_dataset.csv` a partir del corpus PAN12, el siguiente paso consiste en transformar cada conversación en un vector numérico que represente la intensidad de cada una de las 28 emociones del conjunto **GoEmotions**.

Para ello, se utiliza un modelo preentrenado de clasificación emocional (por ejemplo: `sangkm/go-emotions-fine-tuned-distilroberta`), que analiza el texto completo de cada conversación y asigna una probabilidad a cada emoción.

#### 📁 Prerrequisitos

Antes de ejecutar este paso, es necesario:

- Copiar el archivo `conversations_dataset.csv` generado previamente en la carpeta `grooming_dataset_generation/`.
- Asegurarse de tener conexión a internet para que Hugging Face pueda descargar el modelo seleccionado, si no está en caché.
- Verificar que se dispone de suficiente memoria (se recomienda uso de GPU para acelerar el proceso).

#### 📥 Estado del dataset antes del script

El archivo `conversations_dataset.csv` contiene:

- `conversation_id`: identificador de la conversación.
- `conversation`: texto completo concatenado de los mensajes.
- `grooming`: etiqueta binaria (`1` si hay grooming, `0` si no).

Ejemplo:

| conversation_id | conversation                               | grooming |
|-----------------|--------------------------------------------|----------|
| 23              | Hi, how are you? I'm fine, you? ...        | 1        |
| 24              | Hello! Let's play a game...                | 0        |

#### 📤 Resultado del script

El script genera un nuevo archivo llamado `pan12_emotions.csv`, que añade para cada conversación un vector con **28 columnas correspondientes a las emociones** del dataset GoEmotions, más las columnas `conversation_id` y `grooming`.

Ejemplo:

| conversation_id | joy  | anger | fear | ... | remorse | grooming |
|-----------------|------|-------|------|-----|---------|----------|
| 23              | 0.76 | 0.03  | 0.05 | ... | 0.01    | 1        |
| 24              | 0.10 | 0.00  | 0.20 | ... | 0.00    | 0        |

Este archivo resultante se puede guardar en subcarpetas distintas según el modelo utilizado. Por ejemplo:

- `grooming_dataset_generation/model_1/pan12_emotions.csv`
- `grooming_dataset_generation/model_2/pan12_emotions.csv`

Esto permite comparar el rendimiento de distintos modelos de clasificación emocional en pasos posteriores del proyecto.

---

### 🇬🇧 English

After generating `conversations_dataset.csv` from the PAN12 corpus, the next step is to transform each conversation into a numerical vector that represents the intensity of each of the 28 emotions from the **GoEmotions** dataset.

This is done using a pretrained emotion classification model (e.g., `sangkm/go-emotions-fine-tuned-distilroberta`), which processes the full conversation text and returns one probability score per emotion.

#### 📁 Prerequisites

Before running this step, make sure to:

- Place the previously generated `conversations_dataset.csv` into the `grooming_dataset_generation/` folder.
- Ensure internet access so the Hugging Face model can be downloaded (if not already cached).
- Have sufficient memory available (GPU is recommended for faster inference).

#### 📥 Dataset before the script

The input CSV contains:

- `conversation_id`: conversation identifier.
- `conversation`: full text of all messages.
- `grooming`: binary label (`1` for grooming, `0` otherwise).

Example:

| conversation_id | conversation                               | grooming |
|-----------------|--------------------------------------------|----------|
| 23              | Hi, how are you? I'm fine, you? ...        | 1        |
| 24              | Hello! Let's play a game...                | 0        |

#### 📤 Output of the script

The script outputs a new file named `pan12_emotions.csv`, where each row now includes 28 emotion values (between 0 and 1), in addition to `conversation_id` and `grooming`.

Example:

| conversation_id | joy  | anger | fear | ... | remorse | grooming |
|-----------------|------|-------|------|-----|---------|----------|
| 23              | 0.76 | 0.03  | 0.05 | ... | 0.01    | 1        |
| 24              | 0.10 | 0.00  | 0.20 | ... | 0.00    | 0        |

The output file can be saved into different subdirectories depending on the model used. For example:

- `grooming_dataset_generation/model_1/pan12_emotions.csv`
- `grooming_dataset_generation/model_2/pan12_emotions.csv`

This structure allows for later comparison between different emotion models in grooming detection tasks.

---

## 📊 Interpretabilidad con SHAP / Explainability with SHAP

### 🇪🇸 Español

Una vez elegido el modelo de clasificación emocional más adecuado (en este caso, el **modelo 2** por su mejor rendimiento en las métricas de clasificación), se procede a aplicar una técnica de **Inteligencia Artificial Explicable (XAI)** llamada **SHAP** para analizar qué emociones son más relevantes a la hora de predecir grooming.

SHAP (*SHapley Additive exPlanations*) es una técnica basada en la teoría de juegos que calcula la contribución individual de cada variable (en este caso, cada emoción) a la decisión del modelo. Es decir, estima cuánto "empuja" cada emoción la predicción final hacia un resultado positivo o negativo. SHAP permite tanto interpretaciones locales (por ejemplo, de una conversación específica) como globales (en todo el dataset).

#### 📁 Ubicación del script

Este análisis se encuentra implementado en la carpeta:

`xai_application/`

Se aplica sobre el dataset enriquecido con emociones, concretamente sobre el archivo:

`grooming_dataset_generation/model_2/pan12_emotions.csv`

#### ⚙️ ¿Qué hace el script?

- Carga el dataset emocional generado por el modelo 2.
- Entrena un clasificador `XGBoost` sobre los vectores de emociones para predecir grooming.
- Ajusta automáticamente el parámetro de desbalance de clases (`scale_pos_weight`).
- Aplica validación cruzada estratificada (5 folds) para asegurar la robustez del análisis.
- Calcula el valor absoluto medio de SHAP para cada emoción en todas las particiones.
- Genera dos gráficos de barras:
  - **Top-10 emociones más influyentes** en la detección de grooming.
  - **Bottom-10 emociones menos relevantes** según el modelo.

Este análisis proporciona una visión clara y cuantitativa de qué emociones son más determinantes para identificar patrones de grooming, permitiendo construir modelos más interpretables y eficientes.

---

### 🇬🇧 English

Once the most suitable emotion classification model has been selected (in this case, **model 2**, which achieved the best classification metrics), we apply an **Explainable AI (XAI)** technique called **SHAP** to analyze which emotions are most relevant in predicting grooming.

SHAP (*SHapley Additive exPlanations*) is a method based on game theory that calculates the individual contribution of each feature (emotion) to a model’s prediction. It estimates how much each emotion increases or decreases the probability of a conversation being classified as grooming. SHAP can be used both locally (per example) and globally (entire dataset).

#### 📁 Script location

This analysis is implemented in the folder:

`xai_application/`

It is executed over the emotion-enriched dataset generated by:

`grooming_dataset_generation/model_2/pan12_emotions.csv`

#### ⚙️ What does the script do?

- Loads the emotion vectors produced by model 2.
- Trains an `XGBoost` classifier using those vectors to predict grooming.
- Automatically adjusts for class imbalance using `scale_pos_weight`.
- Applies 5-fold stratified cross-validation to ensure robustness.
- Computes the mean absolute SHAP value for each emotion.
- Generates two bar plots:
  - **Top 10 most influential emotions** for grooming prediction.
  - **Bottom 10 least influential emotions**, considered irrelevant by the model.

This SHAP analysis provides a transparent view of the model’s behavior, highlighting which emotional features are most critical for identifying grooming patterns.

---

## 🎁 Bonus: Análisis comparativo de emociones predominantes

### 🇪🇸 Español

Como complemento al análisis con SHAP, se incluye un script adicional que permite identificar las **emociones predominantes** (con mayor valor medio) en conversaciones etiquetadas como grooming (`1`) y en aquellas sin grooming (`0`).

Este análisis se realiza sobre el archivo `pan12_emotions.csv`, previamente generado, y se basa en calcular la **media de activación** de cada emoción, considerando únicamente aquellas cuyo valor supera el umbral de 0.10 (para eliminar ruido de baja activación).

#### ¿Qué hace el script?

- Carga el dataset con los vectores de emociones.
- Excluye la emoción `neutral` por su alta frecuencia y bajo valor discriminativo.
- Calcula la media de activación de cada emoción en:
  - Conversaciones con grooming (`grooming == 1`)
  - Conversaciones sin grooming (`grooming == 0`)
- Muestra las **5 emociones con mayor media** en cada grupo.

Este análisis cualitativo permite observar diferencias en el perfil emocional de las conversaciones y puede ser útil para generar hipótesis o complementar la interpretación de los resultados del modelo.

---

### 🇬🇧 English

As a complement to the SHAP analysis, an additional script is included to identify the **predominant emotions** (highest average scores) in conversations labeled as grooming (`1`) and those without grooming (`0`).

This analysis is performed on the `pan12_emotions.csv` file and calculates the **mean activation** of each emotion, considering only values greater than 0.10 to filter out low-signal noise.

#### What does the script do?

- Loads the dataset with the emotion vectors.
- Excludes the `neutral` emotion due to its high frequency and low discriminative power.
- Computes the mean activation of each emotion in:
  - Grooming conversations (`grooming == 1`)
  - Non-grooming conversations (`grooming == 0`)
- Displays the **top 5 emotions** with the highest mean in each case.

This qualitative analysis offers insights into emotional differences between classes and can support the interpretation of the model's behavior.

---
 
 ## 🧪 Extra: Evaluación de modelos usando solo las emociones más influyentes

### 🇪🇸 Español

Este script permite evaluar la calidad de los modelos preentrenados de clasificación emocional cuando se restringe su salida únicamente a las **10 emociones más relevantes** determinadas previamente mediante SHAP.

El objetivo es comprobar si un subconjunto reducido de emociones es suficiente para mantener un buen rendimiento, lo que puede ser útil en contextos donde se busca mayor eficiencia o interpretabilidad.

#### ⚙️ ¿Qué hace el script?

- Usa el dataset original **GoEmotions** a través de `datasets` de Hugging Face.
- Tokeniza y prepara el conjunto de prueba (`test split`).
- Aplica el modelo preentrenado elegido (`sangkm/go-emotions-fine-tuned-distilroberta`) sobre los textos.
- Filtra solo las predicciones y etiquetas correspondientes a las **10 emociones más influyentes** según el análisis con SHAP.
- Evalúa el rendimiento con varias métricas:
  - F1-score micro
  - F1-score por muestra (`samples`)
  - Hamming loss
  - Exactitud por muestra

Este enfoque proporciona una validación clara de que el modelo sigue funcionando correctamente incluso cuando se reduce la dimensionalidad emocional, lo que puede ser especialmente útil para sistemas explicables o integrados en tiempo real.

---

### 🇬🇧 English

This script evaluates the performance of a pretrained emotion classification model when restricting its output to only the **top 10 most influential emotions** identified earlier using SHAP.

The goal is to check whether a reduced emotional feature space still retains predictive power, which is useful for improving model efficiency and interpretability.

#### ⚙️ What does the script do?

- Loads the original **GoEmotions** dataset via Hugging Face `datasets`.
- Preprocesses and tokenizes the test split.
- Applies the selected pretrained model (`sangkm/go-emotions-fine-tuned-distilroberta`) on the test data.
- Filters predictions and ground truth to keep only the **10 key emotions** selected with SHAP.
- Evaluates the model using several metrics:
  - F1-score (micro average)
  - F1-score per sample
  - Hamming loss
  - Per-sample accuracy

This evaluation confirms that the model can perform well even with a limited set of emotional features, supporting efficient and interpretable applications in grooming detection.

---

## 🔗 Resultados del proyecto / Project Results

### 🇪🇸 Español

Los resultados completos de este Trabajo de Fin de Grado, incluyendo gráficos, análisis de métricas, visualizaciones con SHAP y conclusiones, pueden consultarse en una publicación disponible en mi perfil de LinkedIn:

👉 [Ver publicación en LinkedIn](https://www.linkedin.com/in/tu-nombre-aquí) *(enlace pendiente de actualizar)*

No dudes en contactarme por esa vía si tienes preguntas, sugerencias o interés en colaborar.

---

### 🇬🇧 English

The complete results of this Bachelor's Thesis project — including charts, evaluation metrics, SHAP visualizations and conclusions — are available in a LinkedIn post on my profile:

👉 [See LinkedIn post](https://www.linkedin.com/in/your-name-here) *(link to be updated)*

Feel free to reach out via LinkedIn for questions, suggestions, or potential collaborations.

---
