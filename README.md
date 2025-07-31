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
