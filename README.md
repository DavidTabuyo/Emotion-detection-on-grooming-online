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

...

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

