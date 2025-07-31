#!/usr/bin/env python3
# coding: utf-8

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

def main():
    # Parámetros
    MODEL_NAME = "sangkm/go-emotions-fine-tuned-distilroberta"
    CSV_INPUT  = "conversations_dataset.csv"
    CSV_OUTPUT = "pan12_emotions.csv"

    # Carga del modelo y tokenizador
    print("Cargando modelo y tokenizador…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"  → Modelo cargado en {device}.\n")

    # Lectura del dataset
    print(f"Cargando dataset desde '{CSV_INPUT}'…")
    df = pd.read_csv(CSV_INPUT, encoding="utf-8")
    total = len(df)
    print(f"  → {total} conversaciones cargadas.\n")

    # Validar grooming
    if not set(df['grooming'].unique()).issubset({0, 1}):
        raise ValueError("La columna 'grooming' contiene valores distintos de 0 y 1.")

    # Corregir posibles valores faltantes en 'conversation'
    n_missing = df['conversation'].isna().sum()
    if n_missing:
        print(f"  ¡Atención! {n_missing} conversaciones nulas → las rellenamos con cadena vacía.")
    df['conversation'] = df['conversation'].fillna("").astype(str)

    # Inferencia secuencial, conversación a conversación
    print("Prediciendo emociones conversación a conversación…")
    emotion_vectors = []
    for idx, text in enumerate(df['conversation']):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.sigmoid(outputs.logits).cpu().tolist()[0]

        emotion_vectors.append(scores)

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"  Procesadas {idx + 1}/{total} conversaciones.")

    print("\n→ Predicción completada.\n")

    # Construcción del DataFrame de salida
    print("Construyendo DataFrame final (incluyendo conversation_id, emociones y grooming)…")
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    emotions_df = pd.DataFrame(emotion_vectors, columns=labels)
    emotions_df['grooming'] = df['grooming'].values
    emotions_df['conversation_id'] = df['conversation_id'].values

    # Reordenar columnas para poner conversation_id primero
    cols = ['conversation_id'] + labels + ['grooming']
    emotions_df = emotions_df[cols]

    print("  Columnas resultantes:", list(emotions_df.columns), "\n")

    #  Guardado a CSV
    print(f"Guardando resultados en '{CSV_OUTPUT}'…")
    emotions_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
    print("¡Proceso finalizado con éxito!")

if __name__ == "__main__":
    main()
