from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
import torch
import numpy as np
from tqdm import tqdm

# 🔹 Emociones más influyentes según SHAP
selected_emotions = [
    'nervousness', 'joy', 'annoyance', 'disapproval', 'disappointment',
    'gratitude', 'caring', 'optimism', 'sadness', 'approval'
]

# 🔹 Nombre del modelo
MODEL_NAME = "sangkm/go-emotions-fine-tuned-distilroberta"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# 🔹 Dataset
dataset = load_dataset("go_emotions", "raw")
split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
test_dataset = split_dataset["test"]

# 🔹 Todas las emociones
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# 🔹 Conversión de etiquetas
def convert_labels(example):
    example["labels"] = [example[label] for label in emotion_labels]
    return example

test_dataset = test_dataset.map(convert_labels)

# 🔹 Tokenización
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = test_dataset.map(preprocess, batched=True)
cols_to_keep = ["input_ids", "attention_mask", "labels"]
encoded_dataset = encoded_dataset.remove_columns([col for col in encoded_dataset.column_names if col not in cols_to_keep])
encoded_dataset.set_format("torch")

# 🔹 Dataloader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
dataloader = DataLoader(encoded_dataset, batch_size=16, collate_fn=data_collator)

# 🔹 Inferencia
predictions = []
true_labels = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.set_num_threads(16)

for batch in tqdm(dataloader, desc="Evaluando"):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.3).astype(int)  # 🔸 Umbral ajustado

        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())

# 🔹 Filtrar solo las emociones seleccionadas
selected_indices = [emotion_labels.index(e) for e in selected_emotions]
y_true = np.array(true_labels)[:, selected_indices]
y_pred = np.array(predictions)[:, selected_indices]

# 🔹 Filtrar muestras con al menos una emoción verdadera
mask = y_true.sum(axis=1) > 0
y_true = y_true[mask]
y_pred = y_pred[mask]

# 🔹 Métricas
f1_micro = f1_score(y_true, y_pred, average="micro")
f1_samples = f1_score(y_true, y_pred, average="samples")
hamm = hamming_loss(y_true, y_pred)
exactitud = (y_true == y_pred).mean()

print(f"\n✅ F1 micro (filtrado): {f1_micro:.4f}")
print(f"✅ F1 samples (predicción parcial válida): {f1_samples:.4f}")
print(f"✅ Hamming loss: {hamm:.4f}")
print(f"✅ Exactitud por muestra: {exactitud:.4f}")
