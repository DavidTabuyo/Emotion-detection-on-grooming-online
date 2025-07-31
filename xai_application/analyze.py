import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

# 1. Cargar y preparar datos
df = pd.read_csv("dataset/model_2/pan12_emotions.csv")

emotion_cols = [c for c in df.columns if c not in ("conversation_id", "grooming")]
X = df[emotion_cols].to_numpy()
y = df["grooming"].to_numpy()

# 2. Ajustar el modelo a la proporción de clases
pos, neg = (y == 1).sum(), (y == 0).sum()
if pos == 0:
    raise ValueError("El dataset no contiene ningún caso positivo (grooming = 1).")

scale_pos_weight = neg / pos  # recomendado por la doc. de XGBoost

# Configuración  del modelo
base_params = dict(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
)

# 3. Validación cruzada + SHAP
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mean_abs_shap = np.zeros(len(emotion_cols))

for fold, (train_idx, _) in enumerate(cv.split(X, y), start=1):
    X_train, y_train = X[train_idx], y[train_idx]

    model = XGBClassifier(random_state=fold + 42, **base_params)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_train)          # (n_samples, n_features)
    mean_abs_shap += np.abs(shap_vals).mean(axis=0)

# Promedio de importancias sobre los folds
mean_abs_shap /= cv.get_n_splits()

importance_df = (
    pd.DataFrame({"emotion": emotion_cols, "mean_abs_SHAP": mean_abs_shap})
      .sort_values("mean_abs_SHAP", ascending=False)
      .reset_index(drop=True)
)

# Top-10 emociones más influyentes
top10 = importance_df.head(10)
print("\nTop-10 emociones (media |SHAP| en CV):\n", top10)

# Bottom-10 emociones menos influyentes
bottom10 = importance_df.tail(10).sort_values("mean_abs_SHAP", ascending=True)
print("\nBottom-10 emociones (media |SHAP| en CV):\n", bottom10)

# 4. Gráfico de barras del Top-10
plt.figure(figsize=(7, 5))
plt.barh(top10["emotion"][::-1], top10["mean_abs_SHAP"][::-1])
plt.xlabel("Media |SHAP| (contribución absoluta)")
plt.title("Top-10 emociones más influyentes (validación cruzada)")
plt.tight_layout()
plt.show()

# 5. Gráfico de barras del Bottom-10
plt.figure(figsize=(7, 5))
plt.barh(bottom10["emotion"], bottom10["mean_abs_SHAP"])
plt.xlabel("Media |SHAP| (contribución absoluta)")
plt.title("Bottom-10 emociones menos influyentes (validación cruzada)")
plt.tight_layout()
plt.show()
