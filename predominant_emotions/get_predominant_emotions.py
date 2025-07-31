import pandas as pd

# Carga el dataset
df = pd.read_csv('pan12_emotions.csv')

emociones = [c for c in df.columns if c not in ('conversation_id', 'grooming', 'neutral')]

# Función para calcular media excluyendo valores < 0.10
def media_filtrada(serie):
    filtrada = serie[serie >= 0.10]
    return filtrada.mean() if not filtrada.empty else 0

# Cálculo de medias para grooming = 1
medias_grooming = df[df['grooming'] == 1][emociones].apply(media_filtrada).sort_values(ascending=False)
top5_grooming = medias_grooming.head(5)

# Cálculo de medias para grooming = 0
medias_no_grooming = df[df['grooming'] == 0][emociones].apply(media_filtrada).sort_values(ascending=False)
top5_no_grooming = medias_no_grooming.head(5)

print("Top 5 emociones (mayor media) cuando hay grooming:")
for emocion, valor in top5_grooming.items():
    print(f"{emocion}: {valor:.4f}")

print("\nTop 5 emociones (mayor media) cuando NO hay grooming:")
for emocion, valor in top5_no_grooming.items():
    print(f"{emocion}: {valor:.4f}")
