import xml.etree.ElementTree as ET
import pandas as pd

# Rutas de los ficheros
PREDATORS_FILE     = "dataset/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
CONVERSATIONS_FILE = "dataset/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
OUTPUT_CSV         = "conversations_dataset.csv"
OUTPUT_STATS       = "conversations_stats.txt"

# Cargar IDs de groomers
with open(PREDATORS_FILE, "r", encoding="utf-8") as f:
    groomer_ids = set(line.strip() for line in f if line.strip())

#  Parsear el XML completo
tree = ET.parse(CONVERSATIONS_FILE)
root = tree.getroot()

data = []

#  Iterar sobre cada conversación
for conv in root.findall("conversation"):
    conv_id = conv.get("id")
    messages = []
    authors = set()

    # Recoger todos los mensajes de esta conversación
    for msg in conv.findall("message"):
        author_elem = msg.find("author")
        text_elem   = msg.find("text")

        if author_elem is None or not author_elem.text:
            continue  # saltamos mensajes sin autor

        author = author_elem.text.strip()
        authors.add(author)

        if text_elem is not None and text_elem.text:
            messages.append(text_elem.text.strip())

    # Etiquetar grooming: 1 si algún autor está en la lista de groomers
    is_grooming = int(bool(authors & groomer_ids))

    # Unir todos los mensajes en un solo string
    full_text = " ".join(messages)

    # Añadir UNA FILA al dataset por cada conversación
    data.append({
        "conversation_id": conv_id,
        "conversation": full_text,
        "grooming": is_grooming
    })

# Volcar a CSV
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

# Calcular estadísticas
total               = len(data)
grooming_count      = sum(item["grooming"] for item in data)
non_grooming_count  = total - grooming_count

# Guardar estadísticas en TXT
with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
    f.write(f"Número de conversaciones: {total}\n")
    f.write(f"Con grooming: {grooming_count}\n")
    f.write(f"Sin grooming: {non_grooming_count}\n")

print(f"Procesadas {total} conversaciones.")
print(f"→ Con grooming: {grooming_count}")
print(f"→ Sin grooming: {non_grooming_count}")
print(f"CSV guardado en {OUTPUT_CSV}")
print(f"Estadísticas guardadas en {OUTPUT_STATS}")
