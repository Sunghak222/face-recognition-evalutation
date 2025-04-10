import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

# Config
DATASET_PATH = '/Users/heosunghak/PYTHONWORKSPACE/FaceRecognition/dataset/lfw-deepfunneled/lfw-deepfunneled'
OUTPUT_SESS1 = 'arcface_sess1_data.csv'
OUTPUT_SESS2 = 'arcface_sess2_data.csv'
OUTPUT_SESS3 = 'arcface_sess3_data.csv'
LABEL_SESS1 = 'arcface_sess1_label.csv'
LABEL_SESS2 = 'arcface_sess2_label.csv'
LABEL_SESS3 = 'arcface_sess3_label.csv'

MAX_PEOPLE = 200
IMAGE_PER_PERSON = 3
valid_exts = ['.jpg', '.jpeg', '.png']

# Initialize recognizer
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)

# Extract embeddings
all_entries = []

for person in tqdm(sorted(os.listdir(DATASET_PATH))):
    if len(all_entries) >= MAX_PEOPLE:
        break

    person_dir = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_dir):
        continue

    imgs = [img for img in sorted(os.listdir(person_dir)) if any(img.lower().endswith(ext) for ext in valid_exts)]
    if len(imgs) < IMAGE_PER_PERSON:
        continue

    person_embeddings = []
    used_imgs = []

    for img_name in imgs[:IMAGE_PER_PERSON]:
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img)
        if not faces:
            continue

        emb = faces[0].embedding
        emb = emb / np.linalg.norm(emb)
        person_embeddings.append(emb)
        used_imgs.append(img_name)

    if len(person_embeddings) == IMAGE_PER_PERSON:
        all_entries.append((person, person_embeddings, used_imgs))

# Sort by person name
all_entries.sort(key=lambda x: x[0])

# Save to sessions
sess1, sess2, sess3 = [], [], []
sess1_labels, sess2_labels, sess3_labels = [], [], []

for person, embs, imgs in all_entries:
    sess1.append(embs[0])
    sess2.append(embs[1])
    sess3.append(embs[2])
    sess1_labels.append((person, imgs[0]))
    sess2_labels.append((person, imgs[1]))
    sess3_labels.append((person, imgs[2]))

# Convert and save
sess1 = np.array(sess1)
sess2 = np.array(sess2)
sess3 = np.array(sess3)

np.savetxt(OUTPUT_SESS1, sess1, delimiter=",")
np.savetxt(OUTPUT_SESS2, sess2, delimiter=",")
np.savetxt(OUTPUT_SESS3, sess3, delimiter=",")

pd.DataFrame(sess1_labels, columns=["name", "image"]).to_csv(LABEL_SESS1, index=False)
pd.DataFrame(sess2_labels, columns=["name", "image"]).to_csv(LABEL_SESS2, index=False)
pd.DataFrame(sess3_labels, columns=["name", "image"]).to_csv(LABEL_SESS3, index=False)

print(f"[✓] Saved {len(sess1)} people x 3 sessions")
