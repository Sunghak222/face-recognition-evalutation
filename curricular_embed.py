import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from insightface.app import FaceAnalysis
from backbone.model_irse import IR_101  # IR_101 모델 import
from insightface.utils import face_align

# --------------------
# Config
# --------------------
DATASET_PATH = "/Users/heosunghak/PYTHONWORKSPACE/FaceRecognition/dataset/lfw-deepfunneled/lfw-deepfunneled"
MODEL_PATH = "/Users/heosunghak/PYTHONWORKSPACE/FaceRecognition/CurricularFace-master/backbone/CurricularFace_Backbone.pth"
OUTPUT_SESS1 = "curricular_sess1_data.csv"
OUTPUT_SESS2 = "curricular_sess2_data.csv"
OUTPUT_SESS3 = "curricular_sess3_data.csv"
LABEL_SESS1 = "curricular_sess1_label.csv"
LABEL_SESS2 = "curricular_sess2_label.csv"
LABEL_SESS3 = "curricular_sess3_label.csv"

IMG_SIZE = (112, 112)
MAX_PEOPLE = 200
IMAGE_PER_PERSON = 3

# --------------------
# Load CurricularFace model
# --------------------
device = torch.device("cpu")
model = IR_101(input_size=IMG_SIZE)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# --------------------
# Initialize detector
# --------------------
detector = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
detector.prepare(ctx_id=-1)

# --------------------
# Image transform
# --------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --------------------
# Face preprocessing
# --------------------
def preprocess_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Cannot read image: {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.get(img)
    if not faces:
        print(f"[!] No face detected in {image_path}")
        return None
    box = faces[0].bbox.astype(int)
    cropped = img[box[1]:box[3], box[0]:box[2]]
    if cropped.size == 0:
        print(f"[!] Cropped face is empty in {image_path}")
        return None
    resized = face_align.norm_crop(img, faces[0].kps, image_size=IMG_SIZE[0])  
    #resized = cv2.resize(cropped, IMG_SIZE)
    tensor = transform(resized).unsqueeze(0)
    return tensor

# --------------------
# Extract embeddings
# --------------------
sess1, sess2, sess3 = [], [], []
sess1_labels, sess2_labels, sess3_labels = [], [], []
people_count = 0

for person in tqdm(sorted(os.listdir(DATASET_PATH))):
    if people_count >= MAX_PEOPLE:
        break

    person_dir = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_dir):
        continue

    valid_imgs = [img for img in sorted(os.listdir(person_dir))
                  if any(img.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]

    if len(valid_imgs) < IMAGE_PER_PERSON:
        continue

    embeddings = []
    used_imgs = []

    for img_name in valid_imgs[:IMAGE_PER_PERSON]:
        img_path = os.path.join(person_dir, img_name)
        face_tensor = preprocess_face(img_path)
        if face_tensor is None:
            continue
        with torch.no_grad():
            emb, _ = model(face_tensor)  # output: (embedding, conv_out)
            emb = torch.nn.functional.normalize(emb).squeeze().numpy()

        embeddings.append(emb)
        used_imgs.append(img_name)
        if len(embeddings) == IMAGE_PER_PERSON:
            break

    if len(embeddings) == IMAGE_PER_PERSON:
        sess1.append(embeddings[0])
        sess2.append(embeddings[1])
        sess3.append(embeddings[2])
        sess1_labels.append((person, used_imgs[0]))
        sess2_labels.append((person, used_imgs[1]))
        sess3_labels.append((person, used_imgs[2]))
        people_count += 1
    else:
        print(f"[!] Only {len(embeddings)} valid images for {person}, skipping...")

print(f"Finished. Total subjects processed: {people_count}")

def sort_by_name(embeddings, labels):
    df = pd.DataFrame(labels, columns=["name", "image"])
    df["emb"] = embeddings
    df_sorted = df.sort_values("name")
    emb_sorted = np.stack(df_sorted["emb"].values)
    labels_sorted = df_sorted.drop(columns="emb")
    return emb_sorted, labels_sorted

# --------------------
# Save to CSV
# --------------------
sess1, df1 = sort_by_name(sess1, sess1_labels)
sess2, df2 = sort_by_name(sess2, sess2_labels)
sess3, df3 = sort_by_name(sess3, sess3_labels)

np.savetxt(OUTPUT_SESS1, sess1, delimiter=",")
np.savetxt(OUTPUT_SESS2, sess2, delimiter=",")
np.savetxt(OUTPUT_SESS3, sess3, delimiter=",")
df1.to_csv(LABEL_SESS1, index=False)
df2.to_csv(LABEL_SESS2, index=False)
df3.to_csv(LABEL_SESS3, index=False)

print("Saved to CSV:")
print(f" - {OUTPUT_SESS1}, {OUTPUT_SESS2}, {OUTPUT_SESS3}")
print(f" - {LABEL_SESS1}, {LABEL_SESS2}, {LABEL_SESS3}")