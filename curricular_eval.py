import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# --------------------
# Load data
# --------------------
GALLERY_PATH = "curricular_sess1_data.csv" 
PROBE_PATH1  = "curricular_sess2_data.csv"  
PROBE_PATH2  = "curricular_sess3_data.csv"   
LABEL_GALLERY = "curricular_sess1_label.csv"
LABEL_PROBE1  = "curricular_sess2_label.csv"
LABEL_PROBE2  = "curricular_sess3_label.csv"

gallery_data = np.genfromtxt(GALLERY_PATH, delimiter=',')
probe_data1  = np.genfromtxt(PROBE_PATH1, delimiter=',')
probe_data2  = np.genfromtxt(PROBE_PATH2, delimiter=',')
probe_data = np.concatenate((probe_data1, probe_data2), axis=0)

gallery_labels = pd.read_csv(LABEL_GALLERY)["name"].values
probe_labels1  = pd.read_csv(LABEL_PROBE1)["name"].values
probe_labels2  = pd.read_csv(LABEL_PROBE2)["name"].values
probe_labels = np.concatenate((probe_labels1, probe_labels2), axis=0)

# sort all identities
sorted_names = sorted(set(gallery_labels)) 
gallery_order = np.argsort([sorted_names.index(name) for name in gallery_labels])
probe_order   = np.argsort([sorted_names.index(name) for name in probe_labels if name in sorted_names])

gallery_data = gallery_data[gallery_order]
gallery_labels = gallery_labels[gallery_order]
probe_data = probe_data[probe_order]
probe_labels = probe_labels[probe_order]

# Cosine similarity
def cal_cosine(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def cal_dist(arr1, arr2):
    dist_mat = np.empty((arr1.shape[0], arr2.shape[0]), dtype=np.float32)
    for i in range(arr1.shape[0]):
        for j in range(arr2.shape[0]):
            dist_mat[i, j] = cal_cosine(arr1[i], arr2[j])
    return dist_mat

cos_mat = cal_dist(probe_data, gallery_data)
print("[DEBUG] cos_mat shape:", cos_mat.shape)
print("[DEBUG] First 5 rows of cos_mat:\n", cos_mat[:5, :5])

# --------------------
# Genuine & Imposter
# --------------------
cos_garr, cos_iarr, labels_binary, scores = [], [], [], []

for i in range(len(probe_labels)):
    for j in range(len(gallery_labels)):
        sim = cos_mat[i, j]
        is_genuine = (probe_labels[i] == gallery_labels[j])
        scores.append(sim)
        labels_binary.append(1 if is_genuine else 0)
        if is_genuine:
            cos_garr.append(sim)
        else:
            cos_iarr.append(sim)

cos_garr = np.array(cos_garr)
cos_iarr = np.array(cos_iarr)

print("[DEBUG] First 5 Genuine Pairs and Similarities:")
for i in range(5):
    print(f"→ Probe[{i}] = {probe_labels[i]} vs Gallery[{i}] = {gallery_labels[i]} → Sim: {cos_garr[i]:.4f}")

print("[DEBUG] First 5 Imposter Pairs and Similarities:")
for i in range(5):
    print(f"→ Probe[0] = {probe_labels[0]} vs Gallery[{i+1}] = {gallery_labels[i+1]} → Sim: {cos_iarr[i]:.4f}")

print(f"[DEBUG] Genuine count: {len(cos_garr)}")
print(f"[DEBUG] Imposter count: {len(cos_iarr)}")
print(f"[DEBUG] Genuine mean: {cos_garr.mean():.4f}, std: {cos_garr.std():.4f}")
print(f"[DEBUG] Imposter mean: {cos_iarr.mean():.4f}, std: {cos_iarr.std():.4f}")

# --------------------
# 히스토그램 분포 시각화
# --------------------
plt.figure(figsize=(8, 5))
plt.title("Cosine Similarity Distribution (CurricularFace)")
plt.xlabel("Cosine Similarity")
sns.histplot(cos_garr, kde=True, stat="density", label="Genuine")
sns.histplot(cos_iarr, kde=True, stat="density", label="Imposter")
plt.grid()
plt.legend()
plt.show()

# --------------------
# FAR & FRR
# --------------------
def cal_far_frr(garr, iarr, resolu=1000):
    d_max = max(garr.max(), iarr.max())
    d_min = min(garr.min(), iarr.min())
    far = np.empty(resolu)
    frr = np.empty(resolu)
    for i, d in enumerate(np.linspace(d_min, d_max, resolu)):
        far[i] = np.sum(iarr > d) / iarr.size
        frr[i] = np.sum(garr < d) / garr.size
    return far, frr

resolu = 1000
cos_far, cos_frr = cal_far_frr(cos_garr, cos_iarr, resolu)

plt.figure(figsize=(8, 5))
plt.title("Cosine Similarity FAR & FRR (CurricularFace)")
plt.xlabel("Threshold Index")
plt.ylabel("Rate")
plt.plot(np.arange(resolu), cos_far, label="FAR")
plt.plot(np.arange(resolu), cos_frr, label="FRR")
plt.grid()
plt.legend()
plt.show()

# --------------------
# ROC Curve (Cosine Distance)
# --------------------
plt.figure(figsize=(6, 6))
plt.title("Cosine Similarity ROC Curve (CurricularFace)")
plt.xlabel("False Accept Rate (FAR)")
plt.ylabel("Genuine Accept Rate (1 - FRR)")
plt.xscale('log')
plt.xlim([1e-5, 1])
plt.ylim([0.9, 1.0])
plt.plot(cos_far, 1 - cos_frr, label="Cosine", color='blue')
plt.legend()
plt.grid()
plt.show()

print(f"[DEBUG] ROC sample: FAR = {cos_far[100]:.4f}, GAR = {1 - cos_frr[100]:.4f}")

#equal error rate(EER)
def cal_eer(far, frr):
    return far[np.argmin(np.abs(far - frr))]

cos_eer = cal_eer(cos_far, cos_frr)
print("Cosine EER = %.2f%%" % (cos_eer * 100))

#decidability index(DI)
def cal_di(garr, iarr):
    u_g = garr.mean()
    u_i = iarr.mean()
    sigma_g = garr.std()
    sigma_i = iarr.std()
    return abs(u_g - u_i) / np.sqrt((sigma_g**2 + sigma_i**2) / 2)

cos_di = cal_di(cos_garr, cos_iarr)
print("Cosine Decidability Index = %.2f" % cos_di)

# --------------------
# CMC (Cumulative Match Curve)
# --------------------
correct = []
for i in range(len(probe_labels)):
    sims = cos_mat[i]
    sorted_indices = np.argsort(sims)[::-1]
    correct_index = np.where(gallery_labels[sorted_indices] == probe_labels[i])[0][0]
    correct.append(correct_index)

cmc_curve = np.zeros(len(gallery_labels))
for r in correct:
    cmc_curve[r:] += 1
cmc_curve /= len(correct)

plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(cmc_curve)+1), cmc_curve, marker='o')
plt.title('CMC Curve (CurricularFace)')
plt.xlabel('Rank')
plt.ylabel('Recognition Rate')
plt.grid()
plt.show()

# lets say registered people are 180
REG_NUM = 180
IMAGES_PER_ID = 1  # 1 image per gallary

# probe_data have mixed registered and unregistered identities
# we cannot assume that the first 180 identities are registered
registered_names = set(gallery_labels[:REG_NUM])
reg_idx = [i for i, name in enumerate(probe_labels) if name in registered_names]
unreg_idx = [i for i, name in enumerate(probe_labels) if name not in registered_names]
reg_dists = cos_mat[reg_idx, :REG_NUM]
unreg_dists = cos_mat[unreg_idx, :REG_NUM]

# set threshold range
t_min = min(reg_dists.min(), unreg_dists.min())
t_max = max(reg_dists.max(), unreg_dists.max())
resolu = 1000

cos_fpir = np.empty(resolu)
cos_fnir = np.empty(resolu)

print("[DEBUG] reg_dists.shape =", reg_dists.shape)
print("[DEBUG] unreg_dists.shape =", unreg_dists.shape)

for i, t in enumerate(np.linspace(t_min, t_max, resolu)):
    cos_fnir[i] = np.sum(np.sum(reg_dists >= t, axis=1) == 0) / reg_dists.shape[0]
    cos_fpir[i] = np.sum(np.sum(unreg_dists >= t, axis=1) >= 1) / unreg_dists.shape[0]

# Plot
plt.title("Cosine FPIR vs FNIR (CurricularFace)")
plt.xlabel("FPIR")
plt.ylabel("FNIR")
plt.plot(cos_fpir, cos_fnir, label="Cosine")
plt.grid()
plt.legend()
plt.show()

np.savetxt("curricular_garr.csv", cos_garr, delimiter=",")
np.savetxt("curricular_iarr.csv", cos_iarr, delimiter=",")
