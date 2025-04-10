import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict

#path
GALLERY_PATH = "arcface_sess1_data.csv"
PROBE_PATH1 = "arcface_sess2_data.csv"
PROBE_PATH2 = "arcface_sess3_data.csv"
LABEL_GALLERY = "arcface_sess1_label.csv"
LABEL_PROBE1 = "arcface_sess2_label.csv"
LABEL_PROBE2 = "arcface_sess3_label.csv"

#upload gallary and probe
gallery_data = np.genfromtxt(GALLERY_PATH, delimiter=',')
probe_data1 = np.genfromtxt(PROBE_PATH1, delimiter=',')
probe_data2 = np.genfromtxt(PROBE_PATH2, delimiter=',')
probe_data = np.concatenate((probe_data1, probe_data2), axis=0)

gallery_labels = pd.read_csv(LABEL_GALLERY)["name"].values
probe_labels1 = pd.read_csv(LABEL_PROBE1)["name"].values
probe_labels2 = pd.read_csv(LABEL_PROBE2)["name"].values
probe_labels = np.concatenate((probe_labels1, probe_labels2), axis=0)

print("gallery data's shape is", gallery_data.shape)
print("probe data's shape is", gallery_data.shape)

# the feature vector's dimension, here is 144
DIM = gallery_data.shape[1]

# subject number
SUB_NUM = gallery_data.shape[0]

def cal_cosine(vec1, vec2):
    return np.sum(vec1 * vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def cal_dist(arr1, arr2):
    dist_mat = np.empty((arr1.shape[0], arr2.shape[0]), dtype=np.float32)
    for i in range(arr1.shape[0]):
        for j in range(arr2.shape[0]):
            dist_mat[i, j] = cal_cosine(arr1[i], arr2[j])
    return dist_mat

#distinguish genuine and imposter
cos_garr = []
cos_iarr = []

cos_mat = cal_dist(probe_data, gallery_data)

for i in range(len(probe_labels)):
    for j in range(len(gallery_labels)):
        sim = cos_mat[i, j]
        if probe_labels[i] == gallery_labels[j]:
            cos_garr.append(sim)  # if same, genuine
        else:
            cos_iarr.append(sim)  # else, imposter

cos_garr = np.array(cos_garr)
cos_iarr = np.array(cos_iarr)

plt.title("Cosine Similarity distribution (Arcface)")
plt.xlabel("Cosine Similarity")
plt.grid()
sns.histplot(cos_garr, kde=True, stat="density", label="Genuine")
sns.histplot(cos_iarr, kde=True, stat="density", label="Imposter")
plt.legend()
plt.show()

def cal_far_frr(garr, iarr, resolu=1000):
    d_max = max(garr.max(), iarr.max())
    d_min = min(garr.min(), iarr.min())
    far = np.empty(resolu)
    frr = np.empty(resolu)
    for i, d in enumerate(np.linspace(d_min, d_max, resolu)):
        # For cosine similarity: higher the better
        far[i] = np.sum(iarr > d) / iarr.size
        frr[i] = np.sum(garr < d) / garr.size
    return far, frr

resolu = 1000

cos_far, cos_frr = cal_far_frr(cos_garr, cos_iarr, resolu)

#FAR&FRR graph
#x axis = threshold
plt.title("Cosine distance FAR & FRR (Arcface)")
plt.xlim([0, resolu])
plt.ylim([0, 1])
plt.plot(np.arange(resolu), cos_far, label="FAR")
plt.plot(np.arange(resolu), cos_frr, label="FRR")
plt.grid()
plt.legend()
plt.show()

#ROC curve
plt.title("Cosine Distance ROC Curve (Arcface)")
plt.xlabel("False Accept Rate")
plt.ylabel("Genuine Accept Rate")
# Here we use log scale for x-axis rather than normal scale
plt.xscale('log')
plt.xlim([1e-5, 1])
plt.ylim([0.9, 1.0])
plt.plot(cos_far, 1 - cos_frr, label="Cosine")
plt.legend()
plt.grid()
plt.show()

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

# #CMC curve
# cos_mat = cal_dist(probe_data, gallery_data)

# #the number of subject
# SUB_NUM = 200

# #probe pictures per subject
# PROBE_PER_SUB = 2 

# mat = cos_mat.reshape(SUB_NUM, PROBE_PER_SUB, SUB_NUM)
# rank_arr = np.zeros(SUB_NUM, dtype=np.int32)
# print("[DEBUG] gallery_labels[:5]:", gallery_labels[:5])
# print("[DEBUG] probe_labels[:10]:", probe_labels[:10])

# for sub_i in range(mat.shape[0]):
#     for img_i in range(mat.shape[1]):
#         #bigger the better
#         rank = np.sum(mat[sub_i, img_i] > mat[sub_i, img_i, sub_i])
#         rank_arr[rank:] += 1
#         #rank_arr[np.sum(mat[sub_i, img_i] > mat[sub_i, img_i, sub_i])] += 1

# cos_cmc = np.cumsum(rank_arr) / (SUB_NUM * PROBE_PER_SUB)

# plt.plot(range(1, 11), cos_cmc[:10], label="Cosine")
# plt.title("CMC Curve (Arcface)")
# plt.xlabel("Rank")
# plt.ylabel("Recognition Accuracy")
# plt.xlim((1, 10))
# plt.ylim((.1, 1))
# plt.grid()
# plt.legend()
# plt.show()

correct_ranks = []
for i in range(len(probe_labels)):
    sims = cos_mat[i]
    sorted_indices = np.argsort(sims)[::-1]
    
    correct_gallery_indices = np.where(gallery_labels == probe_labels[i])[0]
    if len(correct_gallery_indices) == 0:
        continue

    ranks = [np.where(sorted_indices == idx)[0][0] for idx in correct_gallery_indices]
    correct_ranks.append(min(ranks))

# CMC
cmc_curve = np.zeros(len(gallery_labels))
for r in correct_ranks:
    cmc_curve[r:] += 1
cmc_curve /= len(correct_ranks)
# Plot
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, 11), cmc_curve[:10], marker='o', label="Cosine")
plt.title("CMC Curve (Arcface)")
plt.xlabel("Rank")
plt.ylabel("Recognition Accuracy")
plt.grid()
plt.legend()
plt.xlim((1, 10))
plt.ylim((0.9, 1)) 
plt.show()



# FPIR & FNIR (ArcFace)
REG_NUM = 180  
IMG_PER_SUB = 2  # gallary image per register

sorted_names = sorted(set(gallery_labels))
gallery_order = np.argsort([sorted_names.index(name) for name in gallery_labels])
probe_order = np.argsort([sorted_names.index(name) for name in probe_labels if name in sorted_names])

gallery_data = gallery_data[gallery_order]
gallery_labels = gallery_labels[gallery_order]
probe_data = probe_data[probe_order]
probe_labels = probe_labels[probe_order]

cos_mat = cal_dist(probe_data, gallery_data)

mat = cos_mat[:, :REG_NUM]  # shape: (400,180)
reg_dists = mat[:REG_NUM * IMG_PER_SUB]      
unreg_dists = mat[REG_NUM * IMG_PER_SUB:]

cos_fpir = np.empty(resolu)
cos_fnir = np.empty(resolu)
t_min = mat.min()
t_max = mat.max()

for i, t in enumerate(np.linspace(t_min, t_max, resolu)):
    # FNIR
    cos_fnir[i] = np.sum(np.sum(reg_dists >= t, axis=1) == 0) / reg_dists.shape[0]
    # FPIR
    cos_fpir[i] = np.sum(np.sum(unreg_dists >= t, axis=1) >= 1) / unreg_dists.shape[0]

plt.title("Cosine FPIR vs FNIR (ArcFace)")
plt.xlabel("FPIR")
plt.ylabel("FNIR")
plt.plot(cos_fpir, cos_fnir, label="Cosine")
plt.grid()
plt.legend()
plt.show()

np.savetxt("arcface_garr.csv", cos_garr, delimiter=",")
np.savetxt("arcface_iarr.csv", cos_iarr, delimiter=",")

# # divide registers and unregisters in cosine similarity matrix
# mat = cos_mat[:, :REG_NUM]  # shape = (400,180)
# reg_dists = mat[:REG_NUM * IMG_PER_SUB]       # 180 * 2
# unreg_dists = mat[REG_NUM * IMG_PER_SUB:]     # 20 * 2

# # Threshold
# t_min = mat.min()
# t_max = mat.max()
# resolu = 1000
# cos_fpir = np.empty(resolu)
# cos_fnir = np.empty(resolu)

# # FPIR / FNIR
# for i, t in enumerate(np.linspace(t_min, t_max, resolu)):
#     # when any one of registers does not match(FNIR)
#     cos_fnir[i] = np.sum(np.sum(reg_dists >= t, axis=1) == 0) / reg_dists.shape[0]
#     # when any one of unregisters not match(FPIR)
#     cos_fpir[i] = np.sum(np.sum(unreg_dists >= t, axis=1) >= 1) / unreg_dists.shape[0]


# # --------------------
# # Plot
# # --------------------
# plt.title("Cosine FPIR vs FNIR (ArcFace)")
# plt.xlabel("FPIR")
# plt.ylabel("FNIR")
# plt.plot(cos_fpir, cos_fnir, label="Cosine")
# plt.grid()
# plt.legend()
# plt.show()
