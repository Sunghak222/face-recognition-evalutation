import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURATION
# -------------------------------
ALGORITHMS = {
    "ArcFace": {
        "sess1_data": "arcface_sess1_data.csv",
        "sess2_data": "arcface_sess2_data.csv",
        "sess3_data": "arcface_sess3_data.csv",
        "sess1_label": "arcface_sess1_label.csv",
        "sess2_label": "arcface_sess2_label.csv",
        "sess3_label": "arcface_sess3_label.csv"
    },
    "CurricularFace": {
        "sess1_data": "curricular_sess1_data.csv",
        "sess2_data": "curricular_sess2_data.csv",
        "sess3_data": "curricular_sess3_data.csv",
        "sess1_label": "curricular_sess1_label.csv",
        "sess2_label": "curricular_sess2_label.csv",
        "sess3_label": "curricular_sess3_label.csv"
    }
}

RESOLU = 1000
REG_NUM = 180
IMAGES_PER_ID = 2

def cal_cosine(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def cal_dist(arr1, arr2):
    dist_mat = np.empty((arr1.shape[0], arr2.shape[0]), dtype=np.float32)
    for i in range(arr1.shape[0]):
        for j in range(arr2.shape[0]):
            dist_mat[i, j] = cal_cosine(arr1[i], arr2[j])
    return dist_mat

def cal_far_frr(garr, iarr, resolu=1000):
    d_max = max(garr.max(), iarr.max())
    d_min = min(garr.min(), iarr.min())
    far = np.empty(resolu)
    frr = np.empty(resolu)
    for i, d in enumerate(np.linspace(d_min, d_max, resolu)):
        far[i] = np.sum(iarr > d) / iarr.size
        frr[i] = np.sum(garr < d) / garr.size
    return far, frr

def cal_eer(far, frr):
    return far[np.argmin(np.abs(far - frr))]

def cal_di(garr, iarr):
    return abs(garr.mean() - iarr.mean()) / np.sqrt((garr.std()**2 + iarr.std()**2) / 2)

results = {}
all_embeddings = {}

for algo_name, files in ALGORITHMS.items():
    gallery = np.genfromtxt(files["sess1_data"], delimiter=",")
    probe1 = np.genfromtxt(files["sess2_data"], delimiter=",")
    probe2 = np.genfromtxt(files["sess3_data"], delimiter=",")
    probe = np.concatenate((probe1, probe2), axis=0)

    gallery_labels = pd.read_csv(files["sess1_label"])["name"].values
    probe_labels1 = pd.read_csv(files["sess2_label"])["name"].values
    probe_labels2 = pd.read_csv(files["sess3_label"])["name"].values
    probe_labels = np.concatenate((probe_labels1, probe_labels2), axis=0)

    sorted_names = sorted(set(gallery_labels))
    gallery_order = np.argsort([sorted_names.index(name) for name in gallery_labels])
    probe_order = np.argsort([sorted_names.index(name) for name in probe_labels if name in sorted_names])

    gallery = gallery[gallery_order]
    gallery_labels = gallery_labels[gallery_order]
    probe = probe[probe_order]
    probe_labels = probe_labels[probe_order]

    cos_mat = cal_dist(probe, gallery)

    all_embeddings[algo_name] = {
        "gallery": gallery,
        "probe": probe,
        "gallery_labels": gallery_labels,
        "probe_labels": probe_labels,
        "cos_mat": cos_mat
    }

    cos_garr, cos_iarr = [], []
    for i in range(len(probe_labels)):
        for j in range(len(gallery_labels)):
            sim = cos_mat[i, j]
            if probe_labels[i] == gallery_labels[j]:
                cos_garr.append(sim)
            else:
                cos_iarr.append(sim)
    cos_garr = np.array(cos_garr)
    cos_iarr = np.array(cos_iarr)

    far, frr = cal_far_frr(cos_garr, cos_iarr, RESOLU)

    results[algo_name] = {
        "cos_mat": cos_mat,
        "genuine": cos_garr,
        "imposter": cos_iarr,
        "far": far,
        "frr": frr,
        "eer": cal_eer(far, frr),
        "di": cal_di(cos_garr, cos_iarr)
    }

# ROC CURVE
plt.figure(figsize=(6, 6))
plt.title("ROC Curve")
plt.xlabel("False Accept Rate (FAR)")
plt.ylabel("Genuine Accept Rate (1 - FRR)")
plt.xscale("log")
plt.xlim([1e-5, 1])
plt.ylim([0.9, 1.0])
for name, res in results.items():
    plt.plot(res["far"], 1 - res["frr"], label=f"{name} (EER: {res['eer']*100:.2f}%)")
plt.grid()
plt.legend()
plt.show()

# CMC CURVE
plt.figure(figsize=(6, 4))
plt.title("CMC Curve")
plt.xlabel("Rank")
plt.ylabel("Recognition Rate")
for name, emb in all_embeddings.items():
    cos_mat = emb["cos_mat"]
    g_labels = emb["gallery_labels"]
    p_labels = emb["probe_labels"]
    correct = []
    for i in range(len(p_labels)):
        sims = cos_mat[i]
        sorted_idx = np.argsort(sims)[::-1]
        correct_rank = np.where(g_labels[sorted_idx] == p_labels[i])[0][0]
        correct.append(correct_rank)
    cmc = np.zeros(len(g_labels))
    for r in correct:
        cmc[r:] += 1
    cmc /= len(correct)
    plt.plot(np.arange(1, 11), cmc[:10], marker='o', label=name)
plt.grid()
plt.legend()
plt.show()

# FPIR vs FNIR
plt.figure(figsize=(6, 4))
plt.title("FPIR vs FNIR")
plt.xlabel("FPIR")
plt.ylabel("FNIR")
for name, emb in all_embeddings.items():
    mat = emb["cos_mat"][:, :REG_NUM]
    reg = mat[:REG_NUM * IMAGES_PER_ID]
    unreg = mat[REG_NUM * IMAGES_PER_ID:]
    fpir = np.empty(RESOLU)
    fnir = np.empty(RESOLU)
    t_min, t_max = mat.min(), mat.max()
    for i, t in enumerate(np.linspace(t_min, t_max, RESOLU)):
        fnir[i] = np.sum(np.sum(reg >= t, axis=1) == 0) / reg.shape[0]
        fpir[i] = np.sum(np.sum(unreg >= t, axis=1) >= 1) / unreg.shape[0]
    plt.plot(fpir, fnir, label=name)
plt.grid()
plt.legend()
plt.show()
