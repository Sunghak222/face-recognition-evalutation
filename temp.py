import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------
# Config
# --------------------
ALGORITHMS = {
    "ArcFace": {
        "sess1_data": "arcface2_sess1_data.csv",
        "sess2_data": "arcface2_sess2_data.csv",
        "sess3_data": "arcface2_sess3_data.csv",
        "sess4_data": "arcface2_sess4_data.csv",
        "sess1_label": "arcface2_sess1_label.csv",
        "sess2_label": "arcface2_sess2_label.csv",
        "sess3_label": "arcface2_sess3_label.csv",
        "sess4_label": "arcface2_sess4_label.csv",
    },
    "CurricularFace": {
        "sess1_data": "curricular2_sess1_data.csv",
        "sess2_data": "curricular2_sess2_data.csv",
        "sess3_data": "curricular2_sess3_data.csv",
        "sess4_data": "curricular2_sess4_data.csv",
        "sess1_label": "curricular2_sess1_label.csv",
        "sess2_label": "curricular2_sess2_label.csv",
        "sess3_label": "curricular2_sess3_label.csv",
        "sess4_label": "curricular2_sess4_label.csv",
    }
}

# --------------------
# Utilities
# --------------------
def cal_cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cal_dist(mat1, mat2):
    dist = np.zeros((mat1.shape[0], mat2.shape[0]), dtype=np.float32)
    for i in range(mat1.shape[0]):
        for j in range(mat2.shape[0]):
            dist[i, j] = cal_cosine(mat1[i], mat2[j])
    return dist

def compute_metrics(gallery_data, gallery_labels, probe_data, probe_labels):
    dist_mat = cal_dist(probe_data, gallery_data)
    cos_garr, cos_iarr = [], []
    for i in range(len(probe_labels)):
        for j in range(len(gallery_labels)):
            sim = dist_mat[i, j]
            if probe_labels[i] == gallery_labels[j]:
                cos_garr.append(sim)
            else:
                cos_iarr.append(sim)
    return np.array(cos_garr), np.array(cos_iarr), dist_mat

def plot_roc_curves(genuine_dict, imposter_dict):
    plt.figure(figsize=(6, 6))
    for key in genuine_dict:
        g, i = genuine_dict[key], imposter_dict[key]
        far, frr = [], []
        thresholds = np.linspace(min(i.min(), g.min()), max(i.max(), g.max()), 1000)
        for t in thresholds:
            far.append(np.sum(i > t) / len(i))
            frr.append(np.sum(g < t) / len(g))
        plt.plot(far, 1 - np.array(frr), label=key)
    plt.xlabel("False Accept Rate")
    plt.ylabel("Genuine Accept Rate")
    plt.title("ROC Curve (300x4)")
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.show()

def plot_cmc(dist_mat, gallery_labels, probe_labels, label):
    correct = []
    for i in range(len(probe_labels)):
        sims = dist_mat[i]
        sorted_indices = np.argsort(sims)[::-1]
        correct_index = np.where(gallery_labels[sorted_indices] == probe_labels[i])[0][0]
        correct.append(correct_index)
    cmc_curve = np.zeros(len(gallery_labels))
    for r in correct:
        cmc_curve[r:] += 1
    cmc_curve /= len(correct)
    plt.plot(np.arange(1, 11), cmc_curve[:10], marker='o', label=label)

# --------------------
# Evaluation
# --------------------
genuine_dict = {}
imposter_dict = {}

for algo_name, files in ALGORITHMS.items():
    gallery_data = np.genfromtxt(files["sess1_data"], delimiter=',')
    probe_data = np.concatenate([
        np.genfromtxt(files["sess2_data"], delimiter=','),
        np.genfromtxt(files["sess3_data"], delimiter=','),
        np.genfromtxt(files["sess4_data"], delimiter=',')
    ], axis=0)

    gallery_labels = pd.read_csv(files["sess1_label"])["name"].values
    probe_labels = np.concatenate([
        pd.read_csv(files["sess2_label"])["name"].values,
        pd.read_csv(files["sess3_label"])["name"].values,
        pd.read_csv(files["sess4_label"])["name"].values
    ], axis=0)

    genuine, imposter, dist_mat = compute_metrics(gallery_data, gallery_labels, probe_data, probe_labels)
    genuine_dict[algo_name] = genuine
    imposter_dict[algo_name] = imposter

    plot_cmc(dist_mat, gallery_labels, probe_labels, label=algo_name)

plt.title("CMC Curve (300x4)")
plt.xlabel("Rank")
plt.ylabel("Recognition Rate")
plt.grid()
plt.legend()
plt.xlim((1, 10))
plt.ylim((0.9, 1))
plt.show()

plot_roc_curves(genuine_dict, imposter_dict)
