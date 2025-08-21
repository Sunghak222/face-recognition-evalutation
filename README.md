# Face Recognition with ArcFace and CurricularFace

This project evaluates face recognition performance using two state-of-the-art face recognition models: ArcFace and CurricularFace with the LFW dataset.  
Metrics such as ROC, CMC, FAR/FRR, and FPIR/FNIR are used for evaluation.

GenAI and external websites helped us for programming
---

## Folder Structure
```plaintext
FaceRecognition/ 
├── ArcFace/ 
│   ├── arcface_embed.py
│   ├── arcface_eval.py
│   └── arcface_eval_300x4.py
│
├── CurricularFace/ 
|   └──inference/
│       ├── curricular_embed.py
│       ├── curricular_eval.py
│       └── curricular_eval_300x4.py 
│
├── arcface_sess1_data.csv
├── arcface_sess2_data.csv
├── arcface_sess3_data.csv
├── arcface_sess1_label.csv
├── arcface_sess2_label.csv
├── arcface_sess3_label.csv
│
├── arcface2_sess1_data.csv   
├── arcface2_sess2_data.csv
├── arcface2_sess3_data.csv
├── arcface2_sess4_data.csv
├── arcface2_sess1_label.csv
├── arcface2_sess2_label.csv
├── arcface2_sess3_label.csv
├── arcface2_sess4_label.csv
│
├── curricular_sess1_data.csv
├── curricular_sess2_data.csv
├── curricular_sess3_data.csv
├── curricular_sess1_label.csv
├── curricular_sess2_label.csv
├── curricular_sess3_label.csv
│
├── curricular2_sess1_data.csv      
├── curricular2_sess2_data.csv
├── curricular2_sess3_data.csv
├── curricular2_sess4_data.csv
├── curricular2_sess1_label.csv
├── curricular2_sess2_label.csv
├── curricular2_sess3_label.csv
├── curricular2_sess4_label.csv
│
├── compare2_eval.py                 
│
└── dataset/ 
    └── lfw-deepfunneled/
```
---

## Requirements

- Python 3.10+
- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `opencv-python`
- `tqdm`
- `scikit-learn`
- `insightface`  
   https://github.com/deepinsight/insightface

---

### Installation


conda create -n arcface python=3.10 -y
conda activate arcface
pip install -r requirements.txt

### run
Set DATASET_PATH and MODEL_PATH for arcface_embed.py, arcface2_embed.py, curricular_embed.py, and curricular2_embed.py.
DATASET_PATH: the path where your ifw dataset located. It will be ../FaceRecognition/dataset/lfw-deepfunneled/lfw-deepfunneled
MODEL_PATH: the path where your backbone model of curricularface located.
            It will be /CurricularFace-master/backbone/CurricularFace_Backbone.pth
# Step 1: Extract face embeddings
python arcface_embed.py
python curricular_embed.py

# Step 2: Evaluate each model separately
python arcface_eval.py
python curricular_eval.py

# Step 3: Compare ArcFace and CurricularFace
python compare_eval.py         # for 200 × 3 setting
python compare_eval_300x4.py   # for 300 × 4 setting
