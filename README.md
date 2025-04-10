# Face Recognition with ArcFace and CurricularFace

This project evaluates face recognition performance using two state-of-the-art deep face recognition models: ArcFace and CurricularFace, with the LFW (Labeled Faces in the Wild) dataset.  
Metrics such as ROC, CMC, FAR/FRR, and FPIR/FNIR are used for evaluation.

---

## Folder Structure
FaceRecognition/ 
    ├── ArcFace/ 
    │ 
    ├── arcface_embed.py 
    │ 
    ├── arcface_eval.py 
    │ 
    └── arcface_sess{1,2,3}_data.csv / label.csv 
    ├── CurricularFace/ 
    │ 
    ├── curricular_embed.py 
    │ 
    ├── curricular_eval.py 
    │ 
    └── curricular_sess{1,2,3}_data.csv / label.csv 
    └── dataset/ 
        └── lfw-deepfunneled/
---

## Requirements

- Python 3.10+
- torch
- torchvision
- numpy
- pandas
- seaborn
- matplotlib
- opencv-python
- insightface
- [InsightFace](https://github.com/deepinsight/insightface)
---

Install with:

```bash
pip install -r requirements.txt

Data Setup
The dataset used is LFW - Labeled Faces in the Wild, specifically the deep funneled version.

200 identities

3 images per identity

Session 1: gallery

Session 2 & 3: probe

## Usage

### 1. **Generate embeddings**

#### ArcFace
cd ArcFace
python arcface_embed.py
python arcface_eval.py

cd CurricularFace
python curricular_embed.py
python curricular_eval.py

python compare_eval.py

Face detection is done with InsightFace (model: buffalo_l)

All identities and embeddings are sorted by name before saving, to ensure alignment across sessions

Normalization is applied after embedding extraction