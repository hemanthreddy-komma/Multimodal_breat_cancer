import os, cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model

model = load_model("multimodal_model.keras")

df = pd.read_csv("data/clinical_data.csv")

df['label'] = df['classification'].astype(str).str.lower()
df = df[df['label'].isin(['benign','malignant'])]
df['label_num'] = df['label'].map({'benign':0,'malignant':1})

density_map = {'A':1,'B':2,'C':3,'D':4}
df['density'] = df['density'].map(density_map)

df['subject_age'] = pd.to_numeric(df['subject_age'], errors='coerce')
df[['subject_age','density']] = df[['subject_age','density']].fillna(df[['subject_age','density']].median())

df['mass shape'] = df['mass shape'].astype(str).fillna("unknown")
df['mass shape'] = LabelEncoder().fit_transform(df['mass shape'])

paths, labels, clin_feats = [], [], []

for cls in ['benign','malignant']:
    folder = f"data/images/test/{cls}"
    for img in os.listdir(folder):
        row = df[df['preprocessed_image_path'].str.contains(img)]
        if not row.empty:
            r = row.iloc[0]
            paths.append(os.path.join(folder,img))
            labels.append(r['label_num'])
            clin_feats.append([r['subject_age'], r['density'], r['mass shape']])

X = []
for p in paths:
    im = cv2.resize(cv2.imread(p),(224,224))/255.0
    X.append(im)

X = np.array(X, dtype=np.float32)
clinical = np.array(clin_feats, dtype=np.float32)
Y = np.array(labels)

preds = model.predict([X, clinical])
pred_labels = (preds > 0.5).astype(int)

print(classification_report(Y, pred_labels))
print("Confusion Matrix:\n", confusion_matrix(Y, pred_labels))
print("ROC-AUC:", roc_auc_score(Y, preds))
