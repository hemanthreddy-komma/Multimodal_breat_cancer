import os, cv2, numpy as np, pandas as pd, tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from models.fusion_model import build_fusion_model

IMG_SIZE = (224,224)
BATCH_SIZE = 16
EPOCHS = 12

df = pd.read_csv("data/clinical_data.csv")

df['label'] = df['classification'].astype(str).str.lower()
df = df[df['label'].isin(['benign','malignant'])]
df['label_num'] = df['label'].map({'benign':0,'malignant':1})

df['image_name'] = df['preprocessed_image_path'].apply(lambda x: os.path.basename(str(x)))

density_map = {'A':1,'B':2,'C':3,'D':4}
df['density'] = df['density'].astype(str).str.upper().map(density_map)

df['subject_age'] = pd.to_numeric(df['subject_age'], errors='coerce')
df[['subject_age','density']] = df[['subject_age','density']].fillna(df[['subject_age','density']].median())

df['mass shape'] = LabelEncoder().fit_transform(df['mass shape'].astype(str))

# ================= BUILD DATAFRAME =================
def build_dataframe(folder):
    records = []
    for cls in ['benign','malignant']:
        class_dir = os.path.join(folder,cls)
        for img in os.listdir(class_dir):
            if not img.lower().endswith(('.jpg','.jpeg','.png')): continue
            row = df[df['image_name'] == img]
            if not row.empty:
                r = row.iloc[0]
                records.append([
                    os.path.join(class_dir,img),
                    r['subject_age'], r['density'], r['mass shape'], r['label_num']
                ])
    return pd.DataFrame(records, columns=['path','age','density','shape','label'])

train_df = build_dataframe("data/images/train")
test_df  = build_dataframe("data/images/test")

print("Train samples:",len(train_df)," Test samples:",len(test_df))

# ================= CLASS WEIGHTS =================
class_weights = dict(enumerate(
    compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
))

# ================= DATASET =================
class MultiModalDataset(tf.keras.utils.Sequence):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return int(np.ceil(len(self.df) / BATCH_SIZE))

    def __getitem__(self, idx):
        batch = self.df.iloc[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
        imgs = []
        for p in batch['path']:
            img = cv2.imread(p)
            img = cv2.resize(img, IMG_SIZE)/255.0
            imgs.append(img)
        imgs = np.array(imgs, np.float32)
        clin = batch[['age','density','shape']].values.astype(np.float32)
        labels = batch['label'].values.astype(np.float32)
        return (imgs, clin), labels

train_ds = MultiModalDataset(train_df)
test_ds  = MultiModalDataset(test_df)

# ================= TRAIN =================
model = build_fusion_model(clinical_features=3)
model.fit(train_ds,
          validation_data=test_ds,
          epochs=EPOCHS,
          class_weight=class_weights)

model.save("multimodal_model.h5")
print("MODEL SAVED")
