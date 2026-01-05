import os, shutil, pandas as pd
from sklearn.model_selection import train_test_split

SOURCE_DIR = r"C:\Users\mahit\Downloads\4-2\major project\Breast Cancer\Dataset\mammobench\Mammo_Bench_v2\Preprocessed_Dataset"
CSV_PATH   = "data/clinical_data.csv"
TARGET_DIR = "data/images"

os.makedirs(TARGET_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df['label'] = df['classification'].astype(str).str.lower()
df = df[df['label'].isin(['benign','malignant'])]

image_map = {}
for _, row in df.iterrows():
    image_map[os.path.basename(row['preprocessed_image_path'])] = row['label']

print("Total labeled images in CSV:", len(image_map))

files=[]
for root,dirs,fns in os.walk(SOURCE_DIR):
    for f in fns:
        if f.lower().endswith((".jpg",".png",".jpeg")):
            files.append(os.path.join(root,f))

print("Total images found recursively:", len(files))

records=[]
for f in files:
    fname=os.path.basename(f)
    if fname in image_map:
        records.append((f,image_map[fname]))

print("Matched labeled images:", len(records))

train,test = train_test_split(records,test_size=0.2,stratify=[r[1] for r in records])

for split in ["train","test"]:
    for cls in ["benign","malignant"]:
        os.makedirs(os.path.join(TARGET_DIR,split,cls),exist_ok=True)

def copy_data(data,split):
    for src,label in data:
        dst=os.path.join(TARGET_DIR,split,label,os.path.basename(src))
        shutil.copy(src,dst)

copy_data(train,"train")
copy_data(test,"test")

print("Dataset split complete!")
print("Train:",len(train))
print("Test :",len(test))
