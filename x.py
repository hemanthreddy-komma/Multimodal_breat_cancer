import os, pandas as pd

CSV_PATH   = "data/clinical_data.csv"
SOURCE_DIR = r"C:\Users\mahit\Downloads\4-2\major project\Breast Cancer\Dataset\mammobench\Mammo_Bench_v2\Preprocessed_Dataset"

df = pd.read_csv(CSV_PATH)

df['img_name'] = df['preprocessed_image_path'].astype(str).apply(lambda x: os.path.basename(x))
df['final_name'] = df['source_dataset'].astype(str) + "_" + df['img_name']

print("\nCSV sample final names:")
print(df['final_name'].head(10).tolist())

disk_files=[]
for root,_,files in os.walk(SOURCE_DIR):
    for f in files:
        if f.lower().endswith((".jpg",".png",".jpeg")):
            disk_files.append(f)

print("\nDISK sample filenames:")
print(disk_files[:10])

matches = set(df['final_name']) & set(disk_files)
print("\nTOTAL MATCHES FOUND:", len(matches))
