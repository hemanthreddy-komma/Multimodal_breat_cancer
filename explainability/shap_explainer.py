import shap, numpy as np, pandas as pd
from tensorflow.keras.models import load_model

model = load_model("multimodal_model.keras")

df = pd.read_csv("data/clinical_data.csv")
df = df[['subject_age','density','mass shape']].fillna(0).values[:100]

explainer = shap.KernelExplainer(lambda x: model.predict([np.zeros((len(x),224,224,3)), x]), df)
shap_values = explainer.shap_values(df[:10])
shap.summary_plot(shap_values, df[:10])
