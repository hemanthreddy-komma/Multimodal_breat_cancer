from lime import lime_image
import numpy as np, cv2
from tensorflow.keras.models import load_model

model = load_model("multimodal_model.keras")

explainer = lime_image.LimeImageExplainer()

img = cv2.resize(cv2.imread("data/images/test/benign/sample.jpg"),(224,224))/255.0

exp = explainer.explain_instance(
    img.astype('double'),
    lambda x: model.predict([x, np.zeros((len(x),3))]),
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

exp.save_to_file("results/lime_output.html")
print("LIME explanation saved in results/lime_output.html")
