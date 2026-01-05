import cv2, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("multimodal_model.keras")

def gradcam(img, layer_name="block14_sepconv2_act"):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model([img, np.zeros((1,3))])
        loss = preds[:,0]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0,1,2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_out), axis=-1)[0]
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    return cam

img = cv2.resize(cv2.imread("data/images/test/benign/sample.jpg"), (224,224))/255.0
heatmap = gradcam(img[np.newaxis,:,:,:])
heatmap = cv2.resize(heatmap, (224,224))
cv2.imwrite("results/gradcam_heatmap.png", heatmap*255)
print("Grad-CAM saved in results/gradcam_heatmap.png")
