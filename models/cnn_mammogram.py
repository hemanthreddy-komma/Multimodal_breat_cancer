from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_image_model(input_shape=(224,224,3)):
    base = Xception(weights="imagenet", include_top=False, input_shape=input_shape)

    # 🔓 Fine-tune last 40 layers
    for layer in base.layers[:-40]:
        layer.trainable = False
    for layer in base.layers[-40:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    features = Dense(128, activation="relu")(x)

    return base.input, features
