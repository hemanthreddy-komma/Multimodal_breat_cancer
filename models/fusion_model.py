import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models

def build_fusion_model(img_size=(224, 224), clinical_features=3):
    """Multimodal Breast Cancer Classifier with Cross-Modal Attention"""

    # ================= IMAGE BRANCH =================
    img_input = layers.Input(shape=(*img_size, 3), name="image_input")

    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_tensor=img_input
    )
    base_model.trainable = False   # fine-tune later

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # ================= CLINICAL BRANCH =================
    clin_input = layers.Input(shape=(clinical_features,), name="clinical_input")

    y = layers.Dense(64, activation="relu")(clin_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(128, activation="relu")(y)

    # ================= CROSS-MODAL ATTENTION =================
    x_proj = layers.Dense(128, activation="relu")(x)
    y_proj = layers.Dense(128, activation="relu")(y)

    fusion_context = layers.Concatenate()([x_proj, y_proj])
    attention = layers.Dense(128, activation="sigmoid")(fusion_context)

    img_attended  = layers.multiply([x_proj, attention])
    clin_attended = layers.multiply([y_proj, attention])

    combined = layers.Add()([img_attended, clin_attended])

    # ================= CLASSIFIER =================
    z = layers.Dense(256, activation="relu")(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.5)(z)

    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(0.4)(z)

    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dropout(0.3)(z)

    output = layers.Dense(1, activation="sigmoid")(z)

    model = models.Model(
        inputs=[img_input, clin_input],
        outputs=output,
        name="multimodal_breast_cancer_classifier"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model
