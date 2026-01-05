from tensorflow.keras.layers import Dense, Input

def build_clinical_model():
    input_layer = Input(shape=(3,))
    x = Dense(64, activation="relu")(input_layer)
    x = Dense(32, activation="relu")(x)
    return input_layer, Dense(16, activation="relu")(x)
