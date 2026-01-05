from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "data/images/train",
    target_size=(224,224),
    batch_size=8,
    class_mode="binary"
)

print("Class mapping:", train_data.class_indices)
print("Total training images:", train_data.samples)
