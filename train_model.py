import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

DATASET_PATH = "dataset"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

# Load training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Load validation dataset
validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Get class names
class_names = train_data.class_names

print("Classes:")
print(class_names)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

validation_data = validation_data.cache().prefetch(
    buffer_size=AUTOTUNE
)

# CNN Model
model = models.Sequential([

    layers.Rescaling(1. / 255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model details
model.summary()

# Train model
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# Save model
model.save("plant_disease_model.keras")

# Plot accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Plant Disease Detection Accuracy")
plt.legend()

plt.show()

print("Model training completed!")
print("Model saved as plant_disease_model.keras")