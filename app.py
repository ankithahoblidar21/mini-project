import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from tensorflow.keras import layers, models
from PIL import Image

# ==============================
# PLANT DISEASE DETECTION SYSTEM
# ==============================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱"
)

st.title("🌱 Plant Disease Detection Using Machine Learning")
st.write("Upload a plant leaf image to detect its disease.")

# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "dataset"
MODEL_PATH = "plant_disease_model.keras"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5

# ==============================
# TRAIN MODEL
# ==============================

def train_model():

    if not os.path.exists(DATASET_PATH):
        st.error("Dataset folder not found!")
        st.info("Create a folder named 'dataset' and add disease folders with images.")
        return None, None

    train_data = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    validation_data = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = train_data.class_names

    model = models.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),

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

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    with st.spinner("Training CNN model..."):
        model.fit(
            train_data,
            validation_data=validation_data,
            epochs=EPOCHS
        )

    model.save(MODEL_PATH)

    return model, class_names


# ==============================
# LOAD OR TRAIN MODEL
# ==============================

if os.path.exists(MODEL_PATH):

    model = tf.keras.models.load_model(MODEL_PATH)

    # Get class names from dataset
    if os.path.exists(DATASET_PATH):
        class_names = sorted([
            folder for folder in os.listdir(DATASET_PATH)
            if os.path.isdir(os.path.join(DATASET_PATH, folder))
        ])
    else:
        class_names = []

else:

    st.warning("Model not found. Please train the model first.")

    if st.button("Train CNN Model"):

        model, class_names = train_model()

        if model is not None:
            st.success("Model trained successfully!")

    else:
        model = None
        class_names = []


# ==============================
# IMAGE UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "Upload Plant Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    if st.button("🔍 Detect Disease"):

        # Convert PIL image to NumPy
        img = np.array(image)

        # OpenCV processing
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalize image
        img = img / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Prediction
        prediction = model.predict(img)

        predicted_index = np.argmax(prediction[0])

        confidence = prediction[0][predicted_index] * 100

        if predicted_index < len(class_names):
            disease = class_names[predicted_index]
        else:
            disease = "Unknown"

        # ==============================
        # RESULT
        # ==============================

        st.success("🌿 Disease Detection Result")

        st.write("### Detected Disease:")
        st.write(disease)

        st.write(
            f"### Confidence: {confidence:.2f}%"
        )

        # ==============================
        # ALL PREDICTIONS
        # ==============================

        st.write("### Prediction Probabilities")

        if len(class_names) == len(prediction[0]):

            results = pd.DataFrame({
                "Disease": class_names,
                "Probability (%)":
                    prediction[0] * 100
            })

            results = results.sort_values(
                "Probability (%)",
                ascending=False
            )

            st.dataframe(
                results,
                use_container_width=True
            )

            # ==============================
            # GRAPH
            # ==============================

            fig, ax = plt.subplots()

            ax.bar(
                results["Disease"],
                results["Probability (%)"]
            )

            ax.set_xlabel("Disease")
            ax.set_ylabel("Probability (%)")
            ax.set_title("Disease Prediction")

            plt.xticks(rotation=45, ha="right")

            st.pyplot(fig)