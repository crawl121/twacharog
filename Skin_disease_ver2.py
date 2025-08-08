from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Define dataset paths
train_dir = r"C:\skin-disease-dataset\train_set"
test_dir = r"C:\skin-disease-dataset\test_set"

# Ensure dataset paths exist
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Testing directory not found: {test_dir}")

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data augmentation & preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Get class indices
class_indices = train_generator.class_indices
classes = list(class_indices.keys())

# Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model layers

# Define the new model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save trained model
model.save("skin_disease_detector.h5")
print("Model training complete. Saved as 'skin_disease_detector.h5'.")

# Load trained model
model = tf.keras.models.load_model("skin_disease_detector.h5")

# Create uploads folder
UPLOAD_FOLDER = r"C:\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to predict disease with confidence
def predict_disease(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch size
    img_array /= 255.0  # Normalize

    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100  # Get highest probability
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class, confidence

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", result="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", result="No file selected")

        # Save uploaded file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Predict disease
        result, confidence = predict_disease(file_path)

        return render_template("index.html", result=f"{result} (Confidence: {confidence:.2f}%)", image_path=file.filename)

    return render_template("index.html", result=None)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)


