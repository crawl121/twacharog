from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

app = Flask(__name__)

#Define dataset paths
train_dir = r"C:\Users\path to \train_set"
test_dir = r"C:\Users\path to \test_set"

# Ensure dataset paths exist
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Testing directory not found: {test_dir}")

#Model parameters
IMG_SIZE = (224, 224)  # MobileNetV2 requires 224x224 images
BATCH_SIZE = 32

#Data augmentation &preprocessing
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

#Load MobileNetV2 (Pretrained Model)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model initially
base_model.trainable = False  

# Define the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(classes), activation='softmax')  # Output layer with softmax
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=10, validation_data=test_generator)#train classifier only 


for layer in base_model.layers[-20:]:#unfreeze last 20 layers 
    layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=10, validation_data=test_generator)

model.save("skin_disease_detector.h5") #save the trained model
print("Model training complete. Saved as 'skin_disease_detector.h5'.")

model = tf.keras.models.load_model("skin_disease_detector.h5")#load model 

UPLOAD_FOLDER = r"C:\Users\path to twacharog \uploads" #create a new upload folder 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def predict_disease(img_path): #Function to predict disease with confidence
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch size
    img_array /= 255.0  # Normalize

    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100  # Get highest probability
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class, confidence

@app.route("/", methods=["GET", "POST"]) #flask route
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

