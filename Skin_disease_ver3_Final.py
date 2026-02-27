from flask import Flask, request, render_template, send_from_directory, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import json
import argparse
import functools
import urllib.request
import urllib.error

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
MODEL_PATH = os.path.join(BASE_DIR, "skin_disease_detector.keras")
MODEL_PATH_LEGACY = os.path.join(BASE_DIR, "skin_disease_detector.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define dataset paths (update if your dataset location changes)
train_dir = os.path.join(BASE_DIR, "skin-disease-dataset", "train_set")
test_dir = os.path.join(BASE_DIR, "skin-disease-dataset", "test_set")

# Model parameters
IMG_SIZE = (224, 224)  # MobileNetV2 requires 224x224 images
BATCH_SIZE = 32

def ensure_dataset_paths():
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Testing directory not found: {test_dir}")


def save_class_names(class_names):
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)


def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        return None
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def train_and_save_model():
    ensure_dataset_paths()

    # Data augmentation & preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load dataset
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    # Get class indices
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())
    save_class_names(class_names)

    # Load MobileNetV2 (Pretrained Model)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze base model initially
    base_model.trainable = False

    # Define the model (Functional API for better serialization compatibility)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_generator, epochs=10, validation_data=test_generator)

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Recompile with a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_generator, epochs=10, validation_data=test_generator)

    model.save(MODEL_PATH)
    print(f"Model training complete. Saved as '{MODEL_PATH}'.")
    return model, class_names


def load_model_and_classes():
    class_names = load_class_names()

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model, class_names

    if os.path.exists(MODEL_PATH_LEGACY):
        try:
            model = tf.keras.models.load_model(MODEL_PATH_LEGACY, compile=False)
            return model, class_names
        except Exception as e:
            print(f"Failed to load legacy .h5 model: {e}")
            return None, class_names

    return None, class_names


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model, classes = load_model_and_classes()


def predict_disease(img_path):  # Function to predict disease with confidence
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch size
    img_array /= 255.0  # Normalize

    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100  # Get highest probability
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class, confidence


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def login_required(view_func):
    @functools.wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper


@app.route("/", methods=["GET"])
def landing():
    return render_template("landing.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            return render_template("signup.html", error="All fields are required.")

        users = load_users()
        if email in users:
            return render_template("signup.html", error="Email already registered.")

        users[email] = {
            "name": name,
            "password_hash": generate_password_hash(password),
        }
        save_users(users)
        return redirect(url_for("login"))

    return render_template("signup.html", error=None)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        users = load_users()
        user = users.get(email)
        if not user or not check_password_hash(user["password_hash"], password):
            return render_template("login.html", error="Invalid email or password.")

        session["user"] = {"email": email, "name": user["name"]}
        return redirect(url_for("app_page"))

    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("landing"))


@app.route("/app", methods=["GET", "POST"])
@login_required
def app_page():
    if model is None or classes is None:
        return render_template(
            "app.html",
            result=None,
            confidence=None,
            image_path=None,
            error="Model not found. Train the model first with --train.",
        )

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("app.html", result=None, confidence=None, image_path=None, error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("app.html", result=None, confidence=None, image_path=None, error="No file selected.")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        result, confidence = predict_disease(file_path)

        return render_template(
            "app.html",
            result=result,
            confidence=f"{confidence:.2f}",
            image_path=filename,
            error=None,
        )

    return render_template("app.html", result=None, confidence=None, image_path=None, error=None)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/ask", methods=["POST"])
@login_required
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    disease = (data.get("disease") or "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set on server."}), 500

    prompt = question
    if disease:
        prompt = f"For the skin condition '{disease}': {question}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                ]
            }
        ]
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-1.5-flash:generateContent?key="
        + api_key
    )

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = ""
        candidates = body.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                text = parts[0].get("text", "")
        if not text:
            text = "No response returned from Gemini."
        return jsonify({"answer": text})
    except urllib.error.HTTPError as e:
        return jsonify({"error": f"Gemini API error: {e.code}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skin Disease Detection App")
    parser.add_argument("--train", action="store_true", help="Train model and save .h5 + class names")
    args = parser.parse_args()

    if args.train:
        train_and_save_model()
        model, classes = load_model_and_classes()

    app.run(debug=True)

