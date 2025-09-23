import os
import cv2
import numpy as np
from flask import Flask, request, send_file, render_template
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed, LSTM, GlobalAveragePooling2D, Dropout, BatchNormalization, Dense, Input
from tensorflow.keras.applications import MobileNetV2

# ------------------ Constants ------------------
IMG_SIZE = 128
FRAME_COUNT = 15
CLASSES = ['Normal_Activities', 'Abnormal_Activities']
# ALARM_SOUND_PATH is not needed for cloud deployment

# ------------------ Build Model Architecture ------------------
def build_model():
    """Builds and returns the model architecture."""
    input_sequence = Input(shape=(FRAME_COUNT, IMG_SIZE, IMG_SIZE, 3))
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = TimeDistributed(base_model)(input_sequence)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=input_sequence, outputs=output)
    return model

# ------------------ Load Model and Handle Errors ------------------
model = build_model()
try:
    model.load_weights("models/cctv_monitoring.keras")
except FileNotFoundError:
    print("Warning: Model weights file 'cctv_monitoring.keras' not found.")
    print("The model will not be able to make predictions.")
    model = None

# ------------------ Flask App ------------------
app = Flask(__name__)

# Route for the home page (GET request)
@app.route("/")
def home():
    """Serves the video upload form."""
    return render_template("index.html")

# The playsound functionality is removed as it's not suitable for cloud deployment
# A cloud server doesn't have an audio device to play sound.
# The front-end is responsible for playing the alarm.
def process_video(input_video_path, output_video_path):
    """Processes the video and overlays the predictions."""
    if model is None:
        raise Exception("Model is not loaded. Cannot process video.")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception("Could not open input video")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    original_frame_buffer = []
    processed_frame_buffer = []

    while True:
        ret, original_frame = cap.read()
        if not ret:
            break

        original_frame_buffer.append(original_frame)
        processed_frame = cv2.resize(original_frame, (IMG_SIZE, IMG_SIZE))
        processed_frame = processed_frame[:, :, ::-1] / 255.0
        processed_frame_buffer.append(processed_frame)

        if len(processed_frame_buffer) == FRAME_COUNT:
            input_sequence = np.expand_dims(np.array(processed_frame_buffer), axis=0)
            prediction = model.predict(input_sequence, verbose=0)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = CLASSES[predicted_class_index]

            # In a deployed environment, we can't play sound on the server.
            # The client-side (frontend) should handle the alarm.
            if predicted_class_name == "Abnormal_Activities":
                print("Abnormal activity detected. This can be used to trigger an event like a notification or a front-end alarm.")

            for frame in original_frame_buffer:
                display_frame = frame.copy()
                text = f"Prediction: {predicted_class_name}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3

                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                x = frame.shape[1] - text_width - 20
                y = 50

                color = (0, 255, 0) if predicted_class_name == "Normal_Activities" else (0, 0, 255)

                cv2.rectangle(display_frame,
                              (x - 10, y - text_height - 10),
                              (x + text_width + 10, y + 10),
                              color, -1)

                cv2.putText(display_frame, text, (x, y),
                            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                out.write(display_frame)

            original_frame_buffer = []
            processed_frame_buffer = []

    cap.release()
    out.release()
    return output_video_path

# Route for handling video uploads (POST request)
@app.route("/upload", methods=["POST"])
def upload_video():
    """Handles the video upload and initiates processing."""
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    input_path = os.path.join("uploads", file.filename)
    output_path = os.path.join("outputs", "predicted_" + file.filename)
    file.save(input_path)

    try:
        processed_file = process_video(input_path, output_path)
        return send_file(processed_file, as_attachment=True)
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)