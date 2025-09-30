import os
import cv2
import numpy as np
from flask import Flask, request, send_file, render_template, jsonify
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed, LSTM, GlobalAveragePooling2D, Dropout, BatchNormalization, Dense, Input
from tensorflow.keras.applications import MobileNetV2
import multiprocessing
import uuid

# ------------------ CONSTANTS & PATHS ------------------
IMG_SIZE = 128
FRAME_COUNT = 15
CLASSES = ['Normal_Activities', 'Abnormal_Activities']

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'outputs')
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'cctv_monitoring.keras')

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------ GLOBAL MULTIPROCESSING VARIABLES ------------------
# These must be initialized globally (or explicitly defined before the pool)
# The actual objects (Manager, Pool) are initialized later inside the main block.
model = None
manager = None
task_status = None
pool = None
num_cpus = multiprocessing.cpu_count()

# ------------------ MODEL & WORKER FUNCTIONS ------------------

def build_model():
    """Defines the MobileNetV2 + LSTM model architecture."""
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
    
    return Model(inputs=input_sequence, outputs=output)

def init_worker(model_path):
    """
    Worker initializer function. Loads the model weights into each worker process 
    only once.
    """
    global model
    # Note: Global variables like task_status are inherited by the spawned process
    # but not explicitly passed here.
    model = build_model()
    try:
        model.load_weights(model_path)
    except FileNotFoundError:
        print(f"Warning: Model weights file '{model_path}' not found. Predictions disabled.")
        model = None
    except Exception as e:
        print(f"Error loading model weights: {e}")
        model = None

# FIX: Named function required for Windows multiprocessing (cannot pickle lambda)
def pool_initializer():
    """Wrapper to call init_worker with the necessary arguments."""
    init_worker(MODEL_PATH)


def process_video_task(task_id, input_video_path, output_video_path):
    """
    Performs the intensive video processing and updates the shared task_status.
    Runs in a separate worker process.
    """
    # The global model object is already initialized by init_worker()
    
    task_status[task_id] = {'status': 'processing', 'progress': 0}

    if model is None:
        task_status[task_id] = {'status': 'error', 'message': 'Model not loaded.'}
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        task_status[task_id] = {'status': 'error', 'message': 'Could not open input video.'}
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use 'mp4v' or 'avc1' for H.264 compatibility, 'mp4v' is generally safer.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    original_frame_buffer = []
    processed_frame_buffer = []
    predicted_class_name = "Normal_Activities" # Initialize prediction

    frame_count = 0
    while True:
        ret, original_frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Update progress for the frontend every 100 frames
        if total_frames > 0 and frame_count % 100 == 0:
             task_status[task_id]['progress'] = int((frame_count / total_frames) * 100)

        original_frame_buffer.append(original_frame)
        processed_frame = cv2.resize(original_frame, (IMG_SIZE, IMG_SIZE))
        processed_frame = processed_frame[:, :, ::-1] / 255.0 # Normalize 
        processed_frame_buffer.append(processed_frame)

        if len(processed_frame_buffer) == FRAME_COUNT:
            # Perform inference on the sequence
            input_sequence = np.expand_dims(np.array(processed_frame_buffer), axis=0)
            prediction = model.predict(input_sequence, verbose=0)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = CLASSES[predicted_class_index]

            # Write buffered frames with the prediction overlay
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
    
    # Clean up input file after processing is complete
    os.remove(input_video_path)

    # Final status update for the frontend
    result = {'status': 'completed', 'progress': 100, 'filename': os.path.basename(output_video_path)}
    result['prediction'] = predicted_class_name
    task_status[task_id] = result

# ------------------ FLASK APPLICATION ROUTES ------------------
app = Flask(__name__)

@app.route("/")
def home():
    """Serves the video upload form."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    """Handles video upload and queues the processing task."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_id = str(uuid.uuid4())
    filename = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, file_id + '_' + filename)
    output_path = os.path.join(OUTPUT_FOLDER, file_id + '_' + 'processed_' + filename)

    try:
        file.save(input_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {e}"}), 500

    # Initialize task status before dispatching
    task_status[file_id] = {'status': 'queued'}

    # Dispatch the processing to the background pool
    pool.apply_async(process_video_task, args=(file_id, input_path, output_path))

    return jsonify({
        "status": "success",
        "message": "Video uploaded and is now processing in the background.",
        "task_id": file_id
    }), 202

@app.route("/status/<task_id>")
def get_status(task_id):
    """Returns the status of a background video processing task for the frontend."""
    status_info = task_status.get(task_id, {'status': 'unknown', 'message': 'Task ID not found.'})
    return jsonify(status_info)

@app.route("/download/<filename>")
def download_file(filename):
    """Allows downloading of the processed video file."""
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    
    # Simple check to prevent directory traversal and ensure file exists
    if os.path.exists(filepath) and filepath.startswith(OUTPUT_FOLDER):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    # --- STARTUP LOGIC ---
    # This block is only run when the script is executed directly (e.g., 'python app.py' or by Gunicorn).
    
    # 1. Initialize Shared Multiprocessing Objects
    manager = multiprocessing.Manager()
    task_status = manager.dict()
    
    # 2. Initialize the Pool (spawns worker processes)
    # The pool uses the named pool_initializer to load the model.
    pool = multiprocessing.Pool(processes=num_cpus, initializer=pool_initializer)
    
    # 3. Start the Flask Server
    app.run(host="0.0.0.0", port=5000, debug=True)
