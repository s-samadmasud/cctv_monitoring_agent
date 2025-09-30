# CCTV Monitoring Agent: Abnormal Activity Detection

## 💡 Project Overview

This project implements a real-time (near-real-time) video monitoring agent that classifies activities in a video stream as either **'Normal'** or **'Abnormal'**. It uses a convolutional recurrent neural network architecture—specifically **MobileNetV2** for feature extraction and **LSTM** layers for sequence prediction—to analyze short clips of video frames.

The application is built using **Flask** and employs **asynchronous multiprocessing** to handle video analysis in the background, preventing web server timeouts during long-running tasks.

## 🚀 Getting Started

Follow these steps to set up the project on your local machine and ensure the dependencies are correctly installed.

### 1. Prerequisites

You must have Python (3.8+) and Git installed.

### 2. Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/s-samadmasud/cctv_monitoring_agent.git](https://github.com/s-samadmasud/cctv_monitoring_agent.git)
    cd cctv_monitoring_agent
    ```

2.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On Linux/macOS:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place the Model Weights:**
    Ensure your trained Keras model file is in the correct path:
    ```
    ./models/cctv_monitoring.keras
    ```

### 3. Running Locally

Start the Flask development server. This runs your main process, which manages the multiprocessing pool for video analysis.

```bash
python app.py