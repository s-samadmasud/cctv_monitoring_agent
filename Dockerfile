FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV and gunicorn
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    libx11-6 \
    libxau6 \
    libxdmcp6 \
    libxcb1 \
    libxrender1 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    ffmpeg \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app will run on
EXPOSE 5000

# Run the application using Gunicorn with a longer timeout
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "app:app"]