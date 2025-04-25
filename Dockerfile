# Use a more complete image
FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir pandas scikit-learn matplotlib

# Copy current directory contents into the container
COPY . .

# Set the default command
CMD ["python", "hello_world_ml.py"]
