# Python 3.11 Slim (Better for production)
FROM python:3.11-slim

# Install system libraries for Image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Torch (CPU version) and FastAPI
# We use the CPU index to keep the image under 2GB
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi uvicorn pillow python-multipart

# Copy files into the image
COPY . .

# Open the port
EXPOSE 8000

# Start the server with your SSL certs
CMD ["uvicorn", "websiteLAES:app", "--host", "0.0.0.0", "--port", "8000", "--ssl-keyfile", "localhost+2-key.pem", "--ssl-certfile", "localhost+2.pem"]