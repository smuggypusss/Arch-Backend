# Use a lightweight official Python image
FROM python:3.12-slim

# Install system dependencies needed for OpenCV, Pillow, SQLite, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port (Cloud Run defaults to PORT env variable)
ENV PORT=8080
EXPOSE 8080

# Run the app with Uvicorn, binding to the port specified by environment variable
CMD uvicorn app.main:app --host 0.0.0.0 --port 8080
