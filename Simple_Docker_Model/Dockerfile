# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the resources into the container
COPY inference.py /app
COPY model.pkl /app
COPY requirements.txt /app

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port for the app
EXPOSE 5432

# Define flask app env var
ENV FLASK_APP=inference.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5432"]