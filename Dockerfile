# Use an official Python runtime as a parent image
#FROM python:3.10.9-slim
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /fraud_detection

# Copy the requirements file first to cache dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

RUN mkdir -p /results

# Run main.py when the container launches
CMD ["python", "script/main.py"]