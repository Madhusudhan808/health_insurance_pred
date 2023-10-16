# Use an official Python runtime as a parent image
FROM python:3.7

# allow the statement and log messages to  apper 
ENV PYTHONUNBUFFERED True
# Set the working directory in the container
WORKDIR /app
ENV  APP_HOME /app

# Copy your model code and dependencies into the container
COPY . /app

# Install required packages
RUN pip install -r requirements.txt

# Define the command to run your model (e.g., your_model_script.py)
CMD exec gunicorn  --bind  :$PORT  --worker 1 --threads 8  --timeout 0  main:app
