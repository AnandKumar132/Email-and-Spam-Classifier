# Use the official Python image as the base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the application files into the working directory
COPY . /app

RUN python -m venv venv

# Activate the virtual environment
RUN /app/venv/bin/activate


# Install the application dependencies
RUN pip install -r requirements.txt

# Define the entry point for the container
CMD ["python", "smsClassifer.py", "runserver", "0.0.0.0:8000"]

