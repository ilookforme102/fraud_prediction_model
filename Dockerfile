FROM python:3.11-slim
RUN pip install numpy==1.26.3
# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY app.py .
COPY rfc_model.pkl .
COPY xgbc_model.pkl .

# Expose the port that the Flask app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]