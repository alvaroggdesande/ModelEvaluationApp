# Use the official slim Python 3.10 image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements first for better Docker layer caching
COPY requirements.txt .
# Install dependencies without caching to minimize image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit listens on
EXPOSE 80

# Launch the Streamlit app
CMD ["streamlit", "run", "app.py", \
     "--server.port", "80", \
     "--server.enableCORS", "false", \
     "--server.baseUrlPath", "/model-monitor"]
