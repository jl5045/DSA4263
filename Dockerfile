# Use official Python runtime as base image with build tools
FROM python:3.11-slim

# Install build dependencies and unzip utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements-docker.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the src directory
COPY src/ ./src/

# Copy the plots directory
COPY plots/ ./plots/

# Copy data-docker.zip file
COPY data-docker.zip .

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Create directories for volumes (models will be mounted at runtime)
RUN mkdir -p ./models 

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1

# Set entrypoint to handle data extraction
ENTRYPOINT ["/docker-entrypoint.sh"]

# Run Streamlit app
CMD ["streamlit", "run", "src/app.py", "--client.showErrorDetails=true", "--logger.level=info"]
