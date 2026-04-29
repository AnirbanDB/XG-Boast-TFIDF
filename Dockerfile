# Dockerfile for Firco XGBoost Compliance Predictor API

# --- Stage 1: Build Environment ---
# We use a full Python image to build our dependencies, including any with C extensions.
FROM python:3.12-slim as builder

# Set the working directory
WORKDIR /app

# Install build essentials for packages that need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file to leverage Docker's layer caching.
# This step will only be re-run if requirements.txt changes.
COPY requirements.txt .

# Install the Python dependencies into a virtual environment.
# Using a venv inside the build stage keeps the global site-packages clean.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Runtime Environment ---
# We use a slim base image for the final container to keep it small and secure.
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the build stage.
# This is more efficient and secure than installing requirements again.
COPY --from=builder /opt/venv /opt/venv

# Copy the application code into the container.
# We copy the entire `xgb` directory content.
COPY . .

# Set the PATH to include the venv's bin directory.
# This ensures that `python` and other commands run from the venv.
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port the application will run on.
# This should match the port configured in your API.
EXPOSE 3004

# The command to run the application when the container starts.
# We use uvicorn to run the FastAPI app, as specified in xgb_app_F.py.
# We bind to 0.0.0.0 to make it accessible from outside the container.
CMD ["uvicorn", "xgb_app_F:app", "--host", "0.0.0.0", "--port", "${PORT:-3004}"]