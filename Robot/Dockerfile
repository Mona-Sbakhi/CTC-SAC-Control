FROM python:3.10-slim

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Default working directory for scripts
WORKDIR /workspace/src

CMD ["bash"]
