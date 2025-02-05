# syntax=docker/dockerfile:1.4
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy APP_HOME=/app DOCKER_BUILDKIT=1
ENV DOCKER_DEFAULT_PLATFORM=linux/amd64
# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements files
COPY uv.lock pyproject.toml ./

# Copy application code
COPY ./src /app/src

# Install project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Create app user
RUN useradd -m app

# Set PATH to include venv binaries
ENV PATH="/app/.venv/bin:$PATH"

USER app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
