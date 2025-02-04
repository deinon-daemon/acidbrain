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


# Create venv
# Install py dependencies now (layer caching improved by multi-stage build)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev



# Copy requirements files
COPY requirements.txt uv.lock pyproject.toml ./

# Copy application code
COPY ./src /app/src

# Install project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.11-slim-bookworm
ENV APP_HOME=/app DOCKER_BUILDKIT=1
# Set working directory
WORKDIR /app

# Create app user
RUN useradd -m app

# Copy the application from the builder
COPY --from=builder --chown=app:app /app /app

# Set PATH to include venv binaries
ENV PATH="/app/.venv/bin:$PATH"

USER app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
