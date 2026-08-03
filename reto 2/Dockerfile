# -------- Builder stage --------
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# -------- Runtime stage --------
FROM python:3.12-slim

# Keeps Python from generating .pyc files and ensures unbuffered output.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy installed dependencies from the builder stage into the runtime image.
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application source + committed sample dataset.
COPY app ./app
COPY data ./data
COPY .env.example ./.env.example

# Create a non-root user for security best practice.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose the FastAPI/uvicorn port.
EXPOSE 8000

# Healthcheck so orchestrators can monitor the service.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run uvicorn with the FastAPI app factory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]