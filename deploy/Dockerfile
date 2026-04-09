# ── Build stage ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY . .

# ── Runtime stage ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --from=builder /build /app

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Inference API + Prometheus metrics
EXPOSE 8080 9090

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "uvicorn", "router.slo_router:app", "--host", "0.0.0.0", "--port", "8080"]
