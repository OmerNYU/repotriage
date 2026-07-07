FROM python:3.11-slim

WORKDIR /app

# curl for container healthcheck only
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir -e ".[ml,db]" "uvicorn[standard]>=0.32"

EXPOSE 8000

# CMD overridden by compose; documented default:
CMD ["repotriage", "serve", \
     "--config", "configs/inference/pandas-dev__pandas/local-v1.json", \
     "--host", "0.0.0.0", "--port", "8000"]
