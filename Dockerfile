FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 aegis && useradd --uid 10001 --gid aegis --no-create-home aegis
COPY requirements-harness.txt ./
ARG PIP_DOWNLOAD_TIMEOUT=120
ENV PIP_DEFAULT_TIMEOUT=${PIP_DOWNLOAD_TIMEOUT}
RUN pip install -r requirements-harness.txt
ENV TIKTOKEN_CACHE_DIR=/app/tokenizer-cache
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && chmod -R a+rX /app/tokenizer-cache
ARG INSTALL_RAG=0
RUN if [ "$INSTALL_RAG" = "1" ]; then pip install 'pymilvus>=2.5,<3' 'sentence-transformers>=3.3,<6' 'numpy>=2,<3'; fi
COPY app ./app
RUN mkdir -p /app/data && chown -R aegis:aegis /app/data
USER aegis
EXPOSE 8000
CMD ["python", "-m", "app.launcher", "--host", "0.0.0.0", "--no-browser"]
