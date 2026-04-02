FROM python:3.11-slim

WORKDIR /app/env

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

COPY requirements.txt /app/env/requirements.txt
RUN pip install --no-cache-dir -r /app/env/requirements.txt

COPY . /app/env

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
