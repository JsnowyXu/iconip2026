FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY platform/backend/requirements.txt /app/platform/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt -r /app/platform/backend/requirements.txt

COPY . /app
WORKDIR /app/platform/backend
RUN mkdir -p /app/platform/runtime

ENTRYPOINT ["/app/platform/backend/entrypoint.sh"]
CMD ["gunicorn", "dhc_platform.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4"]
