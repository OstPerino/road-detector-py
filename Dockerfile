# Многоэтапная сборка для уменьшения размера итогового образа
FROM python:3.10-slim as builder

# Устанавливаем зависимости для сборки
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python зависимости в отдельную директорию
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Финальный образ
FROM python:3.10-slim

# Устанавливаем только runtime зависимости
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Копируем установленные пакеты из builder stage
COPY --from=builder /root/.local /root/.local

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем только необходимые файлы
COPY app.py .
COPY ai/best.pt ai/best.pt

# Создаем директории
RUN mkdir -p temp output

# Создаем непривилегированного пользователя
RUN addgroup --gid 1001 --system appgroup && \
    adduser --uid 1001 --system --gid 1001 appuser && \
    chown -R appuser:appgroup /app

USER appuser

# Убеждаемся что Python находит наши пакеты
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.10/site-packages:$PYTHONPATH

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 