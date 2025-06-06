# Road Marking Detector API v2.0

API сервис для анализа дорожной разметки в видеофайлах с использованием нейронной сети.

## 🚀 Новые возможности v2.0

- ✅ **Аннотированное видео**: API теперь создает видео с выделенными детекциями
- ✅ **Два формата ответа**: JSON-только или ZIP с видео + JSON
- ✅ **Улучшенная обработка ошибок**
- ✅ **Подробное логирование**

## Запуск сервиса

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск сервера

```bash
# Рекомендуемый способ
python run_server.py

# Альтернативные способы
python app.py
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Сервер будет доступен по адресу: http://localhost:8000

## API Endpoints

### POST /analyze-road-marking 🎥

**Основной endpoint** - возвращает ZIP архив с аннотированным видео и JSON результатами.

**Параметры (form-data):**
- `video` (file): Видеофайл для анализа
- `lat1` (float): Широта начальной точки маршрута
- `lon1` (float): Долгота начальной точки маршрута  
- `lat2` (float): Широта конечной точки маршрута
- `lon2` (float): Долгота конечной точки маршрута
- `segment_length_m` (int, optional): Длина сегмента в метрах (по умолчанию: 10)

**Ответ:** ZIP файл содержащий:
- `analysis_results.json` - результаты анализа
- `annotated_[filename]` - видео с выделенными детекциями

**Пример запроса (Postman/ApiDog):**
```
POST http://localhost:8000/analyze-road-marking
Content-Type: multipart/form-data

form-data:
├── video: [выберите видеофайл]
├── lat1: 55.996508
├── lon1: 92.792385
├── lat2: 55.995836
├── lon2: 92.785775
└── segment_length_m: 10
```

**Пример запроса (curl):**
```bash
curl -X POST "http://localhost:8000/analyze-road-marking" \
     -F "video=@path/to/video.mp4" \
     -F "lat1=55.996508" \
     -F "lon1=92.792385" \
     -F "lat2=55.995836" \
     -F "lon2=92.785775" \
     -F "segment_length_m=10" \
     -o "result.zip"
```

### POST /analyze-road-marking-json 📊

**JSON-only endpoint** - возвращает только результаты анализа в JSON формате (быстрее).

**Параметры:** те же, что и выше

**Пример ответа:**
```json
{
  "status": "success",
  "overall_stats": {
    "total_frames": 450,
    "total_distance_meters": 562.3,
    "segment_length_meters": 10,
    "total_segments": 57,
    "segments_with_data": 55,
    "average_coverage": 78.5
  },
  "segments": [
    {
      "segment_id": 1,
      "frames_count": 8,
      "coverage_percentage": 87.5,
      "has_data": true
    }
  ],
  "coordinates": {
    "start": {"lat": 55.996508, "lon": 92.792385},
    "end": {"lat": 55.995836, "lon": 92.785775}
  }
}
```

### GET /health ❤️

Проверка состояния сервиса.

**Ответ:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### GET / ℹ️

Информация об API и доступных endpoints.

## 🔧 Настройка в Postman/ApiDog

### Исправление проблемы "Field required"

Если вы получаете ошибку `"Field required"` для поля video:

1. **Убедитесь, что используете form-data**, а не raw/JSON
2. **Для поля `video` выберите тип "File"**, а не "Text"
3. **Для остальных полей используйте тип "Text"**

### Правильная настройка:

```
Method: POST
URL: http://localhost:8000/analyze-road-marking-json

Body > form-data:
┌─────────────────┬──────┬────────────────────────────┐
│ Key             │ Type │ Value                      │
├─────────────────┼──────┼────────────────────────────┤
│ video           │ File │ [Select Files...] ✅       │
│ lat1            │ Text │ 55.996508                  │
│ lon1            │ Text │ 92.792385                  │
│ lat2            │ Text │ 55.995836                  │
│ lon2            │ Text │ 92.785775                  │
│ segment_length_m│ Text │ 10                         │
└─────────────────┴──────┴────────────────────────────┘
```

## 📋 Тестирование

### Автоматическое тестирование:
```bash
python test_api.py
```

### Ручное тестирование:

1. **Запустите сервер**: `python run_server.py`
2. **Проверьте здоровье**: GET `http://localhost:8000/health`
3. **Тестируйте JSON endpoint**: POST `http://localhost:8000/analyze-road-marking-json`
4. **Тестируйте ZIP endpoint**: POST `http://localhost:8000/analyze-road-marking`

## 📚 Интерактивная документация

FastAPI автоматически генерирует документацию:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📁 Структура проекта

```
road-detector-py/
├── app.py              # Основное FastAPI приложение v2.0
├── run_server.py       # Скрипт для запуска сервера
├── test_api.py         # Скрипт для тестирования API
├── main.py             # Оригинальный скрипт (для справки)
├── requirements.txt    # Зависимости Python
├── API_README.md       # Эта документация
├── ai/
│   ├── best.pt        # Обученная модель YOLO
│   └── input/         # Входные видеофайлы
└── output/            # Выходные файлы
```

## 🎯 Особенности аннотированного видео

Аннотированное видео содержит:
- **Bounding boxes** вокруг обнаруженной разметки
- **Текстовые метки** с номером кадра и статусом обнаружения
- **Цветовое кодирование**: 
  - 🟢 Зеленый = разметка обнаружена
  - 🔴 Красный = разметка не обнаружена

## ⚠️ Важные примечания

- Модель автоматически загружается при старте сервиса
- Поддерживается CUDA (если доступна), иначе CPU
- Временные файлы автоматически удаляются после обработки
- Поддерживаются различные форматы видео (MP4, MOV, AVI, etc.)
- Координаты должны быть в формате WGS84 (десятичные градусы)
- Максимальный размер видео ограничен настройками FastAPI

## 🚨 Решение проблем

### Ошибка "Field required"
- Используйте `form-data`, а не `raw/JSON`
- Поле `video` должно быть типа `File`

### Ошибка "Model not found"
- Убедитесь, что файл `ai/best.pt` существует

### Медленная обработка
- Используйте endpoint `/analyze-road-marking-json` для быстрого получения только результатов
- Проверьте наличие CUDA для ускорения

### Большой размер ZIP
- Это нормально - архив содержит полное аннотированное видео 