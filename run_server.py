#!/usr/bin/env python3
import uvicorn
from app import app

if __name__ == "__main__":
    print("=== Road Marking Detector API Server ===")
    print("Запуск сервера на http://localhost:8000")
    print("Документация API: http://localhost:8000/docs")
    print("Для остановки нажмите Ctrl+C")
    print("=" * 45)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Автоперезагрузка при изменении кода
        log_level="info"
    ) 