import requests
import os
import json

def test_api():
    """Тестирует API сервиса анализа дорожной разметки"""
    
    # URL сервиса
    base_url = "http://localhost:8000"
    
    # Проверяем, что сервис запущен
    try:
        response = requests.get(f"{base_url}/health")
        print("Health check:", response.json())
    except requests.exceptions.ConnectionError:
        print("Сервис не запущен! Запустите сначала: python run_server.py")
        return
    
    # Путь к тестовому видео
    video_path = "ai/input/IMG_9109.MOV"
    
    if not os.path.exists(video_path):
        print(f"Тестовое видео не найдено: {video_path}")
        return
    
    # Данные для тестирования
    data = {
        'lat1': 55.996508,
        'lon1': 92.792385,
        'lat2': 55.995836,
        'lon2': 92.785775,
        'segment_length_m': 10
    }
    
    print("\n=== ТЕСТИРОВАНИЕ JSON ENDPOINT ===")
    print(f"Отправляем видео для анализа: {video_path}")
    
    # Тестируем JSON endpoint
    with open(video_path, 'rb') as video_file:
        files = {'video': ('video.MOV', video_file, 'video/quicktime')}
        
        response = requests.post(
            f"{base_url}/analyze-road-marking-json",
            data=data,
            files=files
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Статус: {result['status']}")
        
        stats = result['overall_stats']
        print(f"\nОбщая статистика:")
        print(f"  Всего кадров: {stats['total_frames']}")
        print(f"  Общее расстояние: {stats['total_distance_meters']} м")
        print(f"  Длина сегмента: {stats['segment_length_meters']} м")
        print(f"  Всего сегментов: {stats['total_segments']}")
        print(f"  Сегментов с данными: {stats['segments_with_data']}")
        print(f"  Среднее покрытие: {stats['average_coverage']}%")
        
        print(f"\nПервые 5 сегментов:")
        for i, segment in enumerate(result['segments'][:5]):
            if segment['has_data']:
                print(f"  Сегмент {segment['segment_id']}: {segment['frames_count']} кадров, покрытие = {segment['coverage_percentage']}%")
            else:
                print(f"  Сегмент {segment['segment_id']}: нет данных")
    else:
        print(f"Ошибка JSON endpoint: {response.status_code}")
        print(response.text)
        return
    
    print("\n=== ТЕСТИРОВАНИЕ ZIP ENDPOINT ===")
    print("Получаем ZIP архив с аннотированным видео...")
    
    # Тестируем ZIP endpoint
    with open(video_path, 'rb') as video_file:
        files = {'video': ('video.MOV', video_file, 'video/quicktime')}
        
        response = requests.post(
            f"{base_url}/analyze-road-marking",
            data=data,
            files=files
        )
    
    if response.status_code == 200:
        # Сохраняем ZIP файл
        zip_filename = "road_analysis_result.zip"
        with open(zip_filename, 'wb') as f:
            f.write(response.content)
        
        print(f"ZIP файл сохранен: {zip_filename}")
        print(f"Размер файла: {len(response.content)} bytes")
        
        # Проверяем содержимое ZIP
        import zipfile
        try:
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                files_in_zip = zip_ref.namelist()
                print(f"Файлы в архиве: {files_in_zip}")
                
                # Извлекаем и показываем JSON результаты
                if 'analysis_results.json' in files_in_zip:
                    with zip_ref.open('analysis_results.json') as json_file:
                        results = json.load(json_file)
                        print(f"JSON результаты в архиве: статус = {results['status']}")
        except Exception as e:
            print(f"Ошибка при чтении ZIP: {e}")
            
    else:
        print(f"Ошибка ZIP endpoint: {response.status_code}")
        print(response.text)

def test_with_curl_examples():
    """Выводит примеры команд curl для тестирования"""
    print("\n=== ПРИМЕРЫ CURL КОМАНД ===")
    
    print("\n1. Получение только JSON результатов:")
    print("""curl -X POST "http://localhost:8000/analyze-road-marking-json" \\
     -F "video=@ai/input/IMG_9109.MOV" \\
     -F "lat1=55.996508" \\
     -F "lon1=92.792385" \\
     -F "lat2=55.995836" \\
     -F "lon2=92.785775" \\
     -F "segment_length_m=10" """)
    
    print("\n2. Получение ZIP архива с видео и JSON:")
    print("""curl -X POST "http://localhost:8000/analyze-road-marking" \\
     -F "video=@ai/input/IMG_9109.MOV" \\
     -F "lat1=55.996508" \\
     -F "lon1=92.792385" \\
     -F "lat2=55.995836" \\
     -F "lon2=92.785775" \\
     -F "segment_length_m=10" \\
     -o "result.zip" """)

if __name__ == "__main__":
    test_api()
    test_with_curl_examples() 