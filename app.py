from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
from geopy.distance import geodesic
import tempfile
import os
from typing import List, Dict, Any, Optional
import json
import uvicorn
import zipfile
import shutil

app = FastAPI(title="Road Marking Detector API", version="1.0.0")

# Загружаем модель при старте приложения
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model_path = "ai/best.pt"
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    
    model = YOLO(model_path)
    try:
        model.to("cuda")
        print("Model loaded on CUDA")
    except:
        print("CUDA not available, using CPU")

class RoadMarkingAnalyzer:
    def __init__(self, model):
        self.model = model
    
    def analyze_video_with_annotations(self, video_path: str, lat1: float, lon1: float, 
                                     lat2: float, lon2: float, segment_length_m: int = 10) -> tuple[Dict[str, Any], str]:
        """
        Анализирует видео и создает аннотированное видео с результатами
        Возвращает кортеж: (результаты_анализа, путь_к_аннотированному_видео)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Получаем параметры видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Создаем выходное видео
        output_video_path = tempfile.mktemp(suffix='_annotated.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_results = []
        frame_count = 0
        
        # Анализ каждого кадра с аннотациями
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            results = self.model(frame)
            
            # Рисуем детекции на кадре
            annotated_frame = results[0].plot()
            
            # Добавляем информацию о покрытии
            has_marking = int(len(results[0].boxes) > 0)
            frame_results.append(has_marking)
            
            # Добавляем текст с информацией
            coverage_text = f"Frame {frame_count + 1}: {'Marking Detected' if has_marking else 'No Marking'}"
            cv2.putText(annotated_frame, coverage_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if has_marking else (0, 0, 255), 2)
            
            out.write(annotated_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        if frame_count == 0:
            raise ValueError("Видео не содержит кадров или не может быть прочитано")
        
        # Интерполяция координат
        N = len(frame_results)
        frame_coords = [
            (
                lat1 + (lat2 - lat1) * i / (N - 1),
                lon1 + (lon2 - lon1) * i / (N - 1)
            )
            for i in range(N)
        ]
        
        # Разбиваем на сегменты
        total_distance = geodesic((lat1, lon1), (lat2, lon2)).meters
        num_segments = int(np.ceil(total_distance / segment_length_m))
        segment_frames = [[] for _ in range(num_segments)]
        
        for idx, (lat, lon) in enumerate(frame_coords):
            dist_from_start = geodesic((lat1, lon1), (lat, lon)).meters
            seg_idx = min(int(dist_from_start // segment_length_m), num_segments - 1)
            segment_frames[seg_idx].append(frame_results[idx])
        
        # Считаем покрытие по сегментам
        segment_coverage = []
        segment_details = []
        
        for i, frames in enumerate(segment_frames):
            if frames:
                coverage = 100 * sum(frames) / len(frames)
                segment_coverage.append(coverage)
                segment_details.append({
                    "segment_id": i + 1,
                    "frames_count": len(frames),
                    "coverage_percentage": round(coverage, 1),
                    "has_data": True
                })
            else:
                segment_coverage.append(None)
                segment_details.append({
                    "segment_id": i + 1,
                    "frames_count": 0,
                    "coverage_percentage": None,
                    "has_data": False
                })
        
        # Общая статистика
        valid_coverages = [c for c in segment_coverage if c is not None]
        overall_stats = {
            "total_frames": frame_count,
            "total_distance_meters": round(total_distance, 2),
            "segment_length_meters": segment_length_m,
            "total_segments": num_segments,
            "segments_with_data": len(valid_coverages),
            "average_coverage": round(sum(valid_coverages) / len(valid_coverages), 1) if valid_coverages else None
        }
        
        analysis_results = {
            "status": "success",
            "overall_stats": overall_stats,
            "segments": segment_details,
            "coordinates": {
                "start": {"lat": lat1, "lon": lon1},
                "end": {"lat": lat2, "lon": lon2}
            }
        }
        
        return analysis_results, output_video_path

@app.post("/analyze-road-marking")
async def analyze_road_marking(
    video: UploadFile = File(..., description="Видеофайл для анализа"),
    lat1: float = Form(..., description="Широта начальной точки"),
    lon1: float = Form(..., description="Долгота начальной точки"),
    lat2: float = Form(..., description="Широта конечной точки"),
    lon2: float = Form(..., description="Долгота конечной точки"),
    segment_length_m: int = Form(10, description="Длина сегмента в метрах")
):
    """
    Анализирует дорожную разметку в видео и возвращает результаты + аннотированное видео
    """
    
    print(f"Received file: {video.filename}, content_type: {video.content_type}")
    
    # Проверяем, что файл является видео
    if not video.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")
        
    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        try:
            content = await video.read()
            tmp_file.write(content)
            tmp_video_path = tmp_file.name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Ошибка при чтении файла: {str(e)}")
    
    try:
        # Анализируем видео
        analyzer = RoadMarkingAnalyzer(model)
        results, annotated_video_path = analyzer.analyze_video_with_annotations(
            video_path=tmp_video_path,
            lat1=lat1, lon1=lon1,
            lat2=lat2, lon2=lon2,
            segment_length_m=segment_length_m
        )
        
        # Создаем архив с результатами
        zip_path = tempfile.mktemp(suffix='.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Добавляем JSON с результатами
            json_path = tempfile.mktemp(suffix='.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            zipf.write(json_path, 'analysis_results.json')
            
            # Добавляем аннотированное видео
            zipf.write(annotated_video_path, f'annotated_{video.filename}')
            
            # Удаляем временные файлы
            os.unlink(json_path)
        
        # Возвращаем архив
        return FileResponse(
            zip_path, 
            media_type='application/zip',
            filename=f'road_analysis_{video.filename.split(".")[0]}.zip',
            headers={"Content-Disposition": f"attachment; filename=road_analysis_{video.filename.split('.')[0]}.zip"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе видео: {str(e)}")
    
    finally:
        # Удаляем временные файлы
        for path in [tmp_video_path, annotated_video_path if 'annotated_video_path' in locals() else None]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

@app.post("/analyze-road-marking-json")
async def analyze_road_marking_json(
    video: UploadFile = File(...),
    lat1: float = Form(...),
    lon1: float = Form(...),
    lat2: float = Form(...),
    lon2: float = Form(...),
    segment_length_m: int = Form(10)
):
    """
    Анализирует дорожную разметку в видео и возвращает только JSON результаты (без видео)
    """
    
    print(f"Received file: {video.filename}, content_type: {video.content_type}")
    
    if not video.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")
    
    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        try:
            content = await video.read()
            tmp_file.write(content)
            tmp_video_path = tmp_file.name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Ошибка при чтении файла: {str(e)}")
    
    try:
        # Анализируем видео (без создания аннотированного видео)
        analyzer = RoadMarkingAnalyzer(model)
        results, _ = analyzer.analyze_video_with_annotations(
            video_path=tmp_video_path,
            lat1=lat1, lon1=lon1,
            lat2=lat2, lon2=lon2,
            segment_length_m=segment_length_m
        )
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе видео: {str(e)}")
    
    finally:
        # Удаляем временный файл
        if os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Корневой endpoint с информацией о API"""
    return {
        "message": "Road Marking Detector API",
        "version": "2.0.0",
        "endpoints": {
            "POST /analyze-road-marking": "Анализ дорожной разметки (возвращает ZIP с видео и JSON)",
            "POST /analyze-road-marking-json": "Анализ дорожной разметки (только JSON результаты)",
            "GET /health": "Проверка состояния сервиса",
            "GET /": "Информация об API"
        }
    }

@app.post("/analyze")
async def analyze_for_go_service(
    video: UploadFile = File(..., description="Видеофайл для анализа"),
    startLat: float = Form(..., description="Широта начальной точки"),
    startLon: float = Form(..., description="Долгота начальной точки"),
    endLat: float = Form(..., description="Широта конечной точки"),
    endLon: float = Form(..., description="Долгота конечной точки"),
    segmentLength: int = Form(10, description="Длина сегмента в метрах")
):
    """
    Анализирует видео дорожной разметки и возвращает результаты для Go сервиса
    """
    
    print(f"Received file: {video.filename}, content_type: {video.content_type}")
    
    # Проверяем, что файл является видео
    if not video.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")
        
    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        try:
            content = await video.read()
            tmp_file.write(content)
            tmp_video_path = tmp_file.name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Ошибка при чтении файла: {str(e)}")
    
    try:
        # Анализируем видео (только нейронная сеть, без географии)
        cap = cv2.VideoCapture(tmp_video_path)
        
        frame_results = []
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            results = model(frame)
            # Если есть хотя бы один detection — считаем, что разметка есть
            has_marking = int(len(results[0].boxes) > 0)
            frame_results.append(has_marking)
            frame_count += 1
        
        cap.release()
        
        if frame_count == 0:
            raise ValueError("Видео не содержит кадров или не может быть прочитано")
        
        print(f"Обработано кадров: {frame_count}")
        
        # Возвращаем результаты в формате, который ожидает Go сервис
        return JSONResponse(content={
            "status": "success",
            "message": f"Успешно обработано {frame_count} кадров",
            "frame_results": frame_results
        })
        
    except Exception as e:
        print(f"Ошибка при анализе видео: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Ошибка при анализе видео: {str(e)}",
                "frame_results": []
            }
        )
    finally:
        # Удаляем временный файл
        if os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 