#!/usr/bin/env python
"""
차량번호 OCR 프로젝트에 필요한 모델 파일들을 자동으로 다운로드하는 스크립트
실행 방법: python download_models.py
"""

import os
import sys
import requests
import shutil
from pathlib import Path
from tqdm import tqdm

# 저장 경로 설정
models_dir = os.path.join("data", "models")
os.makedirs(models_dir, exist_ok=True)

def download_file(url, destination):
    """URL에서 파일을 다운로드하고 진행 상황을 표시합니다."""
    if os.path.exists(destination):
        print(f"파일이 이미 존재합니다: {destination}")
        return
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(destination, 'wb') as file, tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        
        print(f"다운로드 완료: {destination}")
    except Exception as e:
        print(f"다운로드 실패: {url}\n에러: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        sys.exit(1)

def download_yolo_model():
    """YOLOv8s 모델을 다운로드합니다."""
    destination = os.path.join(models_dir, "yolov8s.pt")
    
    if os.path.exists(destination):
        print(f"YOLOv8s 모델이 이미 존재합니다: {destination}")
        return
    
    # 직접 GitHub에서 모델 파일 다운로드
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    print(f"YOLOv8s 모델 다운로드 중: {url}")
    download_file(url, destination)

def download_easyocr_models():
    """EasyOCR 모듈을 사용하여 필요한 모델을 다운로드합니다."""
    try:
        import easyocr
        
        print("EasyOCR 모델 다운로드 중...")
        # 한국어 인식을 위한 EasyOCR Reader 초기화 (처음 호출 시 모델 다운로드)
        reader = easyocr.Reader(['ko', 'en'], gpu=False, download_enabled=True)
        
        # EasyOCR은 모델을 ~/.EasyOCR/model 디렉토리에 저장합니다
        # 프로젝트의 models 디렉토리로 필요한 파일 복사
        home_dir = os.path.expanduser("~")
        easyocr_model_dir = os.path.join(home_dir, '.EasyOCR', 'model')
        
        # 필요한 모델 파일 복사
        model_files = {
            'craft_mlt_25k.pth': 'craft_mlt_25k.pth',  # 텍스트 감지 모델
            'korean_g2.pth': 'korean_g2.pth'           # 한국어 인식 모델
        }
        
        for src_file, dst_file in model_files.items():
            src_path = os.path.join(easyocr_model_dir, src_file)
            dst_path = os.path.join(models_dir, dst_file)
            
            if os.path.exists(src_path):
                if os.path.exists(dst_path):
                    print(f"모델 파일이 이미 존재합니다: {dst_path}")
                else:
                    shutil.copy2(src_path, dst_path)
                    print(f"모델 파일 복사 완료: {src_path} -> {dst_path}")
            else:
                print(f"경고: 소스 모델 파일을 찾을 수 없습니다: {src_path}")
                print(f"EasyOCR이 모델을 다운로드했지만 예상 위치에 없습니다.")
                
        print("EasyOCR 모델 다운로드 및 복사 완료")
        
    except ImportError:
        print("EasyOCR 패키지가 설치되어 있지 않습니다.")
        print("pip install easyocr 명령으로 설치한 후 다시 시도하세요.")
        
        # EasyOCR 없이 직접 다운로드 시도
        craft_model_path = os.path.join(models_dir, "craft_mlt_25k.pth")
        if not os.path.exists(craft_model_path):
            craft_url = "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.pth"
            print("대체 방법으로 CRAFT 텍스트 감지 모델 다운로드 중...")
            download_file(craft_url, craft_model_path)
        
        korean_model_path = os.path.join(models_dir, "korean_g2.pth")
        if not os.path.exists(korean_model_path):
            korean_url = "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/korean_g2.pth"
            print("대체 방법으로 한국어 인식 모델 다운로드 중...")
            download_file(korean_url, korean_model_path)
    
    except Exception as e:
        print(f"EasyOCR 모델 다운로드 중 오류 발생: {e}")
        print("대체 방법으로 직접 다운로드를 시도합니다...")
        
        # 직접 다운로드 시도
        craft_model_path = os.path.join(models_dir, "craft_mlt_25k.pth")
        if not os.path.exists(craft_model_path):
            craft_url = "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.pth"
            print("CRAFT 텍스트 감지 모델 다운로드 중...")
            download_file(craft_url, craft_model_path)
        
        korean_model_path = os.path.join(models_dir, "korean_g2.pth")
        if not os.path.exists(korean_model_path):
            korean_url = "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/korean_g2.pth"
            print("한국어 인식 모델 다운로드 중...")
            download_file(korean_url, korean_model_path)

def download_license_plate_model():
    """번호판 탐지 커스텀 모델 예시 다운로드 (데모용)"""
    destination = os.path.join(models_dir, "license_plate_detection.pt")
    
    if os.path.exists(destination):
        print(f"번호판 탐지 모델이 이미 존재합니다: {destination}")
        return
    
    print("번호판 탐지 모델은 커스텀 학습된 모델입니다.")
    print("데모용 모델 파일이 있다면 여기서 다운로드하거나 수동으로 추가해야 합니다.")
    
    # 실제 프로젝트에서는 학습된 모델 파일의 URL을 여기에 지정
    # 현재는 예시로 빈 파일 생성
    try:
        with open(destination, 'wb') as f:
            # 빈 파일 생성 (실제 프로젝트에서는 이 부분 대신 실제 모델 다운로드 코드 사용)
            print(f"주의: 번호판 감지 모델이 필요합니다. 데모용 빈 파일을 생성했습니다: {destination}")
            print("실제 사용을 위해서는 학습된 모델 파일로 교체해야 합니다.")
    except Exception as e:
        print(f"번호판 모델 파일 생성 실패: {e}")

def main():
    """모든 필요한 모델을 다운로드합니다."""
    print("차량번호 OCR 프로젝트에 필요한 모델 다운로드를 시작합니다...")
    
    # YOLOv8s 모델 다운로드
    download_yolo_model()
    
    # EasyOCR 관련 모델 다운로드
    download_easyocr_models()
    
    # 번호판 탐지 모델 데모
    download_license_plate_model()
    
    print("\n모든 모델 다운로드가 완료되었습니다.")
    print(f"모델 파일 경로: {os.path.abspath(models_dir)}")

if __name__ == "__main__":
    main()