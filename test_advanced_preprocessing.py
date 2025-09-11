#!/usr/bin/env python3
"""
고급 이미지 전처리 테스트 스크립트

이미지 전처리 파이프라인의 고도화 기능을 테스트합니다.
"""

import cv2
import numpy as np
import os
from src.preprocessing.image_processor import ImageProcessor
from src.ocr.ocr_engine import OCREngine

def test_advanced_preprocessing():
    """고급 전처리 기능 테스트"""
    
    # 테스트용 이미지 생성 (실제 번호판 이미지 시뮬레이션)
    test_image = create_test_plate_image()
    
    # ImageProcessor 초기화
    processor = ImageProcessor()
    
    print("=== 고급 이미지 전처리 테스트 ===")
    print(f"원본 이미지 크기: {test_image.shape}")
    
    # 1. 자동 최적화 모드 테스트
    print("\n1. 자동 최적화 모드 테스트...")
    auto_enhanced = processor.auto_enhance(test_image)
    print(f"자동 최적화 결과 크기: {auto_enhanced.shape}")
    print(f"자동 최적화 데이터 타입: {auto_enhanced.dtype}")
    
    # 2. 각 품질 모드 테스트
    for mode in ['fast', 'balanced', 'high_quality']:
        print(f"\n2. {mode} 모드 테스트...")
        try:
            result = processor.process_advanced(test_image, quality_mode=mode)
            enhanced_image = result['processed_image']
            print(f"  - 처리 완료: {enhanced_image.shape}")
            print(f"  - 품질 메트릭 수: {len(result.get('quality_metrics', {}))}")
            print(f"  - 처리 단계 수: {len(result.get('processing_steps', []))}")
        except Exception as e:
            print(f"  - 오류 발생: {e}")
    
    # 3. OCR 엔진과의 통합 테스트
    print("\n3. OCR 엔진 통합 테스트...")
    try:
        ocr_engine = OCREngine()
        
        # 다양한 전처리 모드로 OCR 테스트
        modes = ['off', 'auto', 'fast', 'balanced']
        for mode in modes:
            print(f"  - {mode} 모드 OCR 테스트...")
            try:
                result = ocr_engine.recognize_with_classification(test_image, preprocessing_mode=mode)
                print(f"    인식 결과: '{result['text']}'")
                print(f"    신뢰도: {result['confidence']:.3f}")
                print(f"    전처리 모드: {result['preprocessing_info'].get('mode', 'unknown')}")
            except Exception as e:
                print(f"    오류: {e}")
    except ImportError as e:
        print(f"  OCR 엔진 로드 실패 (EasyOCR 미설치): {e}")
    except Exception as e:
        print(f"  OCR 테스트 오류: {e}")
    
    print("\n=== 테스트 완료 ===")
    return True

def create_test_plate_image():
    """테스트용 번호판 이미지 생성"""
    # 흰색 배경에 검은 글씨로 간단한 번호판 시뮬레이션
    img = np.ones((60, 200, 3), dtype=np.uint8) * 255
    
    # 노이즈 추가 (전처리 효과를 보기 위해)
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # 블러 효과 추가
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    # 텍스트 추가 (실제로는 OCR이 읽을 수 있는 텍스트)
    cv2.putText(img, '12가3456', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img

if __name__ == "__main__":
    test_advanced_preprocessing()