import streamlit as st
import cv2
import numpy as np
import time
from ..components import (
    create_header,
    create_sidebar,
    create_upload_section,
    create_result_section,
    create_processing_options
)
from src.detection.vehicle_detector import VehicleDetector
from src.detection.plate_detector import PlateDetector
from src.preprocessing.image_processor import ImageProcessor
from src.ocr.ocr_engine import OCREngine
from src.utils.visualization import visualize_results

"""
홈 페이지 모듈

이 모듈은 Streamlit 애플리케이션의 메인 홈 페이지를 렌더링합니다.
"""
def render_home_page():
    """
    홈 페이지 렌더링
    """
    # 세션 상태 초기화
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = None
    
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    
    # 헤더 생성
    create_header()
    
    # 사이드바 생성
    settings = create_sidebar()
    
    # 업로드 섹션 생성
    input_image, filename = create_upload_section(settings["input_type"])
    
    # 처리 옵션 생성
    options = create_processing_options()
    
    # 처리 버튼
    process_button = st.button("이미지 처리 시작")
    
    # 버튼 클릭 시 처리 수행
    if process_button and input_image is not None:
        with st.spinner("이미지 처리 중..."):
            # 처리 시작 시간
            start_time = time.time()
            
            # 모델 초기화
            vehicle_detector = VehicleDetector()
            plate_detector = PlateDetector()
            image_processor = ImageProcessor()
            ocr_engine = OCREngine()
            
            # 이미지 처리 파이프라인 실행
            results, processed_images = process_image_pipeline(
                input_image,
                vehicle_detector,
                plate_detector,
                image_processor,
                ocr_engine,
                settings,
                options
            )
            
            # 처리 종료 시간
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 처리 결과 저장
            st.session_state.results = results
            st.session_state.processed_images = processed_images
            st.session_state.original_image = input_image
            
            # 처리 시간 표시
            st.success(f"처리 완료! (소요 시간: {processing_time:.2f}초)")
    
    # 결과 표시
    if st.session_state.results:
        create_result_section(
            st.session_state.results,
            st.session_state.original_image,
            st.session_state.processed_images
        )

def process_image_pipeline(image, vehicle_detector, plate_detector, image_processor, ocr_engine,
                          settings, options):
    """
    이미지 처리 파이프라인
    
    Args:
        image (numpy.ndarray): 입력 이미지
        vehicle_detector (VehicleDetector): 차량 감지 모델
        plate_detector (PlateDetector): 번호판 감지 모델
        image_processor (ImageProcessor): 이미지 처리기
        ocr_engine (OCREngine): OCR 엔진
        settings (dict): 설정 값
        options (dict): 처리 옵션
        
    Returns:
        tuple: (처리 결과 목록, 처리된 이미지 딕셔너리)
    """
    # 원본 이미지 복사
    original_image = image.copy()
    
    # 처리 과정 시각화를 위한 딕셔너리
    processed_images = {"원본": cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)}
    
    # 차량 감지
    vehicle_boxes = vehicle_detector.detect(image, conf_threshold=settings["vehicle_detection_conf"])
    
    # 차량 감지 결과 시각화
    if options["show_vehicle_detection"]:
        vehicle_image = vehicle_detector.visualize(image.copy(), vehicle_boxes)
        processed_images["차량 감지"] = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2RGB)
    
    # 결과 저장 변수
    results = []
    
    # 각 차량에 대해 처리
    for vehicle_box in vehicle_boxes:
        # 차량 영역 추출
        x1, y1, x2, y2 = vehicle_box
        vehicle_image = image[y1:y2, x1:x2].copy()
        
        # 번호판 감지
        plate_boxes = plate_detector.detect(vehicle_image, conf_threshold=settings["plate_detection_conf"])
        
        # 번호판 감지 결과 시각화
        if options["show_plate_detection"] and plate_boxes:
            plate_image = plate_detector.visualize(vehicle_image.copy(), plate_boxes)
            plate_image_key = f"번호판 감지 (차량 {len(results)+1})"
            processed_images[plate_image_key] = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
        
        # 각 번호판에 대해 처리
        for plate_box in plate_boxes:
            # 번호판 영역 추출
            px1, py1, px2, py2 = plate_box
            plate_image = vehicle_image[py1:py2, px1:px2].copy()
            
            # 이미지 전처리 옵션 설정
            preprocess_options = {
                "blur": settings["apply_blur_correction"],
                "perspective": settings["apply_perspective_correction"],
                "normalize": settings["apply_normalize"]
            }
            
            # 전처리 단계별 이미지 (옵션 선택 시)
            if options["show_processing_steps"]:
                preprocessing_steps = image_processor.visualize_steps(plate_image)
                for step_name, step_img in preprocessing_steps.items():
                    if step_name != "original":  # 원본은 이미 표시되어 있음
                        step_key = f"{step_name} (차량 {len(results)+1})"
                        # 이미지가 그레이스케일인지 확인
                        if len(step_img.shape) == 2:
                            # 3채널로 변환
                            step_img = cv2.cvtColor(step_img, cv2.COLOR_GRAY2BGR)
                        processed_images[step_key] = cv2.cvtColor(step_img, cv2.COLOR_BGR2RGB)
            
            # 번호판 이미지 전처리
            processed_plate = image_processor.apply_individual(
                plate_image,
                blur=preprocess_options["blur"],
                perspective=preprocess_options["perspective"],
                normalize=preprocess_options["normalize"]
            )
            
            # OCR 수행
            plate_text, confidence = ocr_engine.recognize_with_confidence(
                processed_plate,
                min_confidence=settings["min_ocr_confidence"]
            )
            
            # 한국 번호판 형식에 맞게 후처리
            formatted_text = ocr_engine.post_processor.format_korean_license_plate(plate_text)
            
            # 결과 저장
            results.append({
                "vehicle_box": vehicle_box,
                "plate_box": plate_box,
                "plate_text": formatted_text,
                "original_text": plate_text,
                "confidence": confidence,
                "plate_image": plate_image,
                "processed_plate": processed_plate
            })
            
            # 처리된 번호판 이미지 저장
            plate_image_key = f"번호판 (차량 {len(results)})"
            processed_images[plate_image_key] = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
            
            processed_plate_key = f"처리된 번호판 (차량 {len(results)})"
            processed_images[processed_plate_key] = processed_plate
    
    # 전체 이미지에 결과 시각화
    if results:
        result_image = visualize_results(original_image, results)
        processed_images["최종 결과"] = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    return results, processed_images