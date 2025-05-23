import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
import os

# 내부 모듈 임포트
from src.detection.vehicle_detector import VehicleDetector
from src.detection.plate_detector import PlateDetector
from src.preprocessing.image_processor import ImageProcessor
from src.ocr.ocr_engine import OCREngine
from src.utils.visualization import visualize_results
import config


def main():
    st.set_page_config(
        page_title="차량번호 OCR 프로그램",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("차량번호 OCR 프로그램")
    
    # 사이드바 설정
    st.sidebar.title("설정")
    input_type = st.sidebar.radio("입력 유형", ["이미지 업로드", "카메라 촬영", "비디오 업로드"])
    
    # 감지 모드 선택
    detection_mode = st.sidebar.radio("감지 모드", [
        "차량 감지 후 번호판 감지", 
        "직접 번호판 감지",
        "자동 감지 모드(권장)"
    ], index=2)
    
    # 모델 초기화
    vehicle_detector = VehicleDetector()
    plate_detector = PlateDetector()
    image_processor = ImageProcessor()
    ocr_engine = OCREngine()
    
    if input_type == "이미지 업로드":
        # 이미지 업로드 로직
        uploaded_file = st.sidebar.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
        
    if uploaded_file is not None:
            # 이미지 처리 및 결과 표시
            process_image(uploaded_file, vehicle_detector, plate_detector, image_processor, ocr_engine, detection_mode)
            
    elif input_type == "카메라 촬영":
        # 카메라 촬영 로직
        camera_input = st.camera_input("사진 촬영")
        
        if camera_input is not None:
            # 이미지 처리 및 결과 표시
            process_image(camera_input, vehicle_detector, plate_detector, image_processor, ocr_engine, detection_mode)
            
    elif input_type == "비디오 업로드":
        # 비디오 업로드 로직
        video_file = st.sidebar.file_uploader("비디오 업로드", type=["mp4", "avi", "mov"])
        
        if video_file is not None:
            # 비디오 처리 및 결과 표시
            st.warning("비디오 처리 기능은 현재 개발 중입니다.")
            #process_video(video_file, vehicle_detector, plate_detector, image_processor, ocr_engine)

def process_image(image_file, vehicle_detector, plate_detector, image_processor, ocr_engine, detection_mode):
    # 이미지 로드 및 처리
    image = Image.open(image_file)
    image_np = np.array(image)
    
    # 처리 시작 시간
    start_time = time.time()
    
    results = []
    
    if detection_mode == "차량 감지 후 번호판 감지":
        # 차량 감지
        vehicle_boxes = vehicle_detector.detect(image_np)
        
        for vehicle_box in vehicle_boxes:
            # 차량 영역 추출
            vehicle_image = image_np[vehicle_box[1]:vehicle_box[3], vehicle_box[0]:vehicle_box[2]]
            
            # 번호판 감지
            plate_boxes = plate_detector.detect(vehicle_image)
            
            for plate_box in plate_boxes:
                # 번호판 영역 추출
                plate_image = vehicle_image[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]
                
                # 번호판 전처리
                processed_plate = image_processor.process(plate_image)
                
                # OCR 처리
                plate_text = ocr_engine.recognize(processed_plate)
                
                # 결과 저장
                global_plate_box = [
                    vehicle_box[0] + plate_box[0], 
                    vehicle_box[1] + plate_box[1],
                    vehicle_box[0] + plate_box[2], 
                    vehicle_box[1] + plate_box[3]
                ]
                
                results.append({
                    "vehicle_box": vehicle_box,
                    "plate_box": global_plate_box,
                    "plate_text": plate_text
                })
    
    elif detection_mode == "직접 번호판 감지":
        # 이미지에서 직접 번호판 감지
        plate_boxes = plate_detector.detect(image_np)
        
        for plate_box in plate_boxes:
            # 번호판 영역 추출
            plate_image = image_np[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]
            
            # 번호판 전처리
            processed_plate = image_processor.process(plate_image)
            
            # OCR 처리
            plate_text = ocr_engine.recognize(processed_plate)
            
            # 결과 저장
            results.append({
                "vehicle_box": None,  # 차량 박스 정보 없음
                "plate_box": plate_box,
                "plate_text": plate_text
            })
    
    else:  # 자동 감지 모드
        # 1단계: 직접 번호판 감지
        plate_boxes = plate_detector.detect(image_np)
        
        if plate_boxes:  # 번호판이 감지되면
            for plate_box in plate_boxes:
                # 번호판 영역 추출
                plate_image = image_np[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]
                
                # 번호판 전처리
                processed_plate = image_processor.process(plate_image)
                
                # OCR 처리
                plate_text = ocr_engine.recognize(processed_plate)
                
                # 결과 저장
                results.append({
                    "vehicle_box": None,  # 차량 박스 정보 없음
                    "plate_box": plate_box,
                    "plate_text": plate_text
                })
        else:  # 번호판이 직접 감지되지 않으면 차량 감지 후 번호판 감지 시도
            # 차량 감지
            vehicle_boxes = vehicle_detector.detect(image_np)
            
            for vehicle_box in vehicle_boxes:
                # 차량 영역 추출
                vehicle_image = image_np[vehicle_box[1]:vehicle_box[3], vehicle_box[0]:vehicle_box[2]]
                
                # 번호판 감지
                plate_boxes = plate_detector.detect(vehicle_image)
                
                for plate_box in plate_boxes:
                    # 번호판 영역 추출
                    plate_image = vehicle_image[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]
                    
                    # 번호판 전처리
                    processed_plate = image_processor.process(plate_image)
                    
                    # OCR 처리
                    plate_text = ocr_engine.recognize(processed_plate)
                    
                    # 결과 저장
                    global_plate_box = [
                        vehicle_box[0] + plate_box[0], 
                        vehicle_box[1] + plate_box[1],
                        vehicle_box[0] + plate_box[2], 
                        vehicle_box[1] + plate_box[3]
                    ]
                    
                    results.append({
                        "vehicle_box": vehicle_box,
                        "plate_box": global_plate_box,
                        "plate_text": plate_text
                    })
    
    # 처리 종료 시간
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 감지 결과 표시
    if results:
        # 결과 시각화
        visualized_image = visualize_results(image_np, results)
        
        # 결과 표시
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("원본 이미지")
            st.image(image, use_column_width=True)
        with col2:
            st.subheader("처리 결과")
            st.image(visualized_image, use_column_width=True)
        
        # 인식된 번호판 표시
        st.subheader("인식된 번호판")
        for idx, result in enumerate(results):
            st.write(f"번호판 {idx+1}: {result['plate_text']}")
        
        st.success(f"{len(results)}개의 번호판이 감지되었습니다.")
    else:
        # 결과 표시
        st.subheader("원본 이미지")
        st.image(image, use_column_width=True)
        st.warning("번호판이 감지되지 않았습니다.")
    
    st.write(f"처리 시간: {processing_time:.2f}초")

# def process_video(video_file, vehicle_detector, plate_detector, image_processor, ocr_engine):
#     # 비디오 처리 로직
#     # 생략...

if __name__ == "__main__":
    main()