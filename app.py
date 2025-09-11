import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
import os
import atexit

# 내부 모듈 임포트
from src.detection.vehicle_detector import VehicleDetector
from src.detection.plate_detector import PlateDetector
from src.preprocessing.image_processor import ImageProcessor
from src.ocr.ocr_engine import OCREngine
from src.utils.visualization import visualize_results
from src.utils.system_optimizer import SystemOptimizer
import config

# 전역 시스템 최적화기
@st.cache_resource
def get_system_optimizer():
    optimizer = SystemOptimizer(monitoring_interval=10.0)  # 10초 간격
    optimizer.start_monitoring()
    
    # 앱 종료 시 모니터링 중지
    def cleanup():
        optimizer.stop_monitoring()
    atexit.register(cleanup)
    
    return optimizer


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
    
    # 전처리 모드 선택
    preprocessing_mode = st.sidebar.selectbox("이미지 전처리 모드", [
        "auto", "high_quality", "balanced", "fast", "off"
    ], index=0, help="auto: 자동 최적화, high_quality: 최고품질, balanced: 균형, fast: 빠른처리, off: 전처리 비활성화")
    
    # 전처리 상세 옵션 (고급 사용자용)
    with st.sidebar.expander("고급 전처리 옵션"):
        show_preprocessing_info = st.checkbox("전처리 정보 표시", value=False)
        show_preprocessing_steps = st.checkbox("처리 단계 표시", value=False)
    
    # 시스템 최적화기 초기화 (전역 캐시 사용)
    system_optimizer = get_system_optimizer()
    
    # 모델 초기화 (한국 번호판 최적화 적용)
    vehicle_detector = VehicleDetector()
    from src.detection.plate_detector import create_optimized_plate_detector
    plate_detector = create_optimized_plate_detector()  # 한국 번호판 최적화 버전
    image_processor = ImageProcessor()
    ocr_engine = OCREngine()
    
    # 시스템 최적화 대시보드
    with st.sidebar.expander("🔧 시스템 최적화", expanded=False):
        system_report = system_optimizer.get_system_report()
        
        # 현재 프로파일 표시
        current_profile = system_report['current_profile']
        st.info(f"**현재 모드**: {current_profile['name']} ({current_profile['level']})")
        
        # 프로파일 변경 옵션
        profile_options = ['conservative', 'balanced', 'maximum']
        current_idx = profile_options.index(current_profile['level'])
        new_profile = st.selectbox(
            "성능 모드 변경",
            profile_options,
            index=current_idx,
            format_func=lambda x: {
                'conservative': '절약 모드 (저사양)',
                'balanced': '균형 모드 (권장)', 
                'maximum': '최고 성능 (고사양)'
            }[x]
        )
        
        if new_profile != current_profile['level']:
            if st.button("모드 변경 적용"):
                system_optimizer.switch_profile(new_profile)
                st.rerun()
        
        # 리소스 상태 표시
        resources = system_report['system_resources']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU", f"{resources['cpu']['current_usage']:.1f}%")
            st.metric("메모리", f"{resources['memory']['current_usage_percent']:.1f}%")
            
        with col2:
            if resources['gpu']['available']:
                st.metric("GPU 메모리", f"{resources['gpu']['usage_percent']:.1f}%")
            else:
                st.metric("GPU", "사용불가")
            st.metric("디스크", f"{resources['disk']['available_gb']:.1f}GB")
        
        # 최적화 권장사항
        recommendations = system_report.get('recommendations', [])
        if recommendations:
            st.write("**권장사항:**")
            for rec in recommendations[:3]:  # 최대 3개만 표시
                st.write(f"• {rec}")
    
    # 기존 번호판 검출기 정보
    if hasattr(plate_detector, 'get_system_recommendations'):
        with st.sidebar.expander("📊 검출 성능 정보"):
            recommendations = plate_detector.get_system_recommendations()
            st.text(recommendations)
    
    if input_type == "이미지 업로드":
        # 이미지 업로드 로직
        uploaded_file = st.sidebar.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
        
    if uploaded_file is not None:
            # 이미지 처리 및 결과 표시
            process_image(uploaded_file, vehicle_detector, plate_detector, image_processor, ocr_engine, detection_mode, preprocessing_mode, show_preprocessing_info, show_preprocessing_steps)
            
    elif input_type == "카메라 촬영":
        # 카메라 촬영 로직
        camera_input = st.camera_input("사진 촬영")
        
        if camera_input is not None:
            # 이미지 처리 및 결과 표시
            process_image(camera_input, vehicle_detector, plate_detector, image_processor, ocr_engine, detection_mode, preprocessing_mode, show_preprocessing_info, show_preprocessing_steps)
            
    elif input_type == "비디오 업로드":
        # 비디오 업로드 로직
        video_file = st.sidebar.file_uploader("비디오 업로드", type=["mp4", "avi", "mov"])
        
        if video_file is not None:
            # 비디오 처리 및 결과 표시
            st.warning("비디오 처리 기능은 현재 개발 중입니다.")
            #process_video(video_file, vehicle_detector, plate_detector, image_processor, ocr_engine)

def process_image(image_file, vehicle_detector, plate_detector, image_processor, ocr_engine, detection_mode, preprocessing_mode='auto', show_preprocessing_info=False, show_preprocessing_steps=False):

    """
    A. 차량 감지 후 번호판 감지
        1. 전체 이미지에서 차량 찾기
        2. 차량 내부에서 번호판 찾기
        => 전체 이미지 -> 차량 탐지 -> 차량 이미지 안에서 번호판 탐지 -> OCR

    B. 직접 번호판 감지
        1. 바로 번호판만 찾기 (차량 잘림 없이 나온 사진이 아니어도 잘 동작 가능)
        => 전체 이미지 -> 번호판 탐지 -> OCR (차량 탐지 X)

    C. 자동 감지
        1. 전체 이미지에서 번호판 직접 찾기
        2. 번호판 감지 실패시 차량 감지 -> 번호판 감지
        => 번호판 감지 -> 실패시 차량 감지 -> 번호판 감지 -> OCR
    """

    # 이미지 로드 및 처리
    image = Image.open(image_file)
    image_np = np.array(image) # 입력한 이미지 배열 변환
    
    # 처리 시작 시간
    start_time = time.time()
    
    results = []

    # 감지 모드 분기 처리
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
                
                # OCR 처리 및 분류 (고급 전처리 적용)
                ocr_result = ocr_engine.recognize_with_classification(processed_plate, preprocessing_mode=preprocessing_mode)
                plate_text = ocr_result['text']
                confidence = ocr_result['confidence']
                classification = ocr_result['classification']
                validation = ocr_result.get('validation', {})
                preprocessing_info = ocr_result.get('preprocessing_info', {})
                
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
                    "plate_text": plate_text,
                    "confidence": confidence,
                    "classification": classification,
                    "validation": validation,
                    "preprocessing_info": preprocessing_info
                })
    
    elif detection_mode == "직접 번호판 감지":
        # 이미지에서 직접 번호판 감지
        plate_boxes = plate_detector.detect(image_np)
        
        for plate_box in plate_boxes:
            # 번호판 영역 추출
            plate_image = image_np[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]
            
            # 번호판 전처리
            processed_plate = image_processor.process(plate_image)
            
            # OCR 처리 및 분류
            ocr_result = ocr_engine.recognize_with_classification(processed_plate)
            plate_text = ocr_result['text']
            confidence = ocr_result['confidence']
            classification = ocr_result['classification']
            
            # 결과 저장
            results.append({
                "vehicle_box": None,  # 차량 박스 정보 없음
                "plate_box": plate_box,
                "plate_text": plate_text,
                "confidence": confidence,
                "classification": classification,
                "validation": validation,
                "preprocessing_info": preprocessing_info
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
                
                # OCR 처리 및 분류 (고급 전처리 적용)
                ocr_result = ocr_engine.recognize_with_classification(processed_plate, preprocessing_mode=preprocessing_mode)
                plate_text = ocr_result['text']
                confidence = ocr_result['confidence']
                classification = ocr_result['classification']
                validation = ocr_result.get('validation', {})
                preprocessing_info = ocr_result.get('preprocessing_info', {})
                
                # 결과 저장
                results.append({
                    "vehicle_box": None,  # 차량 박스 정보 없음
                    "plate_box": plate_box,
                    "plate_text": plate_text,
                    "confidence": confidence,
                    "classification": classification,
                    "validation": validation,
                    "preprocessing_info": preprocessing_info
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
                    
                    # OCR 처리 및 분류
                    ocr_result = ocr_engine.recognize_with_classification(processed_plate)
                    plate_text = ocr_result['text']
                    confidence = ocr_result['confidence']
                    classification = ocr_result['classification']
                    
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
                        "plate_text": plate_text,
                        "confidence": confidence,
                        "classification": classification,
                        "validation": validation,
                        "preprocessing_info": preprocessing_info
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
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**번호판 {idx+1}**: {result['plate_text']}")
                # 유효성 검사 결과 표시
                if 'validation' in result:
                    validation = result['validation']
                    if validation['is_valid']:
                        st.success("✅ 형식 유효")
                    else:
                        st.warning(f"⚠️ 형식 오류: {', '.join(validation['errors'])}")
            
            with col2:
                if 'classification' in result:
                    plate_type = result['classification']['type'].value
                    confidence = result['classification']['confidence']
                    bg_color = result['classification']['background_color']
                    st.write(f"**타입**: {plate_type}")
                    st.write(f"**배경색**: {bg_color}")
                
            with col3:
                if 'confidence' in result:
                    st.write(f"**OCR 신뢰도**: {result['confidence']:.2f}")
                if 'classification' in result:
                    st.write(f"**분류 신뢰도**: {result['classification']['confidence']:.2f}")
        
        st.success(f"{len(results)}개의 번호판이 감지되었습니다.")
        
        # 전처리 정보 표시
        if show_preprocessing_info and results:
            st.subheader("이미지 전처리 정보")
            for idx, result in enumerate(results):
                preprocessing_info = result.get('preprocessing_info', {})
                if preprocessing_info:
                    with st.expander(f"번호판 {idx+1} 전처리 상세"):
                        st.write(f"**전처리 모드**: {preprocessing_info.get('mode', 'unknown')}")
                        
                        # 품질 메트릭 표시
                        quality_metrics = preprocessing_info.get('quality_metrics', {})
                        if quality_metrics:
                            st.write("**품질 메트릭**:")
                            for metric_name, metric_value in quality_metrics.items():
                                if isinstance(metric_value, (int, float)):
                                    st.write(f"  - {metric_name}: {metric_value:.3f}")
                        
                        # 처리 단계 표시
                        if show_preprocessing_steps:
                            processing_steps = preprocessing_info.get('processing_steps', [])
                            if processing_steps:
                                st.write("**처리 단계**:")
                                for step in processing_steps:
                                    st.write(f"  - {step}")
    else:
        # 결과 표시
        st.subheader("원본 이미지")
        st.image(image, use_column_width=True)
        st.warning("번호판이 감지되지 않았습니다.")
    
    # 성능 정보 표시
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("처리 시간", f"{processing_time:.2f}초")
    with col2:
        st.metric("전처리 모드", preprocessing_mode)
    with col3:
        if results:
            avg_confidence = sum([r['confidence'] for r in results]) / len(results)
            st.metric("평균 OCR 신뢰도", f"{avg_confidence:.2f}")
        else:
            st.metric("평균 OCR 신뢰도", "N/A")
    with col4:
        # 번호판 검출 성능 정보
        if hasattr(plate_detector, 'get_performance_stats'):
            perf_stats = plate_detector.get_performance_stats()
            estimated_fps = perf_stats.get('estimated_fps', 0)
            st.metric("예상 FPS", f"{estimated_fps:.1f}")
        else:
            st.metric("예상 FPS", "N/A")
    
    # 상세 성능 정보 (선택적 표시)
    if show_preprocessing_info:
        # 시스템 최적화 상세 정보
        with st.expander("🔧 시스템 최적화 상세"):
            system_report = system_optimizer.get_system_report()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**시스템 리소스**")
                resources = system_report['system_resources']
                st.write(f"CPU: {resources['cpu']['current_usage']:.1f}% ({resources['cpu']['total_cores']}코어)")
                st.write(f"메모리: {resources['memory']['available_mb']:.0f}MB / {resources['memory']['total_mb']:.0f}MB")
                if resources['gpu']['available']:
                    st.write(f"GPU: {resources['gpu']['memory_used_mb']:.0f}MB / {resources['gpu']['memory_total_mb']:.0f}MB")
                
            with col2:
                st.write("**성능 프로파일**")
                profile = system_report['current_profile']
                st.write(f"모드: {profile['name']}")
                st.write(f"CPU 코어: {profile['cpu_cores_used']}개")
                st.write(f"메모리 한도: {profile['max_memory_mb']}MB")
                st.write(f"우선순위: {profile['priority']}")
        
        # 번호판 검출 성능
        if hasattr(plate_detector, 'get_performance_stats'):
            st.subheader("검출 성능 상세")
            perf_stats = plate_detector.get_performance_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**검출 통계**")
                st.write(f"총 검출 수: {perf_stats.get('total_detections', 0)}")
                st.write(f"평균 추론 시간: {perf_stats.get('avg_inference_time', 0):.3f}초")
                
            with col2:
                optimization_info = perf_stats.get('optimization_info', {})
                if optimization_info.get('korean_optimization', False):
                    st.write("**최적화 정보**")
                    st.write("✅ 한국 번호판 최적화 활성화")
                    optimal_config = optimization_info.get('optimal_config', {})
                    if optimal_config:
                        st.write(f"모델 크기: {optimal_config.get('model_size', 'N/A')}")
                        st.write(f"배치 크기: {optimal_config.get('batch_size', 'N/A')}")
                        st.write(f"이미지 크기: {optimal_config.get('imgsz', 'N/A')}")
                else:
                    st.write("**최적화 정보**")
                    st.write("⚠️ 기본 모드 (최적화 미적용)")

# def process_video(video_file, vehicle_detector, plate_detector, image_processor, ocr_engine):
#     # 비디오 처리 로직
#     # 생략...

if __name__ == "__main__":
    main()