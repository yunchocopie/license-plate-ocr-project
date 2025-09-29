import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
import os
import atexit
import pandas as pd
import json
import tempfile
import zipfile
from pathlib import Path

# 내부 모듈 임포트
from src.detection.vehicle_detector import VehicleDetector
from src.detection.plate_detector import PlateDetector
from src.preprocessing.image_processor import ImageProcessor
from src.ocr.ocr_engine import OCREngine
from src.utils.visualization import visualize_results
from src.utils.system_optimizer import SystemOptimizer
from src.batch.batch_processor import BatchProcessor
from src.utils.single_excel_helper import create_single_result_excel, OPENPYXL_AVAILABLE
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
    
    # 감지 모드 (자동 감지 모드 고정)
    detection_mode = "자동 감지 모드(권장)"

    # 전처리 모드 (자동으로 설정)
    preprocessing_mode = "auto"
    
    # 전처리 상세 옵션 (고급 사용자용)
    with st.sidebar.expander("고급 전처리 옵션"):
        show_preprocessing_info = st.checkbox("전처리 정보 표시", value=False)
        show_preprocessing_steps = st.checkbox("처리 단계 표시", value=False)
        show_realtime_progress = st.checkbox("실시간 처리 과정 표시", value=True)
        show_step_images = st.checkbox("단계별 이미지 표시", value=True)
    
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

    # 통합 파일 업로더
    st.subheader("📁 파일 업로드")
    uploaded_files = st.file_uploader(
        "파일을 업로드하세요 (이미지 또는 ZIP 파일)",
        type=["jpg", "jpeg", "png", "zip"],
        accept_multiple_files=True,
        help="단일 이미지, 여러 이미지, 또는 ZIP 파일을 업로드할 수 있습니다."
    )

    # 처리 시작 버튼 및 파일 처리
    if uploaded_files:
        st.success(f"총 {len(uploaded_files)}개의 파일이 업로드되었습니다.")

        if st.button("🚀 처리 시작", type="primary"):
            results = process_uploaded_files(uploaded_files, vehicle_detector, plate_detector, image_processor, ocr_engine, detection_mode, preprocessing_mode, show_preprocessing_info, show_preprocessing_steps, system_optimizer, show_realtime_progress, show_step_images)

            # 단일 이미지 처리 결과에 대한 Excel 다운로드 버튼
            if results and results.get('type') != 'batch' and OPENPYXL_AVAILABLE:
                st.subheader("💾 결과 다운로드")

                if st.button("📈 Excel 보고서 다운로드"):
                    with st.spinner("Excel 파일 생성 중..."):
                        # 임시 이미지 파일 저장
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                            tmp_file.write(uploaded_files[0].getvalue())
                            temp_image_path = tmp_file.name

                        # Excel 파일 생성
                        excel_file = create_single_result_excel(
                            temp_image_path,
                            results.get('plates', []),
                            "single_result.xlsx",
                            include_image=True,
                            processing_time=results.get('processing_time', 0),
                            detection_mode=detection_mode
                        )

                        if excel_file and os.path.exists(excel_file):
                            with open(excel_file, 'rb') as f:
                                excel_data = f.read()

                            st.download_button(
                                label="📊 Excel 보고서 다운로드",
                                data=excel_data,
                                file_name=f"plate_analysis_{int(time.time())}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                            # 임시 파일 정리
                            try:
                                os.unlink(temp_image_path)
                                os.unlink(excel_file)
                            except:
                                pass
                        else:
                            st.error("Excel 파일 생성에 실패했습니다.")

def analyze_uploaded_files(uploaded_files):
    """업로드된 파일들을 분석하여 처리 방식 결정"""
    image_files = []
    zip_files = []

    for file in uploaded_files:
        if file.name.lower().endswith('.zip'):
            zip_files.append(file)
        elif file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(file)

    # ZIP 파일에서 이미지 추출
    extracted_images = []
    for zip_file in zip_files:
        extracted = extract_images_from_zip(zip_file)
        extracted_images.extend(extracted)

    # 모든 이미지 파일 통합
    all_images = image_files + extracted_images

    # 처리 방식 결정
    if len(all_images) == 1:
        return "single", all_images
    elif len(all_images) > 1:
        return "batch", all_images
    else:
        return "none", []

def process_uploaded_files(uploaded_files, vehicle_detector, plate_detector, image_processor, ocr_engine, detection_mode, preprocessing_mode='auto', show_preprocessing_info=False, show_preprocessing_steps=False, system_optimizer=None, show_realtime_progress=True, show_step_images=True):
    """업로드된 파일들을 자동 분석하여 적절한 방식으로 처리"""

    # 파일 분석
    process_type, image_files = analyze_uploaded_files(uploaded_files)

    if process_type == "none":
        st.error("처리할 수 있는 이미지 파일이 없습니다.")
        return None

    elif process_type == "single":
        # 단일 이미지 처리
        st.info("단일 이미지 처리 모드")
        return process_single_image(image_files[0], vehicle_detector, plate_detector, image_processor, ocr_engine, detection_mode, preprocessing_mode, show_preprocessing_info, show_preprocessing_steps, system_optimizer, show_realtime_progress, show_step_images)

    else:  # batch
        # 배치 처리
        st.info(f"배치 처리 모드 ({len(image_files)}개 이미지)")
        return process_batch_files(image_files, vehicle_detector, plate_detector, image_processor, ocr_engine, detection_mode, preprocessing_mode)

def process_batch_files(image_files, vehicle_detector, plate_detector, image_processor, ocr_engine, detection_mode, preprocessing_mode):
    """간단한 배치 처리 (기존 배치 처리 로직 단순화)"""

    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"

        input_dir.mkdir()
        output_dir.mkdir()

        # 업로드된 파일들을 임시 디렉토리에 저장
        for i, file in enumerate(image_files):
            safe_name = getattr(file, 'name', f'image_{i:04d}.jpg')
            safe_name = safe_name.replace('/', '_').replace('\\', '_')
            file_path = input_dir / f"image_{i:04d}_{safe_name}"
            with open(file_path, 'wb') as f:
                f.write(file.getvalue())

        # 배치 프로세서 설정 및 실행
        processor = BatchProcessor(max_workers=4, use_multiprocessing=False)
        processor.configure_processing(
            detection_mode=detection_mode,
            preprocessing_mode=preprocessing_mode,
            min_confidence=0.3,
            chunk_size=10
        )

        # 진행률 표시
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(processed: int, total: int, data: dict):
            progress = processed / total if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(f"처리 중... {processed}/{total} ({progress*100:.1f}%)")

        processor.set_progress_callback(progress_callback)

        try:
            start_time = time.time()
            summary = processor.process_directory(str(input_dir), output_dir=str(output_dir))
            end_time = time.time()

            # 결과 표시
            display_batch_results(summary, end_time - start_time, output_dir)

            return {
                'type': 'batch',
                'summary': summary,
                'processing_time': end_time - start_time,
                'success': summary.processed_files > 0
            }

        except Exception as e:
            st.error(f"배치 처리 중 오류 발생: {str(e)}")
            return None

def process_single_image(image_file, vehicle_detector, plate_detector, image_processor, ocr_engine, detection_mode, preprocessing_mode='auto', show_preprocessing_info=False, show_preprocessing_steps=False, system_optimizer=None, show_realtime_progress=True, show_step_images=True):

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

    # 실시간 진행 상황 표시
    if show_realtime_progress:
        st.subheader("처리 과정")
        status_text = st.empty()
        progress_bar = st.progress(0)

        # 진행 상태 업데이트 함수
        def update_progress(step, total_steps, message):
            progress = step / total_steps
            progress_bar.progress(progress)
            status_text.text(f"{message}")

        update_progress(1, 6, "이미지 분석 시작")

    # 자동 감지 모드 처리
    # 1단계: 직접 번호판 감지
    if show_realtime_progress:
        update_progress(2, 6, "번호판 검출 중")

    plate_boxes = plate_detector.detect(image_np)

    if plate_boxes:  # 번호판이 감지되면
        if show_realtime_progress:
            update_progress(3, 6, f"{len(plate_boxes)}개의 번호판 발견")

        for i, plate_box in enumerate(plate_boxes):
            # 번호판 영역 추출
            plate_image = image_np[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]

            # 번호판 전처리
            if show_realtime_progress:
                update_progress(4, 6, f"번호판 {i+1} 이미지 전처리")

            # 전처리 단계별 이미지 생성 (표시용)
            preprocessing_steps = None
            if show_step_images:
                preprocessing_steps = image_processor.visualize_steps(plate_image)

            processed_plate = image_processor.process(plate_image)

            # OCR 처리 및 분류 (고급 전처리 적용)
            if show_realtime_progress:
                update_progress(5, 6, f"번호판 {i+1} OCR 인식")

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
                "preprocessing_info": preprocessing_info,
                "preprocessing_steps": preprocessing_steps  # 전처리 단계 이미지들
            })
    else:  # 번호판이 직접 감지되지 않으면 차량 감지 후 번호판 감지 시도
        # 차량 감지
        if show_realtime_progress:
            update_progress(2, 6, "차량 검출 중")

        vehicle_boxes = vehicle_detector.detect(image_np)

        if vehicle_boxes:
            if show_realtime_progress:
                update_progress(3, 6, f"{len(vehicle_boxes)}개의 차량 발견")

        for v_idx, vehicle_box in enumerate(vehicle_boxes):
            # 차량 영역 추출
            vehicle_image = image_np[vehicle_box[1]:vehicle_box[3], vehicle_box[0]:vehicle_box[2]]

            # 번호판 감지
            if show_realtime_progress:
                update_progress(3, 6, f"차량 {v_idx+1}에서 번호판 검출")

            plate_boxes = plate_detector.detect(vehicle_image)

            for p_idx, plate_box in enumerate(plate_boxes):
                # 번호판 영역 추출
                plate_image = vehicle_image[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]

                # 번호판 전처리
                if show_realtime_progress:
                    update_progress(4, 6, f"차량 {v_idx+1}의 번호판 {p_idx+1} 이미지 전처리")

                # 전처리 단계별 이미지 생성 (표시용)
                preprocessing_steps = None
                if show_step_images:
                    preprocessing_steps = image_processor.visualize_steps(plate_image)

                processed_plate = image_processor.process(plate_image)

                # OCR 처리 및 분류
                if show_realtime_progress:
                    update_progress(5, 6, f"차량 {v_idx+1} 번호판 {p_idx+1} OCR 인식")

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
                    "preprocessing_info": preprocessing_info,
                    "preprocessing_steps": preprocessing_steps  # 전처리 단계 이미지들
                })
    
    # 처리 종료 시간
    end_time = time.time()
    processing_time = end_time - start_time

    # 처리 완료 표시
    if show_realtime_progress:
        if results:
            update_progress(6, 6, f"처리 완료 - {len(results)}개의 번호판 인식")
        else:
            update_progress(6, 6, "처리 완료 - 번호판을 찾을 수 없음")

    # 감지 결과 표시
    if results:
        # 처리 결과 요약
        st.subheader("처리 결과")

        # 1. 원본 이미지 먼저 표시
        st.write("**1. 원본 이미지**")
        st.image(image, use_column_width=True)

        # 2. 전처리 단계별 이미지 표시 (각 번호판별로)
        if show_step_images and results:
            for idx, result in enumerate(results):
                preprocessing_steps = result.get('preprocessing_steps', {})
                if preprocessing_steps:
                    st.write(f"**2. 번호판 {idx+1} 전처리 단계**")

                    # 전체 전처리 단계를 순서대로 표시
                    step_sequence = [
                        ('original', '원본 (추출)'),
                        ('gray', '그레이스케일'),
                        ('perspective_corrected', '원근 보정'),
                        ('denoised', '노이즈 제거'),
                        ('normalized', '크기 정규화'),
                        ('enhanced_clahe', '대비 향상'),
                        ('final_easyocr_input', '최종 OCR 입력')
                    ]

                    # 2행으로 배치 (위: 4개, 아래: 3개)
                    # 첫 번째 행 (4개)
                    cols_row1 = st.columns(4)
                    for i, (key, name) in enumerate(step_sequence[:4]):
                        if key in preprocessing_steps:
                            with cols_row1[i]:
                                st.caption(name)
                                st.image(preprocessing_steps[key], use_column_width=True)

                    # 두 번째 행 (3개)
                    cols_row2 = st.columns([1, 1, 1, 1])  # 4개 컬럼 중 3개만 사용
                    for i, (key, name) in enumerate(step_sequence[4:]):
                        if key in preprocessing_steps:
                            with cols_row2[i]:
                                st.caption(name)
                                st.image(preprocessing_steps[key], use_column_width=True)

                    st.write("---")

        # 3. 최종 감지 결과 이미지
        visualized_image = visualize_results(image_np, results)
        st.write("**3. 최종 감지 결과**")
        st.image(visualized_image, use_column_width=True)

        # 4. 인식된 번호판 정보
        st.write("---")
        st.write(f"**🚗 인식 결과: {len(results)}개의 번호판**")

        for idx, result in enumerate(results):
            with st.container():
                # 메인 정보만 간단히 표시
                col1, col2 = st.columns([3, 1])

                with col1:
                    # 번호판 텍스트를 크게 표시
                    st.markdown(f"### 번호판 {idx+1}: `{result['plate_text']}`")

                    # 번호판 타입과 유효성
                    if 'classification' in result:
                        plate_type = result['classification']['type'].value
                        st.write(f"**타입**: {plate_type}")

                    if 'validation' in result:
                        validation = result['validation']
                        if validation['is_valid']:
                            st.success("✅ 유효한 번호판 형식")
                        else:
                            st.error(f"❌ 형식 오류: {', '.join(validation['errors'])}")

                with col2:
                    # 신뢰도 정보
                    if 'confidence' in result:
                        st.metric("OCR 신뢰도", f"{result['confidence']:.1f}")
                    if 'classification' in result:
                        st.metric("분류 신뢰도", f"{result['classification']['confidence']:.1f}")

                if idx < len(results) - 1:  # 마지막이 아니면 구분선
                    st.write("---")

        # 상세 성능 정보 (접힘 메뉴)
        with st.expander("📊 상세 성능 정보"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("처리 시간", f"{processing_time:.2f}초")
            with col2:
                st.metric("전처리 모드", "자동 최적화")
            with col3:
                avg_confidence = sum([r['confidence'] for r in results]) / len(results)
                st.metric("평균 OCR 신뢰도", f"{avg_confidence:.1f}")
            with col4:
                if hasattr(plate_detector, 'get_performance_stats'):
                    perf_stats = plate_detector.get_performance_stats()
                    estimated_fps = perf_stats.get('estimated_fps', 0)
                    st.metric("예상 FPS", f"{estimated_fps:.1f}")
    else:
        # 번호판을 찾지 못한 경우
        st.subheader("처리 결과")
        st.image(image, use_column_width=True, caption="원본 이미지")
        st.warning("❌ 번호판을 찾을 수 없습니다")

        # 간단한 처리 정보만 표시
        st.info(f"처리 시간: {processing_time:.2f}초")
    
    # 고급 상세 정보 (매우 선택적)
    if show_preprocessing_info and results:
        with st.expander("🔧 고급 시스템 정보"):
            if system_optimizer:
                system_report = system_optimizer.get_system_report()
                resources = system_report['system_resources']

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**시스템 리소스**")
                    st.write(f"CPU: {resources['cpu']['current_usage']:.1f}%")
                    st.write(f"메모리: {resources['memory']['current_usage_percent']:.1f}%")

                with col2:
                    st.write("**처리 모드**")
                    profile = system_report['current_profile']
                    st.write(f"현재: {profile['name']}")
                    st.write(f"CPU 코어: {profile['cpu_cores_used']}개")
    
    # 처리 결과 반환 (Excel 출력용)
    return {
        'plates': results,
        'processing_time': end_time - start_time,
        'detection_mode': detection_mode,
        'total_plates': len(results),
        'success': len(results) > 0
    }


def extract_images_from_zip(zip_file):
    """ZIP 파일에서 이미지 파일들을 추출"""
    extracted_files = []
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # 파일을 메모리에서 직접 읽기
                    file_data = zip_ref.read(file_info.filename)
                    
                    # Streamlit의 UploadedFile과 유사한 객체 생성
                    class ZipImageFile:
                        def __init__(self, data, name):
                            self.data = data
                            self.name = name
                            
                        def read(self):
                            return self.data
                            
                        def getvalue(self):
                            return self.data
                    
                    extracted_files.append(ZipImageFile(file_data, file_info.filename))
        
        st.success(f"ZIP 파일에서 {len(extracted_files)}개의 이미지를 추출했습니다.")
        
    except Exception as e:
        st.error(f"ZIP 파일 처리 중 오류 발생: {str(e)}")
        return []
        
    return extracted_files


def display_batch_results(summary, processing_time, output_dir):
    """배치 처리 결과 표시"""
    st.success("🎉 배치 처리가 완료되었습니다!")
    
    # 처리 통계
    st.subheader("📊 처리 결과")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 처리 파일", f"{summary.processed_files}")
        
    with col2:
        st.metric("성공률", f"{summary.success_rate_percent:.1f}%")
        
    with col3:
        st.metric("처리 시간", f"{processing_time:.1f}초")
        
    with col4:
        st.metric("처리 속도", f"{summary.throughput_files_per_min:.1f}파일/분")
    
    if summary.total_plates_detected > 0:
        st.subheader("🚗 감지된 번호판 정보")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("총 감지된 번호판", f"{summary.total_plates_detected}개")
            st.metric("유효한 번호판", f"{summary.total_valid_plates}개")
            
        with col2:
            if summary.plate_type_distribution:
                st.write("**번호판 타입 분포:**")
                for plate_type, count in sorted(summary.plate_type_distribution.items()):
                    st.write(f"• {plate_type}: {count}개")
    
    # 결과 다운로드
    st.subheader("💾 결과 다운로드")
    
    # 결과 파일들 확인
    json_file = Path(output_dir) / "batch_results_detailed.json"
    csv_file = Path(output_dir) / "batch_results_summary.csv"
    
    # Excel 파일 찾기 (세션 ID가 포함된 파일명)
    excel_files = list(Path(output_dir).glob("batch_results_comprehensive_*.xlsx"))
    excel_file = excel_files[0] if excel_files else None
    
    # 다운로드 버튼 배치 (Excel이 있으면 3열, 없으면 2열)
    if excel_file:
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2 = st.columns(2)
    
    if json_file.exists():
        with col1:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = f.read()
            
            st.download_button(
                label="📄 상세 결과 (JSON)",
                data=json_data,
                file_name=f"batch_results_{int(time.time())}.json",
                mime="application/json"
            )
    
    if csv_file.exists():
        with col2:
            with open(csv_file, 'r', encoding='utf-8') as f:
                csv_data = f.read()
                
            st.download_button(
                label="📊 요약 결과 (CSV)",
                data=csv_data,
                file_name=f"batch_summary_{int(time.time())}.csv",
                mime="text/csv"
            )
    
    # Excel 파일 다운로드
    if excel_file and excel_file.exists():
        with col3:
            with open(excel_file, 'rb') as f:
                excel_data = f.read()
                
            st.download_button(
                label="📈 종합 보고서 (Excel)",
                data=excel_data,
                file_name=f"comprehensive_report_{int(time.time())}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # 실패한 파일 목록 표시
    if summary.failed_files > 0:
        with st.expander(f"⚠️ 실패한 파일 ({summary.failed_files}개)"):
            failed_file = Path(output_dir) / "batch_failed_files.json"
            if failed_file.exists():
                with open(failed_file, 'r', encoding='utf-8') as f:
                    failed_data = json.load(f)
                
                for item in failed_data[:10]:  # 최대 10개만 표시
                    st.write(f"• {item['file_name']}: {item['error']}")
                
                if len(failed_data) > 10:
                    st.write(f"... 및 {len(failed_data) - 10}개 더")


if __name__ == "__main__":
    main()