import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import time
from ..components import create_sidebar, create_header
from src.detection.vehicle_detector import VehicleDetector
from src.detection.plate_detector import PlateDetector
from src.preprocessing.image_processor import ImageProcessor
from src.ocr.ocr_engine import OCREngine
from src.utils.metrics import calculate_text_similarity, benchmark_pipeline
from src.utils.visualization import visualize_preprocessing_steps
import config

"""
분석 페이지 모듈

이 모듈은 Streamlit 애플리케이션의 분석 페이지를 렌더링합니다.
"""
def render_analysis_page():
    """
    분석 페이지 렌더링
    """
    # 헤더 생성
    st.title("번호판 인식 분석")
    
    # 사이드바 생성
    settings = create_sidebar()
    
    # 탭 생성
    tabs = st.tabs(["전처리 분석", "OCR 성능 테스트", "배치 처리", "통계"])
    
    # 전처리 분석 탭
    with tabs[0]:
        render_preprocessing_tab(settings)
    
    # OCR 성능 테스트 탭
    with tabs[1]:
        render_ocr_test_tab(settings)
    
    # 배치 처리 탭
    with tabs[2]:
        render_batch_processing_tab(settings)
    
    # 통계 탭
    with tabs[3]:
        render_statistics_tab()

def render_preprocessing_tab(settings):
    """
    전처리 분석 탭 렌더링
    
    Args:
        settings (dict): 설정 값
    """
    st.header("이미지 전처리 분석")
    
    # 이미지 업로드
    uploaded_file = st.file_uploader(
        "번호판 이미지를 업로드하세요",
        type=["jpg", "jpeg", "png", "bmp"],
        key="preprocess_uploader"
    )
    
    if uploaded_file is not None:
        # 이미지 로드
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # 이미지 표시
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="원본 이미지", use_column_width=True)
        
        # 전처리 옵션
        st.subheader("전처리 옵션")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            apply_blur = st.checkbox("흐림 보정", value=True)
        
        with col2:
            apply_perspective = st.checkbox("기울기 보정", value=True)
        
        with col3:
            apply_normalize = st.checkbox("비율 정규화", value=True)
        
        # 전처리 버튼
        if st.button("전처리 단계 분석", key="preprocess_button"):
            with st.spinner("이미지 전처리 중..."):
                # 이미지 처리기 초기화
                image_processor = ImageProcessor()
                
                # 전처리 단계별 시각화
                preprocessing_steps = image_processor.visualize_steps(image)
                
                # 결과 이미지 생성
                result_image = visualize_preprocessing_steps(preprocessing_steps)
                
                # 결과 이미지 표시
                st.image(result_image, caption="전처리 단계별 결과", use_column_width=True)
                
                # 선택된 전처리 적용
                processed_image = image_processor.apply_individual(
                    image,
                    blur=apply_blur,
                    perspective=apply_perspective,
                    normalize=apply_normalize
                )
                
                # 선택된 전처리 결과 표시
                st.subheader("선택된 전처리 결과")
                
                cols = st.columns(2)
                with cols[0]:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="원본", use_column_width=True)
                
                with cols[1]:
                    # 그레이스케일 이미지인 경우 처리
                    if len(processed_image.shape) == 2:
                        # 표시만을 위한 3채널 변환
                        display_img = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
                        st.image(display_img, caption="처리 결과", use_column_width=True)
                    else:
                        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="처리 결과", use_column_width=True)

def render_ocr_test_tab(settings):
    """
    OCR 성능 테스트 탭 렌더링
    
    Args:
        settings (dict): 설정 값
    """
    st.header("OCR 성능 테스트")
    
    # 이미지 업로드
    uploaded_file = st.file_uploader(
        "번호판 이미지를 업로드하세요",
        type=["jpg", "jpeg", "png", "bmp"],
        key="ocr_uploader"
    )
    
    # 실제 번호판 텍스트 입력
    ground_truth = st.text_input("실제 번호판 텍스트 (정답):", key="ocr_gt")
    
    # 전처리 옵션
    st.subheader("전처리 옵션")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        apply_blur = st.checkbox("흐림 보정", value=True, key="ocr_blur")
    
    with col2:
        apply_perspective = st.checkbox("기울기 보정", value=True, key="ocr_perspective")
    
    with col3:
        apply_normalize = st.checkbox("비율 정규화", value=True, key="ocr_normalize")
    
    # OCR 엔진 옵션
    st.subheader("OCR 엔진 옵션")
    
    ocr_confidence = st.slider(
        "OCR 최소 신뢰도 임계값",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        key="ocr_confidence"
    )
    
    use_gpu = st.checkbox("GPU 사용 (가능한 경우)", value=settings["use_gpu"], key="ocr_gpu")
    
    if uploaded_file is not None:
        # 이미지 로드
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # 이미지 표시
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="업로드된 이미지", use_column_width=True)
        
        # OCR 테스트 버튼
        if st.button("OCR 테스트 실행", key="ocr_button"):
            with st.spinner("OCR 처리 중..."):
                # 모델 초기화
                image_processor = ImageProcessor()
                ocr_engine = OCREngine(gpu=use_gpu)
                
                # 시작 시간
                start_time = time.time()
                
                # 이미지 전처리
                processed_image = image_processor.apply_individual(
                    image,
                    blur=apply_blur,
                    perspective=apply_perspective,
                    normalize=apply_normalize
                )
                
                # 전처리 결과 표시
                st.subheader("전처리 결과")
                
                # 그레이스케일 이미지인 경우 처리
                if len(processed_image.shape) == 2:
                    # 표시만을 위한 3채널 변환
                    display_img = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
                    st.image(display_img, caption="전처리된 이미지", use_column_width=True)
                else:
                    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="전처리된 이미지", use_column_width=True)
                
                # OCR 테스트: 다양한 전처리 방법 시도
                st.subheader("OCR 결과")
                
                # 다양한 전처리 방법을 시도하여 최적의 결과 찾기
                test_results = ocr_engine.test_preprocess_variations(image)
                
                # 결과 정리
                result_df = pd.DataFrame(
                    [(k, v[0], v[1]) for k, v in test_results.items() if k != "best" and k != "best_method"],
                    columns=["전처리 방법", "인식 결과", "신뢰도"]
                ).sort_values(by="신뢰도", ascending=False)
                
                # 테이블로 결과 표시
                st.dataframe(result_df)
                
                # 최적의 결과 표시
                best_text, best_conf = test_results["best"]
                best_method = test_results["best_method"]
                
                st.success(f"최적 결과: '{best_text}' (신뢰도: {best_conf:.2f}, 방법: {best_method})")
                
                # 종료 시간
                end_time = time.time()
                processing_time = end_time - start_time
                
                st.info(f"처리 시간: {processing_time:.2f}초")
                
                # 정답과 비교 (정답이 입력된 경우)
                if ground_truth:
                    similarity = calculate_text_similarity(best_text, ground_truth)
                    st.subheader("정답 비교")
                    st.write(f"실제 번호판: {ground_truth}")
                    st.write(f"인식 결과: {best_text}")
                    st.write(f"텍스트 유사도: {similarity:.2f} (0~1)")
                    
                    # 유사도에 따른 색상 결정
                    if similarity >= 0.8:
                        st.success(f"인식 성공! 유사도: {similarity:.2f}")
                    elif similarity >= 0.5:
                        st.warning(f"부분 인식. 유사도: {similarity:.2f}")
                    else:
                        st.error(f"인식 실패. 유사도: {similarity:.2f}")

def render_batch_processing_tab(settings):
    """
    배치 처리 탭 렌더링
    
    Args:
        settings (dict): 설정 값
    """
    st.header("배치 처리")
    
    # 디렉토리 입력
    input_dir = st.text_input("이미지 디렉토리 경로:", key="batch_input_dir")
    
    # 파일 확장자 선택
    file_extensions = st.multiselect(
        "파일 확장자:",
        ["jpg", "jpeg", "png", "bmp"],
        default=["jpg", "jpeg", "png"],
        key="batch_extensions"
    )
    
    # 출력 디렉토리 입력
    output_dir = st.text_input("결과 저장 디렉토리:", key="batch_output_dir")
    
    # 처리 옵션
    st.subheader("처리 옵션")
    
    # 전처리 옵션
    col1, col2, col3 = st.columns(3)
    
    with col1:
        apply_blur = st.checkbox("흐림 보정", value=True, key="batch_blur")
    
    with col2:
        apply_perspective = st.checkbox("기울기 보정", value=True, key="batch_perspective")
    
    with col3:
        apply_normalize = st.checkbox("비율 정규화", value=True, key="batch_normalize")
    
    # 처리할 최대 이미지 수
    max_images = st.slider(
        "처리할 최대 이미지 수",
        min_value=1,
        max_value=100,
        value=10,
        key="batch_max_images"
    )
    
    # 결과 저장 옵션
    save_images = st.checkbox("처리된 이미지 저장", value=True, key="batch_save_images")
    save_results = st.checkbox("인식 결과 CSV 저장", value=True, key="batch_save_results")
    
    # 실행 버튼
    if st.button("배치 처리 실행", key="batch_button"):
        # 입력 디렉토리 확인
        if not input_dir or not os.path.isdir(input_dir):
            st.error("유효한 입력 디렉토리를 입력하세요.")
            return
        
        # 출력 디렉토리 확인
        if save_images or save_results:
            if not output_dir:
                st.error("출력 디렉토리를 입력하세요.")
                return
            
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
        
        # 이미지 파일 목록 가져오기
        image_files = []
        for ext in file_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
        
        # 최대 이미지 수 제한
        image_files = image_files[:max_images]
        
        if not image_files:
            st.error(f"지정된 디렉토리({input_dir})에서 이미지 파일을 찾을 수 없습니다.")
            return
        
        # 프로그레스 바 초기화
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 결과 저장 변수
        results = []
        
        # 모델 초기화
        vehicle_detector = VehicleDetector()
        plate_detector = PlateDetector()
        image_processor = ImageProcessor()
        ocr_engine = OCREngine(gpu=settings["use_gpu"])
        
        # 각 이미지 처리
        for i, image_file in enumerate(image_files):
            # 진행 상황 업데이트
            progress = (i + 1) / len(image_files)
            progress_bar.progress(progress)
            status_text.text(f"처리 중: {i+1}/{len(image_files)} - {os.path.basename(image_file)}")
            
            try:
                # 이미지 로드
                image = cv2.imread(image_file)
                
                if image is None:
                    st.warning(f"이미지 로드 실패: {image_file}")
                    continue
                
                # 차량 감지
                vehicle_boxes = vehicle_detector.detect(image, conf_threshold=settings["vehicle_detection_conf"])
                
                # 결과 저장 변수
                file_results = []
                
                # 각 차량에 대해 처리
                for vehicle_box in vehicle_boxes:
                    # 차량 영역 추출
                    x1, y1, x2, y2 = vehicle_box
                    vehicle_image = image[y1:y2, x1:x2].copy()
                    
                    # 번호판 감지
                    plate_boxes = plate_detector.detect(vehicle_image, conf_threshold=settings["plate_detection_conf"])
                    
                    # 각 번호판에 대해 처리
                    for plate_box in plate_boxes:
                        # 번호판 영역 추출
                        px1, py1, px2, py2 = plate_box
                        plate_image = vehicle_image[py1:py2, px1:px2].copy()
                        
                        # 번호판 이미지 전처리
                        processed_plate = image_processor.apply_individual(
                            plate_image,
                            blur=apply_blur,
                            perspective=apply_perspective,
                            normalize=apply_normalize
                        )
                        
                        # OCR 수행
                        plate_text, confidence = ocr_engine.recognize_with_confidence(processed_plate)
                        
                        # 한국 번호판 형식에 맞게 후처리
                        formatted_text = ocr_engine.post_processor.format_korean_license_plate(plate_text)
                        
                        # 결과 저장
                        file_results.append({
                            "vehicle_box": vehicle_box,
                            "plate_box": plate_box,
                            "plate_text": formatted_text,
                            "confidence": confidence
                        })
                        
                        # 처리된 이미지 저장
                        if save_images:
                            # 파일명 생성
                            base_name = os.path.splitext(os.path.basename(image_file))[0]
                            plate_idx = len(file_results)
                            
                            # 원본 번호판 이미지 저장
                            plate_file = os.path.join(output_dir, f"{base_name}_plate_{plate_idx}_original.png")
                            cv2.imwrite(plate_file, plate_image)
                            
                            # 처리된 번호판 이미지 저장
                            processed_file = os.path.join(output_dir, f"{base_name}_plate_{plate_idx}_processed.png")
                            cv2.imwrite(processed_file, processed_plate)
                
                # 파일 결과 저장
                for result in file_results:
                    results.append({
                        "file": os.path.basename(image_file),
                        "plate_text": result["plate_text"],
                        "confidence": result["confidence"]
                    })
                
                # 결과 이미지 저장
                if save_images and file_results:
                    # 결과 시각화
                    result_image = vehicle_detector.visualize(image.copy(), [r["vehicle_box"] for r in file_results])
                    
                    # 파일명 생성
                    base_name = os.path.splitext(os.path.basename(image_file))[0]
                    result_file = os.path.join(output_dir, f"{base_name}_result.png")
                    
                    # 결과 이미지 저장
                    cv2.imwrite(result_file, result_image)
            
            except Exception as e:
                st.error(f"이미지 처리 중 오류 발생: {image_file} - {str(e)}")
        
        # 진행 완료
        progress_bar.progress(1.0)
        status_text.text("처리 완료!")
        
        # 결과 저장
        if save_results and results:
            # 결과 CSV 파일 생성
            result_file = os.path.join(output_dir, "batch_results.csv")
            
            # 데이터프레임 생성 및 저장
            result_df = pd.DataFrame(results)
            result_df.to_csv(result_file, index=False)
            
            # 다운로드 링크 제공
            st.download_button(
                label="결과 CSV 다운로드",
                data=result_df.to_csv(index=False),
                file_name="batch_results.csv",
                mime="text/csv"
            )
        
        # 결과 요약
        st.subheader("처리 결과 요약")
        st.write(f"처리된 이미지 수: {len(image_files)}")
        st.write(f"인식된 번호판 수: {len(results)}")
        
        if results:
            # 결과 테이블 표시
            st.dataframe(pd.DataFrame(results))

def render_statistics_tab():
    """
    통계 탭 렌더링
    """
    st.header("통계 및 성능 분석")
    
    # 통계 설명
    st.write("""
    이 탭에서는 프로그램의 성능 통계와 분석 결과를 확인할 수 있습니다.
    데이터가 수집되면 성능 지표, 처리 시간, 인식 정확도 등이 표시됩니다.
    """)
    
    # 예시 통계 데이터 (실제로는 데이터 수집 및 분석 후 표시)
    st.subheader("성능 지표 예시")
    
    # 예시 데이터
    example_data = {
        "정확도": 0.85,
        "평균 처리 시간": 0.42,
        "번호판 감지율": 0.92,
        "OCR 성공률": 0.88
    }
    
    # 게이지 차트로 표시
    cols = st.columns(len(example_data))
    for i, (metric, value) in enumerate(example_data.items()):
        with cols[i]:
            st.metric(label=metric, value=f"{value:.2f}")
    
    # 시각화 예시
    st.subheader("시각화 예시")
    
    # 예시 차트 데이터
    chart_data = pd.DataFrame({
        "전처리 방법": ["원본", "흐림 보정", "기울기 보정", "비율 정규화", "모든 보정"],
        "정확도": [0.65, 0.72, 0.78, 0.75, 0.85]
    })
    
    # 차트 생성
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(chart_data["전처리 방법"], chart_data["정확도"], color="skyblue")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("정확도")
    ax.set_title("전처리 방법별 정확도")
    
    # 차트 표시
    st.pyplot(fig)
    
    # 추가 통계
    st.subheader("추가 통계")
    
    # 예시 데이터
    additional_stats = pd.DataFrame({
        "항목": ["총 처리 이미지", "인식된 차량", "인식된 번호판", "평균 신뢰도"],
        "값": ["127", "183", "142", "0.76"]
    })
    
    # 테이블 표시
    st.table(additional_stats)