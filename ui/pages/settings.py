import streamlit as st
import os
import json
import yaml
from pathlib import Path
import torch
import config

"""
설정 페이지 모듈

이 모듈은 Streamlit 애플리케이션의 설정 페이지를 렌더링합니다.
"""
def render_settings_page():
    """
    설정 페이지 렌더링
    """
    st.title("설정")
    
    # 탭 생성
    tabs = st.tabs(["일반 설정", "모델 설정", "고급 설정", "정보"])
    
    # 일반 설정 탭
    with tabs[0]:
        render_general_settings()
    
    # 모델 설정 탭
    with tabs[1]:
        render_model_settings()
    
    # 고급 설정 탭
    with tabs[2]:
        render_advanced_settings()
    
    # 정보 탭
    with tabs[3]:
        render_info_tab()

def render_general_settings():
    """
    일반 설정 탭 렌더링
    """
    st.header("일반 설정")
    
    # 설정 폼 생성
    with st.form("general_settings_form"):
        # 기본 입력 형식 설정
        st.subheader("기본 입력 형식")
        default_input = st.radio(
            "기본 입력 형식:",
            ["이미지 업로드", "카메라 촬영", "비디오 업로드"],
            index=0
        )
        
        # 출력 디렉토리 설정
        st.subheader("출력 설정")
        output_dir = st.text_input(
            "결과 저장 디렉토리:",
            value=str(Path.home() / "LicensePlateOCR_Results")
        )
        
        # 자동 저장 설정
        auto_save = st.checkbox("결과 자동 저장", value=False)
        
        # 로그 레벨 설정
        log_level = st.selectbox(
            "로그 레벨:",
            ["INFO", "DEBUG", "WARNING", "ERROR"],
            index=0
        )
        
        # 언어 설정
        language = st.selectbox(
            "언어:",
            ["한국어", "English"],
            index=0
        )
        
        # 저장 버튼
        submit = st.form_submit_button("설정 저장")
        
        if submit:
            # 설정 값 저장
            settings = {
                "default_input": default_input,
                "output_dir": output_dir,
                "auto_save": auto_save,
                "log_level": log_level,
                "language": language
            }
            
            # 설정 파일 저장
            save_settings(settings, "general_settings.json")
            
            st.success("일반 설정이 저장되었습니다.")

def render_model_settings():
    """
    모델 설정 탭 렌더링
    """
    st.header("모델 설정")
    
    # GPU 정보 표시
    st.subheader("하드웨어 정보")
    
    # GPU 사용 가능 여부 확인
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB 단위로 변환
        st.success(f"GPU 사용 가능: {gpu_name} ({gpu_memory:.2f} GB)")
    else:
        st.warning("GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
    
    # 설정 폼 생성
    with st.form("model_settings_form"):
        # GPU 사용 설정
        use_gpu = st.checkbox("GPU 사용 (사용 가능한 경우)", value=gpu_available)
        
        # 차량 감지 모델 설정
        st.subheader("차량 감지 모델")
        
        vehicle_model = st.selectbox(
            "차량 감지 모델:",
            ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"],
            index=1  # YOLOv8s
        )
        
        vehicle_conf = st.slider(
            "차량 감지 신뢰도 임계값:",
            min_value=0.1,
            max_value=0.9,
            value=config.VEHICLE_DETECTION_CONF,
            step=0.05
        )
        
        # 번호판 감지 모델 설정
        st.subheader("번호판 감지 모델")
        
        plate_model = st.selectbox(
            "번호판 감지 모델:",
            ["YOLOv8n", "YOLOv8s", "YOLOv8m"],
            index=1  # YOLOv8s
        )
        
        plate_conf = st.slider(
            "번호판 감지 신뢰도 임계값:",
            min_value=0.1,
            max_value=0.9,
            value=config.PLATE_DETECTION_CONF,
            step=0.05
        )
        
        custom_model_path = st.text_input(
            "사용자 정의 모델 경로 (선택 사항):",
            value=""
        )
        
        # OCR 모델 설정
        st.subheader("OCR 설정")
        
        ocr_engine = st.selectbox(
            "OCR 엔진:",
            ["EasyOCR", "Tesseract"],
            index=0  # EasyOCR
        )
        
        # 저장 버튼
        submit = st.form_submit_button("설정 저장")
        
        if submit:
            # 설정 값 저장
            settings = {
                "use_gpu": use_gpu,
                "vehicle_model": vehicle_model,
                "vehicle_conf": vehicle_conf,
                "plate_model": plate_model,
                "plate_conf": plate_conf,
                "custom_model_path": custom_model_path,
                "ocr_engine": ocr_engine
            }
            
            # 설정 파일 저장
            save_settings(settings, "model_settings.json")
            
            st.success("모델 설정이 저장되었습니다.")
            st.info("변경된 설정은 애플리케이션 재시작 시 적용됩니다.")

def render_advanced_settings():
    """
    고급 설정 탭 렌더링
    """
    st.header("고급 설정")
    
    # 설정 폼 생성
    with st.form("advanced_settings_form"):
        # 이미지 처리 설정
        st.subheader("이미지 처리 설정")
        
        # 이미지 크기 설정
        image_size = st.number_input(
            "이미지 크기 (픽셀):",
            min_value=320,
            max_value=1280,
            value=640,
            step=32
        )
        
        # 번호판 정규화 크기 설정
        plate_width = st.number_input(
            "번호판 정규화 너비 (픽셀):",
            min_value=80,
            max_value=400,
            value=240,
            step=10
        )
        
        plate_height = st.number_input(
            "번호판 정규화 높이 (픽셀):",
            min_value=20,
            max_value=200,
            value=80,
            step=10
        )
        
        # 전처리 설정
        st.subheader("전처리 설정")
        
        blur_kernel_size = st.slider(
            "블러 커널 크기:",
            min_value=1,
            max_value=11,
            value=5,
            step=2
        )
        
        blur_sigma = st.slider(
            "가우시안 블러 시그마:",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1
        )
        
        # OCR 고급 설정
        st.subheader("OCR 고급 설정")
        
        allowed_chars = st.text_area(
            "허용된 문자:",
            value=config.OCR_ALLOWED_CHARS,
            height=150
        )
        
        # 저장 버튼
        submit = st.form_submit_button("설정 저장")
        
        if submit:
            # 설정 값 저장
            settings = {
                "image_size": image_size,
                "plate_width": plate_width,
                "plate_height": plate_height,
                "blur_kernel_size": blur_kernel_size,
                "blur_sigma": blur_sigma,
                "allowed_chars": allowed_chars
            }
            
            # 설정 파일 저장
            save_settings(settings, "advanced_settings.json")
            
            st.success("고급 설정이 저장되었습니다.")
            st.info("변경된 설정은 애플리케이션 재시작 시 적용됩니다.")

def render_info_tab():
    """
    정보 탭 렌더링
    """
    st.header("프로그램 정보")
    
    # 프로그램 설명
    st.markdown("""
    # 차량번호 OCR 프로그램
    
    이 프로그램은 이미지나 비디오에서 차량 번호판을 자동으로 인식합니다.
    
    ## 개발팀 정보
    - 팀명: 번호뭔지알려조
    - 팀원: 최윤정(팀장), 유우림, 김기윤, 이주환
    - 개발기간: 2025.03 ~ 2025.05
    
    ## 기술 스택
    - 차량 탐지: YOLOv8s
    - 번호판 탐지: YOLOv8s (custom trained)
    - 번호판 보정: OpenCV
    - 번호판 숫자 인식: EasyOCR
    - UI: Streamlit
    
    ## 저장소
    - GitHub: [https://github.com/yunchocopie/license-plate-ocr-project](https://github.com/yunchocopie/license-plate-ocr-project)
    
    ## 라이센스
    - MIT License
    
    ## 감사의 말
    이 프로젝트는 사하구청과 협력하여 개발되었으며, 동아대학교 SW중심대학사업의 실증적 SW/AI 프로젝트로 진행되었습니다.
    """)
    
    # 시스템 정보
    st.subheader("시스템 정보")
    
    # Python 버전
    import sys
    st.write(f"Python 버전: {sys.version}")
    
    # 주요 라이브러리 버전
    import torch
    import cv2
    import numpy as np
    
    st.write(f"PyTorch 버전: {torch.__version__}")
    st.write(f"OpenCV 버전: {cv2.__version__}")
    st.write(f"NumPy 버전: {np.__version__}")
    
    # GPU 정보
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.write(f"GPU: {gpu_name}")
    else:
        st.write("GPU: 사용 불가")
    
    # 경로 정보
    st.subheader("경로 정보")
    st.write(f"실행 경로: {os.getcwd()}")
    st.write(f"설정 파일 경로: {os.path.abspath(config.__file__)}")

def save_settings(settings, filename):
    """
    설정 저장
    
    Args:
        settings (dict): 저장할 설정
        filename (str): 설정 파일명
    """
    # 설정 디렉토리 생성
    settings_dir = Path.home() / ".licenseplateocr"
    settings_dir.mkdir(exist_ok=True)
    
    # 설정 파일 경로
    settings_path = settings_dir / filename
    
    # 설정 저장
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)

def load_settings(filename):
    """
    설정 로드
    
    Args:
        filename (str): 설정 파일명
        
    Returns:
        dict: 로드된 설정 또는 빈 딕셔너리
    """
    # 설정 파일 경로
    settings_path = Path.home() / ".licenseplateocr" / filename
    
    # 설정 파일이 없으면 빈 딕셔너리 반환
    if not settings_path.exists():
        return {}
    
    # 설정 로드
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"설정 로드 중 오류 발생: {str(e)}")
        return {}