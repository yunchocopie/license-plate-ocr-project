import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import config

"""
UI 컴포넌트 모듈

이 모듈은 Streamlit UI의 재사용 가능한 컴포넌트들을 제공합니다.
"""
def create_sidebar():
    """
    사이드바 UI 생성
    
    Returns:
        dict: 사이드바 설정 값
    """
    st.sidebar.title("설정")
    
    # 입력 유형 선택
    input_type = st.sidebar.radio("입력 유형", ["이미지 업로드", "카메라 촬영", "비디오 업로드"])
    
    # 고급 설정 섹션
    with st.sidebar.expander("고급 설정", expanded=False):
        # 차량 감지 설정
        st.subheader("차량 감지 설정")
        vehicle_detection_conf = st.slider(
            "차량 감지 신뢰도 임계값",
            min_value=0.1,
            max_value=0.9,
            value=config.VEHICLE_DETECTION_CONF,
            step=0.05
        )
        
        # 번호판 감지 설정
        st.subheader("번호판 감지 설정")
        plate_detection_conf = st.slider(
            "번호판 감지 신뢰도 임계값",
            min_value=0.1,
            max_value=0.9,
            value=config.PLATE_DETECTION_CONF,
            step=0.05
        )
        
        # 이미지 처리 설정
        st.subheader("이미지 처리 설정")
        apply_blur_correction = st.checkbox("흐림 보정 적용", value=True)
        apply_perspective_correction = st.checkbox("기울기 보정 적용", value=True)
        apply_normalize = st.checkbox("비율 정규화 적용", value=True)
        
        # OCR 설정
        st.subheader("OCR 설정")
        min_ocr_confidence = st.slider(
            "OCR 최소 신뢰도 임계값",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05
        )
    
    # GPU 사용 여부
    use_gpu = st.sidebar.checkbox("GPU 사용 (가능한 경우)", value=config.OCR_GPU)
    
    # 설정 저장
    settings = {
        "input_type": input_type,
        "vehicle_detection_conf": vehicle_detection_conf,
        "plate_detection_conf": plate_detection_conf,
        "apply_blur_correction": apply_blur_correction,
        "apply_perspective_correction": apply_perspective_correction,
        "apply_normalize": apply_normalize,
        "min_ocr_confidence": min_ocr_confidence,
        "use_gpu": use_gpu
    }
    
    return settings

def create_header():
    """
    페이지 헤더 UI 생성
    """
    st.title("차량번호 OCR 프로그램")
    st.markdown("""
    이 프로그램은 이미지나 비디오에서 차량 번호판을 자동으로 인식합니다.
    * YOLOv8s를 이용한 차량 및 번호판 탐지
    * OpenCV를 이용한 이미지 보정 (흐림 보정, 기울기 보정, 비율 정규화)
    * EasyOCR을 이용한 번호판 텍스트 인식
    
    왼쪽 사이드바에서 설정을 변경할 수 있습니다.
    """)
    
    # 구분선 추가
    st.markdown("---")

def create_upload_section(input_type):
    """
    파일 업로드 및 입력 섹션 생성
    
    Args:
        input_type (str): 입력 유형
        
    Returns:
        tuple: (입력 이미지/비디오, 파일명)
    """
    if input_type == "이미지 업로드":
        # 이미지 업로드
        uploaded_file = st.file_uploader(
            "이미지 파일을 업로드하세요",
            type=["jpg", "jpeg", "png", "bmp"]
        )
        
        if uploaded_file is not None:
            # 이미지 표시
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", use_column_width=True)
            
            # OpenCV로 이미지 변환
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_cv, uploaded_file.name
        
    elif input_type == "카메라 촬영":
        # 카메라 입력
        img_file_buffer = st.camera_input("카메라로 촬영하세요")
        
        if img_file_buffer is not None:
            # 이미지 표시
            image = Image.open(img_file_buffer)
            
            # OpenCV로 이미지 변환
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_cv, "camera_capture.jpg"
        
    elif input_type == "비디오 업로드":
        # 비디오 업로드
        uploaded_file = st.file_uploader(
            "비디오 파일을 업로드하세요",
            type=["mp4", "avi", "mov", "mkv"]
        )
        
        if uploaded_file is not None:
            # 임시 파일로 저장
            temp_file = io.BytesIO(uploaded_file.read())
            
            # 비디오 재생
            st.video(temp_file)
            
            # TODO: OpenCV로 비디오 처리 추가
            st.warning("비디오 처리 기능은 현재 개발 중입니다")
            
            return None, uploaded_file.name
    
    return None, None

def create_result_section(results, original_image, processed_images=None):
    """
    결과 표시 섹션 생성
    
    Args:
        results (list): 처리 결과 목록
        original_image (numpy.ndarray): 원본 이미지
        processed_images (dict, optional): 처리된 이미지 딕셔너리. 기본값은 None
    """
    if results:
        st.subheader("처리 결과")
        
        # 탭 생성
        tabs = st.tabs(["인식 결과", "전처리 단계", "상세 정보"])
        
        # 인식 결과 탭
        with tabs[0]:
            # 인식된 차량 수
            st.write(f"인식된 차량 수: {len(results)}")
            
            # 각 차량에 대한 결과 표시
            for i, result in enumerate(results):
                with st.expander(f"차량 {i+1}", expanded=i==0):
                    # 차량 결과에서 필요한 정보 추출
                    vehicle_box = result.get('vehicle_box')
                    plate_box = result.get('plate_box')
                    plate_text = result.get('plate_text', '')
                    confidence = result.get('confidence', 0.0)
                    
                    # 결과 표시
                    cols = st.columns(2)
                    with cols[0]:
                        st.write("**인식된 번호판:**")
                        st.markdown(f"<h2 style='text-align: center;'>{plate_text}</h2>", unsafe_allow_html=True)
                        st.write(f"**신뢰도:** {confidence:.2f}")
                    
                    with cols[1]:
                        # 번호판 이미지 표시
                        if 'plate_image' in result:
                            plate_img = result['plate_image']
                            # BGR에서 RGB로 변환
                            plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                            st.image(plate_img_rgb, caption="인식된 번호판", use_column_width=True)
        
        # 전처리 단계 탭
        with tabs[1]:
            if processed_images:
                # 전처리 단계별 이미지 표시
                st.write("전처리 단계:")
                
                # 그리드 레이아웃 생성
                cols = st.columns(len(processed_images))
                
                # 각 단계 이미지 표시
                for i, (step_name, step_img) in enumerate(processed_images.items()):
                    with cols[i % len(cols)]:
                        # 이미지가 그레이스케일인지 확인
                        if len(step_img.shape) == 2 or step_img.shape[2] == 1:
                            # 그레이스케일 이미지 표시
                            st.image(step_img, caption=step_name, use_column_width=True)
                        else:
                            # BGR에서 RGB로 변환
                            step_img_rgb = cv2.cvtColor(step_img, cv2.COLOR_BGR2RGB)
                            st.image(step_img_rgb, caption=step_name, use_column_width=True)
        
        # 상세 정보 탭
        with tabs[2]:
            st.write("처리 시간: [계산된 처리 시간] 초")
            
            # 감지된 물체 좌표 정보
            st.write("감지 정보:")
            for i, result in enumerate(results):
                with st.expander(f"차량 {i+1}"):
                    vehicle_box = result.get('vehicle_box')
                    plate_box = result.get('plate_box')
                    
                    if vehicle_box:
                        st.write(f"차량 좌표: {vehicle_box}")
                    
                    if plate_box:
                        st.write(f"번호판 좌표: {plate_box}")
            
            # 결과 다운로드 옵션
            st.download_button(
                label="결과 JSON 다운로드",
                data=str(results),
                file_name="detection_results.json",
                mime="application/json"
            )
    else:
        st.warning("처리 결과가 없습니다. 이미지를 업로드하고 처리 버튼을 클릭하세요.")

def get_image_download_link(img, filename, text):
    """
    이미지 다운로드 링크 생성
    
    Args:
        img (numpy.ndarray): 다운로드할 이미지
        filename (str): 다운로드 파일명
        text (str): 링크 텍스트
        
    Returns:
        str: HTML 다운로드 링크
    """
    # 이미지가 그레이스케일인지 확인
    if len(img.shape) == 2 or img.shape[2] == 1:
        # 3채널로 변환
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # BGR에서 RGB로 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 이미지를 PIL 이미지로 변환
    pil_img = Image.fromarray(img_rgb)
    
    # 이미지를 바이트로 변환
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    
    # 바이트를 Base64로 인코딩
    img_str = base64.b64encode(buf.getvalue()).decode()
    
    # HTML 다운로드 링크 생성
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    
    return href

def create_processing_options():
    """
    처리 옵션 UI 생성
    
    Returns:
        dict: 처리 옵션 설정 값
    """
    st.subheader("처리 옵션")
    
    # 처리 옵션 선택
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_vehicle_detection = st.checkbox("차량 감지 표시", value=True)
    
    with col2:
        show_plate_detection = st.checkbox("번호판 감지 표시", value=True)
    
    with col3:
        show_text_overlay = st.checkbox("텍스트 오버레이 표시", value=True)
    
    # 추가 옵션
    show_processing_steps = st.checkbox("전처리 단계 표시", value=False)
    
    # 옵션 저장
    options = {
        "show_vehicle_detection": show_vehicle_detection,
        "show_plate_detection": show_plate_detection,
        "show_text_overlay": show_text_overlay,
        "show_processing_steps": show_processing_steps
    }
    
    return options