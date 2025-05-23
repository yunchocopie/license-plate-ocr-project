import streamlit as st
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

# 내부 모듈 임포트
from src.detection.vehicle_detector import VehicleDetector
from src.detection.plate_detector import PlateDetector
from src.preprocessing.image_processor import ImageProcessor
from src.ocr.ocr_engine import OCREngine
from src.utils.visualization import visualize_vehicle_detection, visualize_plate_detection, visualize_results

st.set_page_config(page_title="OCR Debug", page_icon="🚗", layout="wide")
st.title("차량번호 OCR - 개발 디버그")

# 모델 초기화
@st.cache_resource
def load_models():
    return VehicleDetector(), PlateDetector(), ImageProcessor(), OCREngine()

vehicle_detector, plate_detector, image_processor, ocr_engine = load_models()

# 이미지 업로드
uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    st.header("원본 이미지")
    st.image(image, width=600)
    
    total_start = time.time()
    results = []
    
    # 1단계: 차량 감지
    st.header("1단계: 차량 감지")
    vehicle_start = time.time()
    vehicle_boxes = vehicle_detector.detect(image_np)
    vehicle_time = time.time() - vehicle_start
    
    st.write(f"감지된 차량: {len(vehicle_boxes)}개")
    st.write(f"처리 시간: {vehicle_time:.3f}초")
    
    if vehicle_boxes:
        st.image(visualize_vehicle_detection(image_np.copy(), vehicle_boxes), width=600)
        
        # 차량별 번호판 감지
        for idx, vehicle_box in enumerate(vehicle_boxes):
            st.subheader(f"차량 {idx+1}")
            x1, y1, x2, y2 = vehicle_box
            vehicle_img = image_np[y1:y2, x1:x2]
            
            # 번호판 감지
            plate_start = time.time()
            plate_boxes = plate_detector.detect(vehicle_img)
            plate_time = time.time() - plate_start
            
            st.write(f"번호판 감지: {len(plate_boxes)}개, 시간: {plate_time:.3f}초")
            
            if plate_boxes:
                st.image(visualize_plate_detection(vehicle_img.copy(), plate_boxes), width=400)
                
                # 각 번호판 처리
                for pidx, plate_box in enumerate(plate_boxes):
                    px1, py1, px2, py2 = plate_box
                    plate_img = vehicle_img[py1:py2, px1:px2]
                    
                    st.write(f"번호판 {pidx+1}")
                    
                    # 품질 평가 (1단계 기능)
                    quality_start = time.time()
                    try:
                        if hasattr(image_processor, 'assess_image_quality'):
                            quality_metrics, processing_level = image_processor.assess_image_quality(plate_img)
                            st.write(f"품질 점수: {quality_metrics['overall_score']:.1f}/100")
                            st.write(f"처리 수준: {processing_level.upper()}")
                        else:
                            processing_level = 'full'
                            quality_metrics = {'overall_score': 0}
                    except:
                        processing_level = 'full'
                        quality_metrics = {'overall_score': 0}
                    quality_time = time.time() - quality_start
                    
                    # 전처리
                    preprocess_start = time.time()
                    processed = image_processor.process(plate_img)
                    preprocess_time = time.time() - preprocess_start
                    
                    # OCR
                    ocr_start = time.time()
                    text, conf = ocr_engine.recognize_with_confidence(processed)
                    ocr_time = time.time() - ocr_start
                    
                    # 단계별 전처리 시각화 (Matplotlib + Streamlit)
                    if st.checkbox(f"전처리 단계별 시각화 보기 (번호판 {pidx+1})", key=f"viz_{idx}_{pidx}"):
                        steps = image_processor.visualize_steps(plate_img)
                        fig, axes = plt.subplots(1, len(steps), figsize=(3*len(steps), 3))
                        if len(steps) == 1:
                            axes = [axes]
                        for ax, (step_name, step_img) in zip(axes, steps.items()):
                            if len(step_img.shape) == 2:
                                ax.imshow(step_img, cmap='gray')
                            else:
                                ax.imshow(step_img)
                            ax.set_title(step_name)
                            ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # 결과 표시
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(plate_img, caption="원본", width=150)
                    with col2:
                        st.image(processed, caption=f"처리됨({processing_level})", width=150)
                    with col3:
                        st.write(f"텍스트: **{text}**")
                        st.write(f"신뢰도: {conf:.2f}")
                        st.write(f"품질평가: {quality_time*1000:.1f}ms")
                        st.write(f"전처리: {preprocess_time*1000:.1f}ms")
                        st.write(f"OCR: {ocr_time*1000:.1f}ms")
                    
                    # 결과 저장
                    results.append({
                        "vehicle_box": vehicle_box,
                        "plate_box": [vehicle_box[0]+px1, vehicle_box[1]+py1, vehicle_box[0]+px2, vehicle_box[1]+py2],
                        "plate_text": text,
                        "confidence": conf,
                        "quality_score": quality_metrics['overall_score'],
                        "processing_level": processing_level
                    })
    
    # 차량 감지 실패 시 직접 번호판 감지
    if len(results) == 0:
        st.header("대체: 직접 번호판 감지")
        direct_start = time.time()
        plate_boxes = plate_detector.detect(image_np)
        direct_time = time.time() - direct_start
        
        st.write(f"직접 감지: {len(plate_boxes)}개, 시간: {direct_time:.3f}초")
        
        if plate_boxes:
            st.image(visualize_plate_detection(image_np.copy(), plate_boxes), width=600)
            
            for pidx, plate_box in enumerate(plate_boxes):
                px1, py1, px2, py2 = plate_box
                plate_img = image_np[py1:py2, px1:px2]
                
                st.write(f"번호판 {pidx+1}")
                
                # 품질 평가
                quality_start = time.time()
                try:
                    if hasattr(image_processor, 'assess_image_quality'):
                        quality_metrics, processing_level = image_processor.assess_image_quality(plate_img)
                    else:
                        processing_level = 'full'
                        quality_metrics = {'overall_score': 0}
                except:
                    processing_level = 'full'
                    quality_metrics = {'overall_score': 0}
                quality_time = time.time() - quality_start
                
                # 전처리
                preprocess_start = time.time()
                processed = image_processor.process(plate_img)
                preprocess_time = time.time() - preprocess_start
                
                # OCR
                ocr_start = time.time()
                text, conf = ocr_engine.recognize_with_confidence(processed)
                ocr_time = time.time() - ocr_start
                
                # 단계별 전처리 시각화 (Matplotlib + Streamlit)
                if st.checkbox(f"전처리 단계별 시각화 보기 (번호판 {pidx+1})", key=f"viz_direct_{pidx}"):
                    steps = image_processor.visualize_steps(plate_img)
                    fig, axes = plt.subplots(1, len(steps), figsize=(3*len(steps), 3))
                    if len(steps) == 1:
                        axes = [axes]
                    for ax, (step_name, step_img) in zip(axes, steps.items()):
                        if len(step_img.shape) == 2:
                            ax.imshow(step_img, cmap='gray')
                        else:
                            ax.imshow(step_img)
                        ax.set_title(step_name)
                        ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
                
                # 결과 표시
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(plate_img, caption="원본", width=150)
                with col2:
                    st.image(processed, caption=f"처리됨({processing_level})", width=150)
                with col3:
                    st.write(f"텍스트: **{text}**")
                    st.write(f"신뢰도: {conf:.2f}")
                    st.write(f"품질평가: {quality_time*1000:.1f}ms")
                    st.write(f"전처리: {preprocess_time*1000:.1f}ms")
                    st.write(f"OCR: {ocr_time*1000:.1f}ms")
                
                # 결과 저장
                results.append({
                    "vehicle_box": None,
                    "plate_box": plate_box,
                    "plate_text": text,
                    "confidence": conf,
                    "quality_score": quality_metrics['overall_score'],
                    "processing_level": processing_level
                })
    
    # 최종 결과
    total_time = time.time() - total_start
    
    st.header("최종 결과")
    st.write(f"총 처리 시간: {total_time:.3f}초")
    st.write(f"인식된 번호판: {len(results)}개")
    
    if results:
        # 최종 시각화
        final_image = visualize_results(image_np.copy(), results)
        st.image(final_image, width=600)
        
        # 결과 테이블
        st.subheader("상세 결과")
        for idx, result in enumerate(results):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"번호판 {idx+1}")
            with col2:
                st.write(f"**{result['plate_text']}**")
            with col3:
                st.write(f"신뢰도: {result['confidence']:.2f}")
            with col4:
                st.write(f"품질: {result['quality_score']:.1f}")
        
        # 처리 수준 통계 (1단계 기능)
        if any(r.get('processing_level') for r in results):
            processing_levels = [r['processing_level'] for r in results if r.get('processing_level')]
            level_counts = {}
            for level in processing_levels:
                level_counts[level] = level_counts.get(level, 0) + 1
            
            st.subheader("처리 수준 분포")
            for level, count in level_counts.items():
                st.write(f"{level.upper()}: {count}개")
    else:
        st.error("번호판을 찾지 못했습니다.")
    
    # 디버그 정보
    with st.expander("디버그 데이터"):
        debug_info = {
            "총_처리_시간": total_time,
            "감지된_번호판_수": len(results),
            "결과": results
        }
        st.json(debug_info)
