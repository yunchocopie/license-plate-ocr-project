import streamlit as st
from PIL import Image
import numpy as np
import time

# 내부 모듈 임포트
from src.detection.vehicle_detector import VehicleDetector
from src.detection.plate_detector import PlateDetector
from src.preprocessing.image_processor import ImageProcessor
from src.ocr.ocr_engine import OCREngine
from src.utils.visualization import visualize_vehicle_detection, visualize_plate_detection, visualize_results

st.set_page_config(
    page_title="차량번호 OCR 프로그램 (디버그 모드)",
    page_icon="🚗",
    layout="wide"
)
st.title("차량번호 OCR 프로그램 (디버그 모드)")
st.markdown("자동 파이프라인: 차량 감지 → 실패 시 직접 번호판 감지")

# 모델 초기화
try:
    vehicle_detector = VehicleDetector()
    plate_detector = PlateDetector()
    image_processor = ImageProcessor()
    ocr_engine = OCREngine()
except Exception as e:
    st.error(f"모델 초기화 중 오류가 발생했습니다: {str(e)}")
    st.stop()

# 이미지 업로드 UI
uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # 1. 이미지 로드
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.header("원본 이미지")
        st.image(image, use_column_width=True)
        total_start = time.time()
        
        # 파이프라인 1단계: 차량 감지
        vehicle_start = time.time()
        vehicle_boxes = vehicle_detector.detect(image_np)
        vehicle_time = time.time() - vehicle_start
        
        results = []
        plate_images = []
        processed_plates = []
        plate_texts = []
        plate_time = 0
        preprocess_time = 0
        ocr_time = 0
        direct_plate_time = 0
        
        # 차량이 감지되었을 경우
        if len(vehicle_boxes) > 0:
            st.header("1단계: 차량 감지 결과")
            st.write(f"감지된 차량: {len(vehicle_boxes)}개 (처리 시간: {vehicle_time:.3f}초)")
            st.image(visualize_vehicle_detection(image_np.copy(), vehicle_boxes), use_column_width=True)
            
            # 파이프라인 2단계: 차량별 번호판 감지 및 OCR
            for idx, vehicle_box in enumerate(vehicle_boxes):
                x1, y1, x2, y2 = vehicle_box
                vehicle_img = image_np[y1:y2, x1:x2]
                # 번호판 감지
                plate_start = time.time()
                plate_boxes = plate_detector.detect(vehicle_img)
                plate_time += time.time() - plate_start
                
                if len(plate_boxes) > 0:
                    st.subheader(f"차량 {idx+1}의 번호판 감지 결과")
                    st.write(f"감지된 번호판: {len(plate_boxes)}개")
                    st.image(visualize_plate_detection(vehicle_img.copy(), plate_boxes), use_column_width=True)
                    
                    for pidx, plate_box in enumerate(plate_boxes):
                        px1, py1, px2, py2 = plate_box
                        plate_img = vehicle_img[py1:py2, px1:px2]
                        plate_images.append(plate_img)
                        # 전처리
                        pre_start = time.time()
                        processed = image_processor.process(plate_img)
                        preprocess_time += time.time() - pre_start
                        processed_plates.append(processed)
                        # OCR
                        ocr_start = time.time()
                        text, conf = ocr_engine.recognize_with_confidence(processed)
                        ocr_time += time.time() - ocr_start
                        plate_texts.append(text)
                        # 단계별 시각화
                        st.markdown(f"**차량 {idx+1}, 번호판 {pidx+1}**")
                        cols = st.columns(2)
                        cols[0].image(plate_img, caption="원본 번호판", use_column_width=True)
                        cols[1].image(processed, caption="처리된 번호판", use_column_width=True)
                        # 단계별 전처리 시각화
                        with st.expander("전처리 단계별 이미지"):
                            steps = image_processor.visualize_steps(plate_img)
                            for step_name, step_img in steps.items():
                                st.image(step_img, caption=step_name, use_column_width=True)
                        st.write(f"OCR 결과: {text} (신뢰도: {conf:.2f})")
                        # 결과 저장
                        results.append({
                            "vehicle_box": vehicle_box,
                            "plate_box": [vehicle_box[0]+px1, vehicle_box[1]+py1, vehicle_box[0]+px2, vehicle_box[1]+py2],
                            "plate_text": text
                        })
        
        # 차량이 감지되지 않았거나, 차량에서 번호판을 찾지 못한 경우
        if len(results) == 0:
            st.header("차량 감지 실패 또는 번호판 미발견")
            st.warning("차량 감지에 실패했거나 차량에서 번호판을 찾지 못했습니다. 직접 번호판 감지를 시도합니다.")
            
            # 파이프라인 대체 단계: 직접 번호판 감지
            direct_plate_start = time.time()
            plate_boxes = plate_detector.detect(image_np)
            direct_plate_time = time.time() - direct_plate_start
            
            st.subheader("직접 번호판 감지 결과")
            st.write(f"감지된 번호판: {len(plate_boxes)}개 (처리 시간: {direct_plate_time:.3f}초)")
            
            if len(plate_boxes) > 0:
                st.image(visualize_plate_detection(image_np.copy(), plate_boxes), use_column_width=True)
                
                for pidx, plate_box in enumerate(plate_boxes):
                    px1, py1, px2, py2 = plate_box
                    plate_img = image_np[py1:py2, px1:px2]
                    plate_images.append(plate_img)
                    # 전처리
                    pre_start = time.time()
                    processed = image_processor.process(plate_img)
                    preprocess_time += time.time() - pre_start
                    processed_plates.append(processed)
                    # OCR
                    ocr_start = time.time()
                    text, conf = ocr_engine.recognize_with_confidence(processed)
                    ocr_time += time.time() - ocr_start
                    plate_texts.append(text)
                    # 단계별 시각화
                    st.markdown(f"**직접 감지 번호판 {pidx+1}**")
                    cols = st.columns(2)
                    cols[0].image(plate_img, caption="원본 번호판", use_column_width=True)
                    cols[1].image(processed, caption="처리된 번호판", use_column_width=True)
                    # 단계별 전처리 시각화
                    with st.expander("전처리 단계별 이미지"):
                        steps = image_processor.visualize_steps(plate_img)
                        for step_name, step_img in steps.items():
                            st.image(step_img, caption=step_name, use_column_width=True)
                    st.write(f"OCR 결과: {text} (신뢰도: {conf:.2f})")
                    # 결과 저장 (차량 박스 없음)
                    results.append({
                        "vehicle_box": None,
                        "plate_box": plate_box,
                        "plate_text": text
                    })
        
        # 최종 결과 요약
        total_time = time.time() - total_start
        st.header("최종 인식 결과")
        
        if len(results) > 0:
            # 최종 결과 시각화
            final_image = visualize_results(image_np.copy(), results)
            st.image(final_image, use_column_width=True)
            
            # 인식된 번호판 텍스트 표시
            st.subheader("인식된 번호판")
            for idx, result in enumerate(results):
                st.write(f"번호판 {idx+1}: {result['plate_text']}")
                
            st.success(f"{len(results)}개의 번호판이 감지되었습니다.")
        else:
            st.error("번호판을 찾지 못했습니다.")
            
        # 시간 비중 그래프
        st.subheader("단계별 처리 시간")
        time_data = {
            "단계": ["차량 감지", "번호판 감지", "직접 번호판 감지", "이미지 전처리", "OCR 처리", "기타"],
            "시간(초)": [
                vehicle_time, 
                plate_time, 
                direct_plate_time,
                preprocess_time, 
                ocr_time, 
                total_time - (vehicle_time + plate_time + direct_plate_time + preprocess_time + ocr_time)
            ]
        }
        st.bar_chart(time_data, x="단계", y="시간(초)")
        
        # 디버그 정보
        with st.expander("원본 디버그 데이터"):
            st.json({
                "총 처리 시간": total_time,
                "차량 감지 시간": vehicle_time,
                "번호판 감지 시간": plate_time,
                "직접 번호판 감지 시간": direct_plate_time,
                "이미지 전처리 시간": preprocess_time,
                "OCR 처리 시간": ocr_time,
                "감지된 번호판 수": len(plate_images),
                "인식된 번호판 텍스트": plate_texts,
                "결과": results
            })
    except Exception as e:
        st.error(f"이미지 처리 중 오류가 발생했습니다: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")