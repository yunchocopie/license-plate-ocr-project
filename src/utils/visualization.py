import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

"""
결과 시각화 도구 모듈

이 모듈은 차량 감지, 번호판 감지, OCR 결과를 시각화하는 함수들을 제공합니다.
"""
def visualize_results(image, results):
    """
    전체 처리 결과 시각화
    
    Args:
        image (numpy.ndarray): 원본 이미지
        results (list): 처리 결과 목록 [{'vehicle_box': [x1, y1, x2, y2], 'plate_box': [x1, y1, x2, y2], 'plate_text': 'text'}, ...]
        
    Returns:
        numpy.ndarray: 결과가 시각화된 이미지
    """
    # 이미지 복사
    vis_img = image.copy()
    
    # 각 차량 및 번호판 결과 시각화
    for result in results:
        # 차량 박스 그리기
        vehicle_box = result.get('vehicle_box')
        if vehicle_box:
            x1, y1, x2, y2 = vehicle_box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, "Vehicle", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 번호판 박스 그리기
        plate_box = result.get('plate_box')
        if plate_box:
            # 차량 상대 좌표를 전체 이미지 좌표로 변환
            if vehicle_box:
                vx1, vy1 = vehicle_box[0], vehicle_box[1]
                px1, py1, px2, py2 = plate_box
                plate_box_global = [vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2]
            else:
                plate_box_global = plate_box
            
            x1, y1, x2, y2 = plate_box_global
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 번호판 텍스트 표시
            plate_text = result.get('plate_text', '')
            if plate_text:
                # 텍스트 배경 (가독성 향상)
                text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(vis_img, (x1, y1 - 25), (x1 + text_size[0], y1), (255, 0, 0), -1)
                # 텍스트
                cv2.putText(vis_img, plate_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis_img

def visualize_vehicle_detection(image, boxes):
    """
    차량 감지 결과 시각화
    
    Args:
        image (numpy.ndarray): 원본 이미지
        boxes (list): 감지된 차량의 바운딩 박스 목록 [x1, y1, x2, y2]
        
    Returns:
        numpy.ndarray: 차량 감지 결과가 시각화된 이미지
    """
    # 이미지 복사
    vis_img = image.copy()
    
    # 각 차량 박스 그리기
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, "Vehicle", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return vis_img

def visualize_plate_detection(image, boxes):
    """
    번호판 감지 결과 시각화
    
    Args:
        image (numpy.ndarray): 원본 이미지
        boxes (list): 감지된 번호판의 바운딩 박스 목록 [x1, y1, x2, y2]
        
    Returns:
        numpy.ndarray: 번호판 감지 결과가 시각화된 이미지
    """
    # 이미지 복사
    vis_img = image.copy()
    
    # 각 번호판 박스 그리기
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis_img, "License Plate", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return vis_img

def visualize_ocr_result(image, text, box=None):
    """
    OCR 결과 시각화
    
    Args:
        image (numpy.ndarray): 번호판 이미지
        text (str): 인식된 텍스트
        box (list, optional): 텍스트 영역 바운딩 박스 [x1, y1, x2, y2]. 기본값은 None
        
    Returns:
        numpy.ndarray: OCR 결과가 시각화된 이미지
    """
    # 이미지 복사
    vis_img = image.copy()
    
    # 텍스트 영역 표시 (박스가 있는 경우)
    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 인식된 텍스트 표시
    h, w = vis_img.shape[:2]
    # 텍스트 배경 (가독성 향상)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    cv2.rectangle(vis_img, (0, h - 35), (text_size[0] + 10, h), (0, 0, 0), -1)
    # 텍스트
    cv2.putText(vis_img, text, (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return vis_img

def visualize_preprocessing_steps(steps_dict):
    """
    전처리 단계별 시각화
    
    Args:
        steps_dict (dict): 전처리 단계별 이미지 딕셔너리
        
    Returns:
        numpy.ndarray: 모든 전처리 단계가 시각화된 이미지
    """
    # 시각화할 단계 수
    n_steps = len(steps_dict)
    
    # matplotlib 그림 생성
    fig = Figure(figsize=(15, 10))
    canvas = FigureCanvas(fig)
    
    # 서브플롯 생성
    rows = (n_steps + 2) // 3  # 3개 열에 맞게 행 수 계산
    axes = fig.subplots(rows, 3)
    axes = axes.flatten() if n_steps > 3 else [axes] if n_steps == 1 else axes
    
    # 각 단계 시각화
    for i, (step_name, step_img) in enumerate(steps_dict.items()):
        if i < len(axes):
            ax = axes[i]
            
            # 이미지가 그레이스케일인지 확인
            if len(step_img.shape) == 2 or step_img.shape[2] == 1:
                ax.imshow(step_img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(step_img, cv2.COLOR_BGR2RGB))
            
            ax.set_title(step_name)
            ax.axis('off')
    
    # 남은 서브플롯 숨기기
    for i in range(n_steps, len(axes)):
        axes[i].axis('off')
    
    # 그림 레이아웃 조정
    fig.tight_layout()
    
    # 캔버스를 이미지로 변환
    canvas.draw()
    vis_img = np.array(canvas.renderer.buffer_rgba())
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGBA2BGR)
    
    return vis_img

def create_comparison_image(original, processed, text=None):
    """
    원본 이미지와 처리된 이미지 비교 시각화
    
    Args:
        original (numpy.ndarray): 원본 이미지
        processed (numpy.ndarray): 처리된 이미지
        text (str, optional): 처리된 이미지에 표시할 텍스트. 기본값은 None
        
    Returns:
        numpy.ndarray: 비교 이미지
    """
    # 원본 이미지와 처리된 이미지의 크기 확인
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]
    
    # 두 이미지의 높이를 맞춤
    max_h = max(h1, h2)
    
    # 원본 이미지 리사이징
    if h1 != max_h:
        scale = max_h / h1
        new_w1 = int(w1 * scale)
        original = cv2.resize(original, (new_w1, max_h))
    else:
        new_w1 = w1
    
    # 처리된 이미지 리사이징
    if h2 != max_h:
        scale = max_h / h2
        new_w2 = int(w2 * scale)
        processed = cv2.resize(processed, (new_w2, max_h))
    else:
        new_w2 = w2
    
    # 텍스트가 있으면 처리된 이미지에 추가
    if text:
        # 처리된 이미지에 텍스트 추가
        # 텍스트 배경 (가독성 향상)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(processed, (0, 0), (text_size[0] + 10, text_size[1] + 10), (0, 0, 0), -1)
        # 텍스트
        cv2.putText(processed, text, (5, text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 두 이미지를 가로로 연결
    comparison = np.zeros((max_h, new_w1 + new_w2 + 10, 3), dtype=np.uint8)
    
    # 원본 이미지가 그레이스케일인 경우 3채널로 변환
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    # 처리된 이미지가 그레이스케일인 경우 3채널로 변환
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    # 이미지 합성
    comparison[:, :new_w1] = original
    comparison[:, new_w1+10:] = processed
    
    # 레이블 추가
    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Processed", (new_w1 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 구분선 추가
    cv2.line(comparison, (new_w1 + 5, 0), (new_w1 + 5, max_h), (255, 255, 255), 2)
    
    return comparison