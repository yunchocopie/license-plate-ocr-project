import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import Levenshtein
import time
from typing import Tuple, Dict

"""
성능 측정 도구 모듈

이 모듈은 다음과 같은 기능을 제공합니다:
1. 객체 감지 및 OCR 인식 성능 측정
2. 이미지 품질 평가 (선명도, 대비도, 노이즈 레벨)
3. 처리 시간 벤치마크
"""

# ==========================================
# 이미지 품질 평가 함수들
# ==========================================

def calculate_sharpness(image: np.ndarray) -> float:
    """
    이미지 선명도를 측정 (Laplacian 분산 방법)
    
    Args:
        image (numpy.ndarray): 입력 이미지 (그레이스케일 또는 컬러)
        
    Returns:
        float: 선명도 점수 (높을수록 선명함)
    """
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Laplacian 필터 적용
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # 분산 계산 (선명도 지표)
    sharpness = laplacian.var()
    
    return float(sharpness)

def calculate_contrast(image: np.ndarray) -> float:
    """
    이미지 대비도를 측정 (표준편차 방법)
    
    Args:
        image (numpy.ndarray): 입력 이미지 (그레이스케일 또는 컬러)
        
    Returns:
        float: 대비도 점수 (높을수록 대비가 좋음)
    """
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 표준편차 계산 (대비도 지표)
    contrast = gray.std()
    
    return float(contrast)

def calculate_noise_level(image: np.ndarray) -> float:
    """
    이미지 노이즈 레벨을 측정 (Bilateral Filter 비교 방법)
    
    Args:
        image (numpy.ndarray): 입력 이미지 (그레이스케일 또는 컬러)
        
    Returns:
        float: 노이즈 레벨 (높을수록 노이즈가 많음)
    """
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Bilateral Filter 적용 (노이즈 제거)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 원본과 필터링된 이미지의 차이 계산
    noise = np.abs(gray.astype(np.float32) - filtered.astype(np.float32))
    
    # 노이즈 레벨 (평균 절대 차이)
    noise_level = noise.mean()
    
    return float(noise_level)

def calculate_brightness(image: np.ndarray) -> float:
    """
    이미지 밝기를 측정
    
    Args:
        image (numpy.ndarray): 입력 이미지 (그레이스케일 또는 컬러)
        
    Returns:
        float: 밝기 점수 (0-255 범위)
    """
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 평균 밝기 계산
    brightness = gray.mean()
    
    return float(brightness)

def assess_image_quality(image: np.ndarray) -> Dict[str, float]:
    """
    이미지의 전반적인 품질을 평가
    
    Args:
        image (numpy.ndarray): 입력 이미지
        
    Returns:
        dict: 품질 평가 결과
            - sharpness: 선명도 점수
            - contrast: 대비도 점수
            - noise_level: 노이즈 레벨
            - brightness: 밝기 점수
            - overall_score: 종합 품질 점수 (0-100)
    """
    # 개별 품질 지표 계산
    sharpness = calculate_sharpness(image)
    contrast = calculate_contrast(image)
    noise_level = calculate_noise_level(image)
    brightness = calculate_brightness(image)
    
    # 정규화를 위한 기준값들 (경험적 설정)
    sharpness_max = 1000.0  # 일반적인 선명한 이미지의 최대값
    contrast_max = 80.0     # 일반적인 대비 좋은 이미지의 최대값
    noise_max = 20.0        # 일반적인 노이즈 많은 이미지의 최대값
    brightness_optimal = 127.5  # 최적 밝기 (중간값)
    
    # 정규화 (0-1 범위)
    sharpness_norm = min(sharpness / sharpness_max, 1.0)
    contrast_norm = min(contrast / contrast_max, 1.0)
    noise_norm = max(0.0, 1.0 - (noise_level / noise_max))  # 노이즈는 낮을수록 좋음
    
    # 밝기는 중간값(127.5)에서 멀어질수록 점수 감소
    brightness_norm = 1.0 - abs(brightness - brightness_optimal) / brightness_optimal
    brightness_norm = max(0.0, brightness_norm)
    
    # 가중 평균으로 종합 점수 계산
    weights = {
        'sharpness': 0.35,    # 선명도가 가장 중요
        'contrast': 0.25,     # 대비도 중요
        'noise': 0.25,        # 노이즈 레벨 중요
        'brightness': 0.15    # 밝기는 상대적으로 덜 중요
    }
    
    overall_score = (
        sharpness_norm * weights['sharpness'] +
        contrast_norm * weights['contrast'] +
        noise_norm * weights['noise'] +
        brightness_norm * weights['brightness']
    ) * 100.0  # 0-100 스케일로 변환
    
    return {
        'sharpness': sharpness,
        'contrast': contrast,
        'noise_level': noise_level,
        'brightness': brightness,
        'sharpness_norm': sharpness_norm,
        'contrast_norm': contrast_norm,
        'noise_norm': noise_norm,
        'brightness_norm': brightness_norm,
        'overall_score': overall_score
    }

def determine_processing_level(quality_metrics: Dict[str, float]) -> str:
    """
    품질 평가 결과를 바탕으로 처리 수준을 결정
    
    Args:
        quality_metrics (dict): assess_image_quality()의 결과
        
    Returns:
        str: 처리 수준 ('minimal', 'moderate', 'full')
    """
    overall_score = quality_metrics['overall_score']
    sharpness_norm = quality_metrics['sharpness_norm']
    noise_norm = quality_metrics['noise_norm']
    
    # 고품질 이미지 (최소 처리)
    if overall_score >= 75 and sharpness_norm >= 0.8 and noise_norm >= 0.8:
        return 'minimal'
    
    # 중간 품질 이미지 (적당한 처리)
    elif overall_score >= 45 and sharpness_norm >= 0.4:
        return 'moderate'
    
    # 저품질 이미지 (전체 처리)
    else:
        return 'full'

# ==========================================
# 기존 성능 측정 함수들
# ==========================================

def calculate_iou(box1, box2):
    """
    두 박스 간의 IoU(Intersection over Union) 계산
    
    Args:
        box1 (list): 첫 번째 박스 좌표 [x1, y1, x2, y2]
        box2 (list): 두 번째 박스 좌표 [x1, y1, x2, y2]
        
    Returns:
        float: IoU 값 (0~1)
    """
    # 박스 좌표 추출
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 교차 영역 계산
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # 교차 영역이 없는 경우
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    # 교차 영역의 넓이
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 각 박스의 넓이
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 합집합 영역의 넓이
    union_area = box1_area + box2_area - intersection_area
    
    # IoU 계산
    iou = intersection_area / union_area
    
    return iou

def calculate_detection_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    객체 감지 성능 지표 계산
    
    Args:
        pred_boxes (list): 예측된 박스 목록 [x1, y1, x2, y2]
        gt_boxes (list): 실제 박스 목록 [x1, y1, x2, y2]
        iou_threshold (float, optional): IoU 임계값. 기본값은 0.5
        
    Returns:
        dict: 성능 지표 (precision, recall, f1_score, mAP)
    """
    # 예측 및 실제 박스가 없는 경우
    if not pred_boxes and not gt_boxes:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0,
            'mAP': 1.0
        }
    elif not pred_boxes:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mAP': 0.0
        }
    elif not gt_boxes:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mAP': 0.0
        }
    
    # IoU 행렬 계산
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou_matrix[i, j] = calculate_iou(gt_box, pred_box)
    
    # 각 실제 박스에 대한 최대 IoU를 가진 예측 박스 찾기
    matched_indices = []
    for i in range(len(gt_boxes)):
        # IoU가 임계값 이상인 예측 박스 중 최대 IoU를 가진 박스 선택
        max_j = -1
        max_iou = iou_threshold
        for j in range(len(pred_boxes)):
            if j not in [idx[1] for idx in matched_indices]:  # 이미 매칭된 예측 박스 제외
                if iou_matrix[i, j] > max_iou:
                    max_j = j
                    max_iou = iou_matrix[i, j]
        
        if max_j != -1:
            matched_indices.append((i, max_j))
    
    # True Positives, False Positives, False Negatives 계산
    tp = len(matched_indices)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    
    # 성능 지표 계산
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    # mAP 계산 (간소화된 버전)
    ap = precision * recall
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mAP': ap
    }

def calculate_text_similarity(pred_text, gt_text):
    """
    예측 텍스트와 실제 텍스트 간의 유사도 계산
    
    Args:
        pred_text (str): 예측된 텍스트
        gt_text (str): 실제 텍스트
        
    Returns:
        float: 텍스트 유사도 (0~1)
    """
    # 모두 비어있는 경우
    if not pred_text and not gt_text:
        return 1.0
    # 둘 중 하나가 비어있는 경우
    elif not pred_text or not gt_text:
        return 0.0
    
    # 레벤슈타인 거리 계산
    distance = Levenshtein.distance(pred_text, gt_text)
    
    # 최대 거리 (두 문자열 중 긴 것의 길이)
    max_length = max(len(pred_text), len(gt_text))
    
    # 유사도 계산 (1 - 정규화된 거리)
    similarity = 1 - (distance / max_length)
    
    return similarity

def calculate_ocr_metrics(pred_texts, gt_texts):
    """
    OCR 인식 성능 지표 계산
    
    Args:
        pred_texts (list): 예측된 텍스트 목록
        gt_texts (list): 실제 텍스트 목록
        
    Returns:
        dict: 성능 지표 (accuracy, avg_similarity, character_accuracy)
    """
    # 텍스트 목록 길이가 다른 경우 조정
    if len(pred_texts) != len(gt_texts):
        min_len = min(len(pred_texts), len(gt_texts))
        pred_texts = pred_texts[:min_len]
        gt_texts = gt_texts[:min_len]
    
    # 정확히 일치하는 텍스트 비율 (accuracy)
    exact_matches = sum(1 for p, g in zip(pred_texts, gt_texts) if p == g)
    accuracy = exact_matches / len(gt_texts) if gt_texts else 0
    
    # 평균 텍스트 유사도
    similarities = [calculate_text_similarity(p, g) for p, g in zip(pred_texts, gt_texts)]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    
    # 문자 단위 정확도
    total_chars = 0
    correct_chars = 0
    
    for p, g in zip(pred_texts, gt_texts):
        # 문자 단위로 비교하기 위해 문자열 길이 조정
        max_len = max(len(p), len(g))
        p_padded = p.ljust(max_len)
        g_padded = g.ljust(max_len)
        
        # 문자별 비교
        for p_char, g_char in zip(p_padded, g_padded):
            total_chars += 1
            if p_char == g_char:
                correct_chars += 1
    
    character_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    
    return {
        'accuracy': accuracy,
        'avg_similarity': avg_similarity,
        'character_accuracy': character_accuracy
    }

def measure_processing_time(func, *args, **kwargs):
    """
    함수 실행 시간 측정
    
    Args:
        func (callable): 측정할 함수
        *args: 함수 인자
        **kwargs: 함수 키워드 인자
        
    Returns:
        tuple: (함수 결과값, 실행 시간(초))
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    return result, execution_time

def benchmark_pipeline(pipeline, test_images, gt_boxes=None, gt_texts=None, num_runs=1):
    """
    전체 파이프라인 성능 벤치마크
    
    Args:
        pipeline (callable): 벤치마크할 파이프라인 함수
        test_images (list): 테스트 이미지 목록
        gt_boxes (list, optional): 실제 박스 목록. 기본값은 None
        gt_texts (list, optional): 실제 텍스트 목록. 기본값은 None
        num_runs (int, optional): 실행 횟수. 기본값은 1
        
    Returns:
        dict: 벤치마크 결과
    """
    # 결과 저장 변수
    execution_times = []
    detection_metrics = []
    ocr_metrics = []
    
    # 각 이미지에 대해 파이프라인 실행
    for i, image in enumerate(test_images):
        # 여러 번 실행하여 평균 시간 계산
        run_times = []
        results = None
        
        for _ in range(num_runs):
            result, exec_time = measure_processing_time(pipeline, image)
            run_times.append(exec_time)
            if _ == 0:  # 첫 번째 실행 결과만 성능 측정에 사용
                results = result
        
        # 평균 실행 시간
        avg_exec_time = sum(run_times) / len(run_times)
        execution_times.append(avg_exec_time)
        
        # 감지 성능 측정 (gt_boxes가 있는 경우)
        if gt_boxes and i < len(gt_boxes):
            pred_boxes = [r.get('plate_box') for r in results if 'plate_box' in r]
            metrics = calculate_detection_metrics(pred_boxes, gt_boxes[i])
            detection_metrics.append(metrics)
        
        # OCR 성능 측정 (gt_texts가 있는 경우)
        if gt_texts and i < len(gt_texts):
            pred_texts = [r.get('plate_text', '') for r in results if 'plate_text' in r]
            metrics = calculate_ocr_metrics(pred_texts, gt_texts[i])
            ocr_metrics.append(metrics)
    
    # 결과 종합
    result = {
        'execution_time': {
            'mean': np.mean(execution_times),
            'std': np.std(execution_times),
            'min': min(execution_times),
            'max': max(execution_times)
        }
    }
    
    # 감지 성능 종합
    if detection_metrics:
        result['detection'] = {
            'precision': np.mean([m['precision'] for m in detection_metrics]),
            'recall': np.mean([m['recall'] for m in detection_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in detection_metrics]),
            'mAP': np.mean([m['mAP'] for m in detection_metrics])
        }
    
    # OCR 성능 종합
    if ocr_metrics:
        result['ocr'] = {
            'accuracy': np.mean([m['accuracy'] for m in ocr_metrics]),
            'avg_similarity': np.mean([m['avg_similarity'] for m in ocr_metrics]),
            'character_accuracy': np.mean([m['character_accuracy'] for m in ocr_metrics])
        }
    
    return result
