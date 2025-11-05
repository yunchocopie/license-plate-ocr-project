"""
적응형 전처리 파이프라인 모듈

이미지 품질에 따라 적절한 전처리 파이프라인을 선택하고 적용합니다.
docs/ocr_structured_pipeline_plan.md 기반 구현
"""

import cv2
import numpy as np
from typing import Callable, List, Dict, Tuple
from .quality import analyze_quality
from .warp import TemplateMeta


# 전처리 함수 정의

def resize_to_template(image: np.ndarray, template_meta: TemplateMeta) -> np.ndarray:
    """
    템플릿 크기로 리사이즈 (이미 워프된 경우 생략 가능)

    Args:
        image: 입력 이미지
        template_meta: 템플릿 메타데이터

    Returns:
        resized: 리사이즈된 이미지
    """
    width, height = template_meta.size
    current_h, current_w = image.shape[:2]

    # 이미 동일한 크기면 리사이즈 생략
    if current_h == height and current_w == width:
        return image

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def normalize(image: np.ndarray, template_meta: TemplateMeta = None) -> np.ndarray:
    """
    이미지 정규화 (0~255 범위로 스케일링)

    Args:
        image: 입력 이미지
        template_meta: 템플릿 메타데이터 (사용 안함, 인터페이스 통일용)

    Returns:
        normalized: 정규화된 이미지
    """
    # Grayscale 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Min-Max 정규화
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    return normalized.astype(np.uint8)


def clahe(image: np.ndarray, template_meta: TemplateMeta = None,
          clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용

    저대비 이미지의 대비를 향상시킵니다.

    Args:
        image: 입력 이미지
        template_meta: 템플릿 메타데이터 (사용 안함)
        clip_limit: 대비 제한 값
        tile_grid_size: 타일 그리드 크기

    Returns:
        enhanced: 대비가 향상된 이미지
    """
    # Grayscale 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # CLAHE 적용
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe_obj.apply(gray)

    return enhanced


def unsharp_mask(image: np.ndarray, template_meta: TemplateMeta = None,
                 kernel_size: Tuple[int, int] = (5, 5), sigma: float = 1.0,
                 amount: float = 1.5, threshold: int = 0) -> np.ndarray:
    """
    Unsharp Mask 필터로 선명도 향상

    흐린 이미지를 선명하게 만듭니다.

    Args:
        image: 입력 이미지
        template_meta: 템플릿 메타데이터 (사용 안함)
        kernel_size: 가우시안 블러 커널 크기
        sigma: 가우시안 블러 시그마
        amount: 선명도 강도 (1.0~2.0 권장)
        threshold: 임계값

    Returns:
        sharpened: 선명도가 향상된 이미지
    """
    # Grayscale 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 가우시안 블러로 흐린 버전 생성
    blurred = cv2.GaussianBlur(gray, kernel_size, sigma)

    # Unsharp mask 공식: sharpened = original + amount * (original - blurred)
    sharpened = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)

    # 임계값 적용 (선택적)
    if threshold > 0:
        low_contrast_mask = np.absolute(gray - blurred) < threshold
        sharpened = np.where(low_contrast_mask, gray, sharpened)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def bilateral_filter(image: np.ndarray, template_meta: TemplateMeta = None,
                    d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    양방향 필터로 노이즈 제거 (엣지 보존)

    노이즈가 많은 이미지에서 엣지를 보존하면서 노이즈를 제거합니다.

    Args:
        image: 입력 이미지
        template_meta: 템플릿 메타데이터 (사용 안함)
        d: 필터 크기 (직경)
        sigma_color: 색상 공간 시그마
        sigma_space: 좌표 공간 시그마

    Returns:
        denoised: 노이즈가 제거된 이미지
    """
    # Grayscale 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 양방향 필터 적용
    denoised = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)

    return denoised


def adaptive_threshold(image: np.ndarray, template_meta: TemplateMeta = None,
                      block_size: int = 15, c: int = 8) -> np.ndarray:
    """
    적응형 임계값 이진화

    Args:
        image: 입력 이미지
        template_meta: 템플릿 메타데이터 (사용 안함)
        block_size: 블록 크기 (홀수)
        c: 상수값

    Returns:
        binary: 이진화된 이미지
    """
    # Grayscale 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 적응형 임계값 적용
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c
    )

    return binary


# 파이프라인 정의
# 각 파이프라인은 함수 리스트로 구성됨
# 순서대로 적용됨

PIPELINES: Dict[str, List[Callable]] = {
    "default": [
        normalize
    ],

    "low_contrast": [
        clahe,
        unsharp_mask,
        normalize
    ],

    "blurry": [
        unsharp_mask,
        normalize
    ],

    "noisy": [
        bilateral_filter,
        normalize
    ],

    "multi_issue": [
        bilateral_filter,
        clahe,
        unsharp_mask,
        normalize
    ]
}


def select_pipeline(quality: Dict[str, float],
                   blur_threshold: float = 120.0,
                   contrast_threshold: float = 20.0,
                   noise_threshold: float = 35.0,
                   adaptive_enabled: bool = True) -> str:
    """
    품질 지표를 기반으로 전처리 파이프라인 선택

    Args:
        quality: analyze_quality() 결과
        blur_threshold: 블러 임계값 (이 값보다 낮으면 blurry)
        contrast_threshold: 대비 임계값 (이 값보다 낮으면 low_contrast)
        noise_threshold: 노이즈 임계값 (이 값보다 높으면 noisy)
        adaptive_enabled: 적응형 전처리 활성화 여부

    Returns:
        pipeline_key: 선택된 파이프라인 키 ("default", "low_contrast", "blurry", "noisy", "multi_issue")
    """
    if not adaptive_enabled:
        return "default"

    issues = []

    if quality["blur"] < blur_threshold:
        issues.append("blurry")

    if quality["contrast"] < contrast_threshold:
        issues.append("low_contrast")

    if quality["noise"] > noise_threshold:
        issues.append("noisy")

    # 문제가 없으면 기본 파이프라인
    if not issues:
        return "default"

    # 문제가 2개 이상이면 복합 파이프라인
    if len(issues) > 1:
        return "multi_issue"

    # 문제가 1개면 해당 파이프라인
    return issues[0]


def apply_pipeline(image: np.ndarray, pipeline_key: str, template_meta: TemplateMeta) -> np.ndarray:
    """
    파이프라인 적용

    Args:
        image: 입력 이미지
        pipeline_key: 파이프라인 키
        template_meta: 템플릿 메타데이터

    Returns:
        processed: 전처리된 이미지
    """
    if pipeline_key not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_key}. Available: {list(PIPELINES.keys())}")

    pipeline = PIPELINES[pipeline_key]
    processed = image.copy()

    for func in pipeline:
        processed = func(processed, template_meta)

    return processed


def preprocess_plate_image(warped_image: np.ndarray, template_meta: TemplateMeta,
                           blur_threshold: float = 120.0,
                           contrast_threshold: float = 20.0,
                           noise_threshold: float = 35.0,
                           adaptive_enabled: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    번호판 이미지 전처리 (엔트리 포인트)

    워프된 번호판 이미지를 분석하고 적절한 전처리 파이프라인을 적용합니다.

    Args:
        warped_image: 워프된 번호판 이미지
        template_meta: 템플릿 메타데이터
        blur_threshold: 블러 임계값
        contrast_threshold: 대비 임계값
        noise_threshold: 노이즈 임계값
        adaptive_enabled: 적응형 전처리 활성화 여부

    Returns:
        processed: 전처리된 이미지
        meta: 메타데이터
            {
                "quality": {...},           # 품질 지표
                "pipeline": str,            # 선택된 파이프라인
                "template": str             # 템플릿 타입
            }
    """
    # 1. 품질 분석
    quality = analyze_quality(warped_image)

    # 2. 파이프라인 선택
    pipeline_key = select_pipeline(
        quality,
        blur_threshold=blur_threshold,
        contrast_threshold=contrast_threshold,
        noise_threshold=noise_threshold,
        adaptive_enabled=adaptive_enabled
    )

    # 3. 파이프라인 적용
    processed = apply_pipeline(warped_image, pipeline_key, template_meta)

    # 4. 메타데이터 생성
    meta = {
        "quality": quality,
        "pipeline": pipeline_key,
        "template": template_meta.plate_type
    }

    return processed, meta
