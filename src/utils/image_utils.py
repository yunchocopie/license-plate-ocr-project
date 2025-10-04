import cv2
import numpy as np
from typing import Optional

"""
이미지 유틸리티 함수 모음

프로젝트 전체에서 사용되는 공통 이미지 처리 함수들을 제공합니다.
- 색공간 변환
- 이미지 검증
- 기본 전처리
"""


def ensure_gray(image: np.ndarray) -> np.ndarray:
    """
    이미지를 그레이스케일로 변환합니다.

    Args:
        image (np.ndarray): 입력 이미지 (그레이스케일 또는 컬러)

    Returns:
        np.ndarray: 그레이스케일 이미지

    Example:
        >>> gray_img = ensure_gray(bgr_image)
    """
    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if len(image.shape) == 3:
        # BGR to Gray
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        # 이미 그레이스케일
        return image.copy()
    else:
        raise ValueError(f"지원하지 않는 이미지 shape: {image.shape}")


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """
    이미지를 BGR 컬러로 변환합니다.

    Args:
        image (np.ndarray): 입력 이미지 (그레이스케일 또는 컬러)

    Returns:
        np.ndarray: BGR 컬러 이미지

    Example:
        >>> bgr_img = ensure_bgr(gray_image)
    """
    if image is None or image.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if len(image.shape) == 2:
        # Gray to BGR
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # 이미 BGR
        return image.copy()
    else:
        raise ValueError(f"지원하지 않는 이미지 shape: {image.shape}")


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """
    이미지를 RGB 컬러로 변환합니다 (Streamlit/PIL 표시용).

    Args:
        image (np.ndarray): 입력 이미지 (그레이스케일 또는 BGR)

    Returns:
        np.ndarray: RGB 컬러 이미지

    Example:
        >>> rgb_img = ensure_rgb(bgr_image)
        >>> st.image(rgb_img)
    """
    if image is None or image.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if len(image.shape) == 2:
        # Gray to RGB
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # BGR to RGB
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"지원하지 않는 이미지 shape: {image.shape}")


def is_valid_image(image: Optional[np.ndarray], min_width: int = 10, min_height: int = 10) -> bool:
    """
    이미지가 유효한지 검증합니다.

    Args:
        image: 검증할 이미지
        min_width: 최소 너비
        min_height: 최소 높이

    Returns:
        bool: 유효하면 True, 아니면 False

    Example:
        >>> if is_valid_image(img, min_width=30, min_height=10):
        ...     process_image(img)
    """
    if image is None:
        return False

    if image.size == 0:
        return False

    if len(image.shape) < 2:
        return False

    h, w = image.shape[:2]
    if h < min_height or w < min_width:
        return False

    return True


def resize_with_aspect_ratio(image: np.ndarray, target_width: Optional[int] = None,
                             target_height: Optional[int] = None,
                             max_width: Optional[int] = None,
                             max_height: Optional[int] = None) -> np.ndarray:
    """
    종횡비를 유지하면서 이미지 크기를 조정합니다.

    Args:
        image: 입력 이미지
        target_width: 목표 너비 (height는 자동 계산)
        target_height: 목표 높이 (width는 자동 계산)
        max_width: 최대 너비 제한
        max_height: 최대 높이 제한

    Returns:
        np.ndarray: 리사이즈된 이미지

    Example:
        >>> resized = resize_with_aspect_ratio(img, target_width=640)
    """
    if not is_valid_image(image):
        return image

    h, w = image.shape[:2]

    if target_width is not None:
        # 너비 기준으로 리사이즈
        aspect_ratio = h / w
        new_w = target_width
        new_h = int(target_width * aspect_ratio)
    elif target_height is not None:
        # 높이 기준으로 리사이즈
        aspect_ratio = w / h
        new_h = target_height
        new_w = int(target_height * aspect_ratio)
    else:
        new_w, new_h = w, h

    # 최대 크기 제한 적용
    if max_width is not None and new_w > max_width:
        aspect_ratio = new_h / new_w
        new_w = max_width
        new_h = int(max_width * aspect_ratio)

    if max_height is not None and new_h > max_height:
        aspect_ratio = new_w / new_h
        new_h = max_height
        new_w = int(max_height * aspect_ratio)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def normalize_brightness(image: np.ndarray) -> np.ndarray:
    """
    이미지 밝기를 정규화합니다.

    Args:
        image: 입력 이미지 (그레이스케일 권장)

    Returns:
        np.ndarray: 밝기 정규화된 이미지

    Example:
        >>> normalized = normalize_brightness(gray_img)
    """
    if not is_valid_image(image):
        return image

    gray = ensure_gray(image)

    # 히스토그램 평활화
    equalized = cv2.equalizeHist(gray)

    return equalized


def get_image_quality_metrics(image: np.ndarray) -> dict:
    """
    이미지 품질 메트릭을 계산합니다.

    Args:
        image: 입력 이미지

    Returns:
        dict: 품질 메트릭 딕셔너리
            - blur_measure: 블러 정도 (라플라시안 분산)
            - contrast: 대비 (표준편차)
            - brightness: 평균 밝기
            - resolution: 해상도 (픽셀 수)

    Example:
        >>> metrics = get_image_quality_metrics(img)
        >>> if metrics['blur_measure'] < 50:
        ...     print("이미지가 너무 흐립니다")
    """
    if not is_valid_image(image):
        return {
            'blur_measure': 0,
            'contrast': 0,
            'brightness': 0,
            'resolution': 0
        }

    gray = ensure_gray(image)

    # 블러 측정 (라플라시안 분산)
    blur_measure = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 대비 측정 (표준편차)
    contrast = np.std(gray)

    # 밝기 측정 (평균)
    brightness = np.mean(gray)

    # 해상도
    h, w = gray.shape
    resolution = h * w

    return {
        'blur_measure': float(blur_measure),
        'contrast': float(contrast),
        'brightness': float(brightness),
        'resolution': int(resolution)
    }
