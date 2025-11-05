"""
이미지 품질 분석 모듈

번호판 이미지의 blur, contrast, noise 등 품질 지표를 계산합니다.
docs/ocr_structured_pipeline_plan.md 기반 구현
"""

import cv2
import numpy as np
from typing import Dict


def analyze_quality(image: np.ndarray) -> Dict[str, float]:
    """
    이미지 품질 분석

    다음 지표들을 계산합니다:
    - blur: 블러 측정값 (Laplacian variance)
    - contrast: 대비 (표준편차)
    - noise: 노이즈 레벨 (high-pass filter의 표준편차)

    Args:
        image: 입력 이미지 (BGR 또는 Grayscale)

    Returns:
        quality_metrics: 품질 지표 딕셔너리
            {
                "blur": float,      # 높을수록 선명 (100 미만이면 흐림)
                "contrast": float,  # 높을수록 대비 좋음 (30 미만이면 저대비)
                "noise": float      # 높을수록 노이즈 많음 (35 초과시 noisy)
            }
    """
    # Grayscale 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 1. Blur 측정 (Laplacian variance)
    # 라플라시안 연산자는 2차 미분으로 엣지를 검출
    # 분산이 낮으면 엣지가 약함 = 블러된 이미지
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur = float(laplacian.var())

    # 2. Contrast 측정 (표준편차)
    # 표준편차가 낮으면 픽셀값이 평균 근처에 모여있음 = 저대비
    contrast = float(gray.std())

    # 3. Noise 측정 (high-pass filter의 표준편차)
    # 라플라시안을 노이즈 측정에도 활용 (고주파 성분)
    # 16비트로 변환하여 정밀도 향상
    highpass = cv2.Laplacian(gray, cv2.CV_16S)
    noise = float(highpass.std())

    return {
        "blur": blur,
        "contrast": contrast,
        "noise": noise
    }


def calculate_sharpness(image: np.ndarray) -> float:
    """
    이미지 선명도 계산 (Laplacian variance)

    Args:
        image: 입력 이미지

    Returns:
        sharpness: 선명도 값 (높을수록 선명)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_contrast(image: np.ndarray) -> float:
    """
    이미지 대비 계산 (표준편차)

    Args:
        image: 입력 이미지

    Returns:
        contrast: 대비 값 (높을수록 대비 좋음)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return float(gray.std())


def calculate_brightness(image: np.ndarray) -> float:
    """
    이미지 밝기 계산 (평균값)

    Args:
        image: 입력 이미지

    Returns:
        brightness: 밝기 값 (0~255)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return float(gray.mean())


def is_good_quality(quality: Dict[str, float],
                   blur_threshold: float = 100.0,
                   contrast_threshold: float = 30.0,
                   noise_threshold: float = 35.0) -> bool:
    """
    품질이 양호한지 판단

    Args:
        quality: analyze_quality() 결과
        blur_threshold: 블러 임계값
        contrast_threshold: 대비 임계값
        noise_threshold: 노이즈 임계값

    Returns:
        is_good: True if 품질 양호
    """
    return (
        quality["blur"] >= blur_threshold and
        quality["contrast"] >= contrast_threshold and
        quality["noise"] <= noise_threshold
    )


def get_quality_summary(quality: Dict[str, float]) -> str:
    """
    품질 지표를 사람이 읽을 수 있는 문자열로 변환

    Args:
        quality: analyze_quality() 결과

    Returns:
        summary: 품질 요약 문자열
    """
    issues = []

    if quality["blur"] < 100:
        issues.append(f"흐림(blur={quality['blur']:.1f})")

    if quality["contrast"] < 30:
        issues.append(f"저대비(contrast={quality['contrast']:.1f})")

    if quality["noise"] > 35:
        issues.append(f"노이즈(noise={quality['noise']:.1f})")

    if not issues:
        return "양호"

    return ", ".join(issues)


def compare_quality(before: np.ndarray, after: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    전처리 전후 품질 비교

    Args:
        before: 전처리 전 이미지
        after: 전처리 후 이미지

    Returns:
        comparison: 비교 결과
            {
                "before": {"blur": float, "contrast": float, "noise": float},
                "after": {"blur": float, "contrast": float, "noise": float},
                "improvement": {"blur": float, "contrast": float, "noise": float}
            }
    """
    before_quality = analyze_quality(before)
    after_quality = analyze_quality(after)

    improvement = {
        "blur": after_quality["blur"] - before_quality["blur"],
        "contrast": after_quality["contrast"] - before_quality["contrast"],
        "noise": before_quality["noise"] - after_quality["noise"]  # 노이즈는 감소가 개선
    }

    return {
        "before": before_quality,
        "after": after_quality,
        "improvement": improvement
    }
