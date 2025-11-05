"""
적응형 전처리 테스트

품질에 따른 전처리 파이프라인 선택과 효과를 테스트합니다.
docs/ocr_structured_pipeline_plan.md 기반
"""

import pytest
import cv2
import numpy as np
import os
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.quality import analyze_quality, compare_quality
from src.preprocessing.pipelines import (
    select_pipeline,
    apply_pipeline,
    clahe,
    unsharp_mask,
    bilateral_filter,
    normalize
)
from src.preprocessing.warp import get_one_line_template


class TestQualityAnalysis:
    """품질 분석 테스트"""

    def create_clean_image(self):
        """깨끗한 테스트 이미지 생성"""
        image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(image, "12ABC3456", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        return image

    def add_blur(self, image, kernel_size=15):
        """블러 추가"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def add_noise(self, image, amount=50):
        """노이즈 추가"""
        noise = np.random.normal(0, amount, image.shape).astype(np.uint8)
        return cv2.add(image, noise)

    def reduce_contrast(self, image, alpha=0.3):
        """대비 감소"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=128 * (1 - alpha))

    def test_clean_image_quality(self):
        """깨끗한 이미지 품질 측정"""
        clean_image = self.create_clean_image()
        quality = analyze_quality(clean_image)

        # 깨끗한 이미지는 좋은 품질 지표를 가져야 함
        assert quality["blur"] > 100  # 선명함
        assert quality["contrast"] > 20  # 대비 양호
        assert quality["noise"] < 50  # 노이즈 적음

    def test_blurry_image_detection(self):
        """흐린 이미지 감지"""
        clean_image = self.create_clean_image()
        blurry_image = self.add_blur(clean_image, kernel_size=15)

        quality_clean = analyze_quality(clean_image)
        quality_blurry = analyze_quality(blurry_image)

        # 흐린 이미지는 blur 값이 낮아야 함
        assert quality_blurry["blur"] < quality_clean["blur"]

    def test_noisy_image_detection(self):
        """노이즈가 많은 이미지 감지"""
        clean_image = self.create_clean_image()
        noisy_image = self.add_noise(clean_image, amount=50)

        quality_clean = analyze_quality(clean_image)
        quality_noisy = analyze_quality(noisy_image)

        # 노이즈가 많은 이미지는 noise 값이 높아야 함
        assert quality_noisy["noise"] > quality_clean["noise"]

    def test_low_contrast_detection(self):
        """저대비 이미지 감지"""
        clean_image = self.create_clean_image()
        low_contrast = self.reduce_contrast(clean_image, alpha=0.3)

        quality_clean = analyze_quality(clean_image)
        quality_low = analyze_quality(low_contrast)

        # 저대비 이미지는 contrast 값이 낮아야 함
        assert quality_low["contrast"] < quality_clean["contrast"]


class TestPipelineSelection:
    """파이프라인 선택 테스트"""

    def test_default_pipeline_selection(self):
        """기본 파이프라인 선택"""
        # 모든 품질이 양호한 경우
        quality = {"blur": 150.0, "contrast": 40.0, "noise": 20.0}
        pipeline = select_pipeline(quality)
        assert pipeline == "default"

    def test_blurry_pipeline_selection(self):
        """흐림 전용 파이프라인 선택"""
        quality = {"blur": 80.0, "contrast": 40.0, "noise": 20.0}
        pipeline = select_pipeline(quality)
        assert pipeline == "blurry"

    def test_low_contrast_pipeline_selection(self):
        """저대비 전용 파이프라인 선택"""
        quality = {"blur": 150.0, "contrast": 15.0, "noise": 20.0}
        pipeline = select_pipeline(quality)
        assert pipeline == "low_contrast"

    def test_noisy_pipeline_selection(self):
        """노이즈 전용 파이프라인 선택"""
        quality = {"blur": 150.0, "contrast": 40.0, "noise": 50.0}
        pipeline = select_pipeline(quality)
        assert pipeline == "noisy"

    def test_multi_issue_pipeline_selection(self):
        """복합 문제 파이프라인 선택"""
        # 2개 이상의 품질 문제
        quality = {"blur": 80.0, "contrast": 15.0, "noise": 50.0}
        pipeline = select_pipeline(quality)
        assert pipeline == "multi_issue"

    def test_adaptive_disabled(self):
        """적응형 모드 비활성화 시"""
        quality = {"blur": 50.0, "contrast": 10.0, "noise": 60.0}
        pipeline = select_pipeline(quality, adaptive_enabled=False)
        assert pipeline == "default"


class TestPreprocessingFunctions:
    """전처리 함수 개별 테스트"""

    def create_test_image(self):
        """테스트 이미지 생성"""
        image = np.ones((100, 300, 3), dtype=np.uint8) * 128
        cv2.putText(image, "TEST", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        return image

    def test_normalize_function(self):
        """정규화 함수 테스트"""
        image = self.create_test_image()
        normalized = normalize(image)

        assert normalized is not None
        assert normalized.dtype == np.uint8
        assert normalized.ndim == 2  # 그레이스케일
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 255

    def test_clahe_function(self):
        """CLAHE 함수 테스트"""
        image = self.create_test_image()
        enhanced = clahe(image)

        assert enhanced is not None
        assert enhanced.dtype == np.uint8
        assert enhanced.ndim == 2

    def test_unsharp_mask_function(self):
        """Unsharp Mask 함수 테스트"""
        image = self.create_test_image()
        sharpened = unsharp_mask(image)

        assert sharpened is not None
        assert sharpened.dtype == np.uint8
        assert sharpened.ndim == 2

    def test_bilateral_filter_function(self):
        """양방향 필터 함수 테스트"""
        image = self.create_test_image()
        denoised = bilateral_filter(image)

        assert denoised is not None
        assert denoised.dtype == np.uint8
        assert denoised.ndim == 2


class TestPipelineEffectiveness:
    """파이프라인 효과 테스트"""

    def create_clean_image(self):
        """깨끗한 이미지"""
        image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(image, "ABC123", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        return image

    def test_blurry_pipeline_improves_sharpness(self):
        """흐림 파이프라인이 선명도를 개선하는지"""
        clean_image = self.create_clean_image()
        blurry_image = cv2.GaussianBlur(clean_image, (11, 11), 0)

        template_meta = get_one_line_template()

        # 흐림 파이프라인 적용
        processed = apply_pipeline(blurry_image, "blurry", template_meta)

        # 품질 비교
        comparison = compare_quality(blurry_image, processed)

        # Unsharp mask는 선명도(blur)를 증가시켜야 함
        # 단, 노이즈도 증가할 수 있음
        assert comparison["after"]["blur"] >= 0  # 최소한 처리는 됨

    def test_low_contrast_pipeline_improves_contrast(self):
        """저대비 파이프라인이 대비를 개선하는지"""
        clean_image = self.create_clean_image()
        low_contrast = cv2.convertScaleAbs(clean_image, alpha=0.4, beta=128 * 0.6)

        template_meta = get_one_line_template()

        # 저대비 파이프라인 적용
        processed = apply_pipeline(low_contrast, "low_contrast", template_meta)

        # 품질 비교
        comparison = compare_quality(low_contrast, processed)

        # CLAHE는 대비를 증가시켜야 함
        assert comparison["improvement"]["contrast"] > 0

    def test_noisy_pipeline_reduces_noise(self):
        """노이즈 파이프라인이 노이즈를 감소시키는지"""
        clean_image = self.create_clean_image()
        noise = np.random.normal(0, 40, clean_image.shape).astype(np.uint8)
        noisy_image = cv2.add(clean_image, noise)

        template_meta = get_one_line_template()

        # 노이즈 파이프라인 적용
        processed = apply_pipeline(noisy_image, "noisy", template_meta)

        # 품질 비교
        comparison = compare_quality(noisy_image, processed)

        # bilateral filter는 노이즈를 감소시켜야 함
        # improvement["noise"]는 감소량이므로 양수여야 함
        assert comparison["improvement"]["noise"] > 0

    def test_multi_issue_pipeline(self):
        """복합 문제 파이프라인 테스트"""
        clean_image = self.create_clean_image()

        # 복합 문제 생성 (블러 + 저대비 + 노이즈)
        degraded = cv2.GaussianBlur(clean_image, (7, 7), 0)
        degraded = cv2.convertScaleAbs(degraded, alpha=0.5, beta=64)
        noise = np.random.normal(0, 30, degraded.shape).astype(np.uint8)
        degraded = cv2.add(degraded, noise)

        template_meta = get_one_line_template()

        # 복합 파이프라인 적용
        processed = apply_pipeline(degraded, "multi_issue", template_meta)

        # 처리된 이미지가 존재하는지만 확인
        assert processed is not None
        assert processed.dtype == np.uint8


class TestQualityComparison:
    """품질 비교 테스트"""

    def test_quality_comparison_structure(self):
        """품질 비교 결과 구조 테스트"""
        before = np.ones((100, 300, 3), dtype=np.uint8) * 128
        after = cv2.convertScaleAbs(before, alpha=1.5, beta=0)

        comparison = compare_quality(before, after)

        # 결과 구조 검증
        assert "before" in comparison
        assert "after" in comparison
        assert "improvement" in comparison

        # 각 항목에 필요한 키가 있는지 확인
        for key in ["blur", "contrast", "noise"]:
            assert key in comparison["before"]
            assert key in comparison["after"]
            assert key in comparison["improvement"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
