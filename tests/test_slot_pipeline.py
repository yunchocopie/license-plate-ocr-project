"""
슬롯 기반 OCR 파이프라인 테스트

docs/ocr_structured_pipeline_plan.md의 전체 파이프라인을 테스트합니다.
"""

import pytest
import cv2
import numpy as np
import os
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.preprocessing.warp import process_plate_warp, select_template
from src.preprocessing.pipelines import preprocess_plate_image
from src.preprocessing.quality import analyze_quality

# src.ocr 패키지를 import하지 않고 직접 파일 import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'ocr')))
from korean_plate_postprocessor import KoreanPlatePostProcessor

# EasyOCR이 없을 수 있으므로 조건부 import
try:
    from src.ocr.slot_classifier import SlotClassifier, recognize_plate_slots
    SLOT_CLASSIFIER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SLOT_CLASSIFIER_AVAILABLE = False
    SlotClassifier = None
    recognize_plate_slots = None


class TestSlotPipeline:
    """슬롯 기반 파이프라인 통합 테스트"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """테스트 설정"""
        self.postprocessor = KoreanPlatePostProcessor()
        if SLOT_CLASSIFIER_AVAILABLE:
            self.slot_classifier = SlotClassifier(use_easyocr=True)
        else:
            self.slot_classifier = None

    def create_test_image(self, width=520, height=110):
        """테스트용 더미 이미지 생성"""
        # 흰색 배경
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 검정색 테두리
        cv2.rectangle(image, (0, 0), (width-1, height-1), (0, 0, 0), 2)

        return image

    def test_template_selection(self):
        """템플릿 선택 테스트"""
        # 1행 번호판 (4.7:1 비율)
        template = select_template(520, 110)
        assert template.plate_type == "ONE_LINE"
        assert len(template.slots) == 7  # 숫자2 + 한글1 + 숫자4

        # 2행 번호판 (1.9:1 비율)
        template = select_template(340, 180)
        assert template.plate_type == "TWO_LINE"

        # 소형 2행 (1.1:1 비율)
        template = select_template(280, 260)
        assert template.plate_type == "TWO_LINE_SMALL"

        # 이륜차 (0.9:1 비율)
        template = select_template(200, 220)
        assert template.plate_type == "MOTORCYCLE"

    def test_warp_pipeline(self):
        """워프 파이프라인 테스트"""
        # 테스트 이미지 생성
        test_image = self.create_test_image(520, 110)

        # 워프 처리
        warped, template_meta = process_plate_warp(test_image, debug=False)

        # 검증
        assert warped is not None
        assert warped.shape[0] == template_meta.size[1]  # height
        assert warped.shape[1] == template_meta.size[0]  # width
        assert template_meta.plate_type in ["ONE_LINE", "TWO_LINE", "TWO_LINE_SMALL", "MOTORCYCLE"]
        assert 0.0 <= template_meta.corners_confidence <= 1.0

    def test_quality_analysis(self):
        """품질 분석 테스트"""
        # 테스트 이미지
        test_image = self.create_test_image()

        # 품질 분석
        quality = analyze_quality(test_image)

        # 검증
        assert "blur" in quality
        assert "contrast" in quality
        assert "noise" in quality
        assert quality["blur"] > 0
        assert quality["contrast"] >= 0
        assert quality["noise"] >= 0

    def test_adaptive_preprocessing(self):
        """적응형 전처리 테스트"""
        from src.preprocessing.warp import get_one_line_template

        test_image = self.create_test_image()
        template_meta = get_one_line_template()

        # 전처리 파이프라인 적용
        processed, meta = preprocess_plate_image(test_image, template_meta)

        # 검증
        assert processed is not None
        assert "quality" in meta
        assert "pipeline" in meta
        assert "template" in meta
        assert meta["pipeline"] in ["default", "low_contrast", "blurry", "noisy", "multi_issue"]
        assert meta["template"] == "ONE_LINE"

    def test_preprocessing_pipeline_selection(self):
        """전처리 파이프라인 선택 로직 테스트"""
        from src.preprocessing.pipelines import select_pipeline

        # 정상 품질
        quality_good = {"blur": 150.0, "contrast": 40.0, "noise": 20.0}
        pipeline = select_pipeline(quality_good)
        assert pipeline == "default"

        # 흐림
        quality_blurry = {"blur": 80.0, "contrast": 40.0, "noise": 20.0}
        pipeline = select_pipeline(quality_blurry)
        assert pipeline == "blurry"

        # 저대비
        quality_low_contrast = {"blur": 150.0, "contrast": 15.0, "noise": 20.0}
        pipeline = select_pipeline(quality_low_contrast)
        assert pipeline == "low_contrast"

        # 노이즈
        quality_noisy = {"blur": 150.0, "contrast": 40.0, "noise": 50.0}
        pipeline = select_pipeline(quality_noisy)
        assert pipeline == "noisy"

        # 복합 문제
        quality_multi = {"blur": 80.0, "contrast": 15.0, "noise": 50.0}
        pipeline = select_pipeline(quality_multi)
        assert pipeline == "multi_issue"

    def test_postprocessor_validation(self):
        """후처리기 검증 테스트"""
        # ONE_LINE 패턴 (12가3456)
        text, is_valid = self.postprocessor.validate_by_template("12가3456", "ONE_LINE")
        assert is_valid is True
        assert text == "12가3456"

        # 잘못된 패턴
        text, is_valid = self.postprocessor.validate_by_template("12abc3456", "ONE_LINE")
        assert is_valid is False

        # TWO_LINE 패턴 (경기79사4711)
        text, is_valid = self.postprocessor.validate_by_template("경기79사4711", "TWO_LINE")
        assert is_valid is True

    @pytest.mark.skipif(not SLOT_CLASSIFIER_AVAILABLE, reason="SlotClassifier 사용 불가")
    def test_end_to_end_pipeline(self):
        """전체 파이프라인 End-to-End 테스트"""
        # 실제 번호판 이미지가 있다면 테스트
        # 여기서는 더미 이미지로 파이프라인 흐름만 확인

        test_image = self.create_test_image(520, 110)

        # 1. 워프
        warped, template_meta = process_plate_warp(test_image)
        assert warped is not None

        # 2. 전처리
        processed, meta = preprocess_plate_image(warped, template_meta)
        assert processed is not None

        # 3. 슬롯 인식 (더미이므로 실패 예상)
        text, chars, probs = recognize_plate_slots(
            processed,
            template_meta,
            self.slot_classifier
        )

        # 4. 검증 (더미이므로 검증 실패 예상)
        validated_text, is_valid = self.postprocessor.validate_by_template(
            text,
            template_meta.plate_type
        )

        # 파이프라인 자체는 동작해야 함
        assert validated_text is not None
        assert isinstance(chars, list)
        assert isinstance(probs, list)


@pytest.mark.skipif(not SLOT_CLASSIFIER_AVAILABLE, reason="SlotClassifier 사용 불가")
class TestSlotClassifier:
    """슬롯 분류기 개별 테스트"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """테스트 설정"""
        if SLOT_CLASSIFIER_AVAILABLE:
            self.classifier = SlotClassifier(use_easyocr=True)
        else:
            self.classifier = None

    def test_classifier_initialization(self):
        """분류기 초기화 테스트"""
        assert self.classifier is not None
        assert self.classifier.use_easyocr is True

    def test_slot_extraction(self):
        """슬롯 추출 테스트"""
        from src.preprocessing.warp import get_one_line_template
        from src.ocr.slot_classifier import extract_slots

        # 테스트 이미지
        template = get_one_line_template()
        test_image = np.ones((template.size[1], template.size[0]), dtype=np.uint8) * 255

        # 슬롯 추출
        slots = extract_slots(test_image, template.slots)

        # 검증
        assert len(slots) == len(template.slots)
        for slot_img in slots:
            assert slot_img is not None
            assert slot_img.ndim == 2  # 그레이스케일


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
