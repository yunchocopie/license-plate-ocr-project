"""
번호판 OCR 리그레션 테스트

1차 작업(Task 1A~1D) 완료 후 성능 검증을 위한 테스트
각 번호판 유형별로 검출 수 및 OCR 문자열을 비교
"""

import os
import json
import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.vehicle_detector import VehicleDetector
from src.detection.plate_detector import PlateDetector
from src.preprocessing.image_processor import ImageProcessor
from src.ocr.ocr_engine import OCREngine
import config


class TestPlateRegression:
    """번호판 OCR 리그레션 테스트"""

    @classmethod
    def setup_class(cls):
        """테스트 클래스 초기화"""
        cls.vehicle_detector = VehicleDetector()
        cls.plate_detector = PlateDetector()
        cls.image_processor = ImageProcessor()
        cls.ocr_engine = OCREngine()

        cls.test_data_dir = Path(config.DATA_DIR) / "test"
        cls.plate_types = [
            "general",      # 일반 자가용
            "commercial",   # 영업용
            "electric",     # 전기차
            "diplomatic",   # 외교관용
            "military",     # 군용
            "construction", # 건설기계
            "motorcycle",   # 이륜차
            "temporary",    # 임시운행
            "special"       # 특수용도
        ]

    def load_ground_truth(self, plate_type):
        """GT(Ground Truth) 데이터 로드"""
        gt_file = self.test_data_dir / plate_type / "labels.json"

        if not gt_file.exists():
            pytest.skip(f"GT 파일이 존재하지 않음: {gt_file}")

        with open(gt_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_test_images(self, plate_type):
        """테스트 이미지 목록 가져오기"""
        type_dir = self.test_data_dir / plate_type

        if not type_dir.exists():
            pytest.skip(f"테스트 디렉토리가 존재하지 않음: {type_dir}")

        image_files = list(type_dir.glob("*.png")) + list(type_dir.glob("*.jpg"))

        if not image_files:
            pytest.skip(f"테스트 이미지가 없음: {type_dir}")

        return image_files

    def process_image(self, image_path):
        """이미지 처리 파이프라인 실행"""
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        results = []

        # 1. 차량 검출
        vehicle_boxes = self.vehicle_detector.detect(image)

        # 차량이 없으면 전체 이미지를 차량으로 간주
        if not vehicle_boxes:
            h, w = image.shape[:2]
            vehicle_boxes = [[0, 0, w, h]]

        # 2. 각 차량에 대해 번호판 검출 및 OCR
        for vehicle_box in vehicle_boxes:
            x1, y1, x2, y2 = vehicle_box
            vehicle_image = image[y1:y2, x1:x2]

            # 번호판 검출
            plate_boxes = self.plate_detector.detect(vehicle_image)

            for plate_box in plate_boxes:
                px1, py1, px2, py2 = plate_box
                plate_image = vehicle_image[py1:py2, px1:px2]

                # 전처리
                processed_plate = self.image_processor.process(plate_image)

                # OCR
                plate_text, confidence = self.ocr_engine.recognize_with_confidence(
                    processed_plate,
                    min_confidence=0.1
                )

                # 후처리
                formatted_text = self.ocr_engine.post_processor.format_korean_license_plate(plate_text)

                results.append({
                    "text": formatted_text,
                    "confidence": confidence,
                    "vehicle_box": vehicle_box,
                    "plate_box": plate_box
                })

        return results

    @pytest.mark.parametrize("plate_type", [
        "general", "commercial", "electric", "diplomatic",
        "military", "construction", "motorcycle", "temporary", "special"
    ])
    def test_detection_count(self, plate_type):
        """검출 수 테스트"""
        ground_truth = self.load_ground_truth(plate_type)
        image_files = self.get_test_images(plate_type)

        total_gt_plates = 0
        total_detected_plates = 0

        for image_file in image_files:
            filename = image_file.name

            # GT 데이터 가져오기
            if filename not in ground_truth:
                continue

            gt_data = ground_truth[filename]
            gt_count = gt_data.get("plate_count", 1)
            total_gt_plates += gt_count

            # 실제 검출
            results = self.process_image(image_file)
            if results:
                detected_count = len(results)
                total_detected_plates += detected_count

        # 검출 수가 GT와 비슷한지 확인 (±30% 허용)
        if total_gt_plates > 0:
            detection_rate = total_detected_plates / total_gt_plates
            assert 0.7 <= detection_rate <= 1.3, \
                f"{plate_type}: 검출 수 차이가 큼 (GT: {total_gt_plates}, 검출: {total_detected_plates})"

    @pytest.mark.parametrize("plate_type", [
        "general", "commercial", "electric", "diplomatic",
        "military", "construction", "motorcycle", "temporary", "special"
    ])
    def test_ocr_accuracy(self, plate_type):
        """OCR 정확도 테스트"""
        ground_truth = self.load_ground_truth(plate_type)
        image_files = self.get_test_images(plate_type)

        correct = 0
        total = 0

        for image_file in image_files:
            filename = image_file.name

            # GT 데이터 가져오기
            if filename not in ground_truth:
                continue

            gt_data = ground_truth[filename]
            gt_text = gt_data.get("text", "")

            # 실제 OCR
            results = self.process_image(image_file)

            if results:
                # 첫 번째 결과 사용 (신뢰도가 가장 높을 것으로 가정)
                detected_text = results[0]["text"]

                # 정확도 계산 (완전 일치)
                if detected_text == gt_text:
                    correct += 1
                total += 1

        # 정확도 확인 (최소 50% 이상)
        if total > 0:
            accuracy = correct / total
            print(f"\n{plate_type} OCR 정확도: {accuracy:.2%} ({correct}/{total})")
            assert accuracy >= 0.5, \
                f"{plate_type}: OCR 정확도가 낮음 (정확도: {accuracy:.2%})"

    def test_overall_performance(self):
        """전체 성능 요약 테스트"""
        summary = {
            "total_images": 0,
            "total_detections": 0,
            "total_correct_ocr": 0,
            "by_type": {}
        }

        for plate_type in self.plate_types:
            try:
                ground_truth = self.load_ground_truth(plate_type)
                image_files = self.get_test_images(plate_type)

                type_summary = {
                    "images": len(image_files),
                    "detections": 0,
                    "correct_ocr": 0
                }

                for image_file in image_files:
                    filename = image_file.name

                    if filename not in ground_truth:
                        continue

                    gt_data = ground_truth[filename]
                    gt_text = gt_data.get("text", "")

                    results = self.process_image(image_file)

                    if results:
                        type_summary["detections"] += len(results)

                        for result in results:
                            if result["text"] == gt_text:
                                type_summary["correct_ocr"] += 1
                                break

                summary["by_type"][plate_type] = type_summary
                summary["total_images"] += type_summary["images"]
                summary["total_detections"] += type_summary["detections"]
                summary["total_correct_ocr"] += type_summary["correct_ocr"]

            except Exception as e:
                print(f"\n{plate_type} 처리 중 오류: {e}")
                continue

        # 결과 출력
        print("\n" + "="*50)
        print("전체 성능 요약")
        print("="*50)
        print(f"총 이미지 수: {summary['total_images']}")
        print(f"총 검출 수: {summary['total_detections']}")
        print(f"정확한 OCR: {summary['total_correct_ocr']}")

        if summary["total_images"] > 0:
            overall_accuracy = summary["total_correct_ocr"] / summary["total_images"]
            print(f"전체 정확도: {overall_accuracy:.2%}")

        print("\n유형별 상세:")
        for plate_type, type_summary in summary["by_type"].items():
            if type_summary["images"] > 0:
                accuracy = type_summary["correct_ocr"] / type_summary["images"]
                print(f"  {plate_type}: {accuracy:.2%} "
                      f"({type_summary['correct_ocr']}/{type_summary['images']})")
        print("="*50)


if __name__ == "__main__":
    # pytest 실행
    pytest.main([__file__, "-v", "-s"])
