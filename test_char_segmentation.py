"""
Character Segmentation 모듈 테스트 스크립트

Contour 기반 문자 영역 추출 기능을 테스트하고 시각화합니다.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from src.preprocessing.character_segmentation import CharacterSegmentation
from src.preprocessing.image_processor import ImageProcessor
import config

def test_single_image(image_path, plate_type='general'):
    """
    단일 이미지에 대해 Character Segmentation 테스트

    Args:
        image_path: 이미지 경로
        plate_type: 번호판 타입
    """
    print(f"\n{'='*60}")
    print(f"테스트 이미지: {image_path}")
    print(f"번호판 타입: {plate_type}")
    print(f"{'='*60}")

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 불러올 수 없습니다: {image_path}")
        return

    # CharacterSegmentation 인스턴스 생성
    segmenter = CharacterSegmentation(plate_type=plate_type)

    # 문자 영역 추출
    result = segmenter.extract_character_regions(image)

    # 결과 출력
    print(f"\n📊 추출 결과:")
    print(f"  - 성공 여부: {'✅ 성공' if result['success'] else '❌ 실패'}")
    if result['success']:
        print(f"  - 검출된 문자 수: {result['num_characters']}개")
        print(f"  - ROI 좌표: {result.get('roi_coords', 'N/A')}")
        print(f"  - 문자 박스 수: {len(result['character_boxes'])}개")
    else:
        print(f"  - 오류: {result.get('error', '알 수 없음')}")

    # 시각화 단계별 이미지
    steps = segmenter.visualize_segmentation(image)

    # 결과 저장
    output_dir = Path(config.DEBUG_DIR) / "char_segmentation_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem

    for step_name, step_image in steps.items():
        if step_image is not None:
            output_path = output_dir / f"{image_name}_{step_name}.jpg"
            cv2.imwrite(str(output_path), step_image)
            print(f"  💾 저장: {output_path}")

    print(f"\n✅ 테스트 완료!")
    return result


def test_adaptive_processing(image_path):
    """
    적응형 전처리 테스트 (품질 기반 자동 선택)

    Args:
        image_path: 이미지 경로
    """
    print(f"\n{'='*60}")
    print(f"적응형 전처리 테스트: {image_path}")
    print(f"{'='*60}")

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 불러올 수 없습니다: {image_path}")
        return

    # ImageProcessor 생성
    processor = ImageProcessor()

    # 적응형 전처리 실행
    result = processor.process_adaptive(image, plate_type='general')

    # 결과 출력
    print(f"\n📊 적응형 전처리 결과:")
    print(f"  - 선택된 방법: {result['method']}")

    if result['quality_analysis']:
        qa = result['quality_analysis']
        print(f"\n  🔍 이미지 품질 분석:")
        print(f"    - 블러 측정값: {qa['blur_measure']:.2f}")
        print(f"    - 대비: {qa['contrast']:.2f}")
        print(f"    - 노이즈 레벨: {qa['noise_level']:.2f}")
        print(f"    - 배경 복잡도: {qa['background_complexity']:.4f}")

        # 임계값과 비교
        thresholds = config.CHAR_SEGMENTATION_THRESHOLDS
        print(f"\n  📏 임계값 비교:")
        print(f"    - 블러: {qa['blur_measure']:.2f} {'<' if qa['blur_measure'] < thresholds['blur_threshold'] else '>='} {thresholds['blur_threshold']} (임계값)")
        print(f"    - 배경 복잡도: {qa['background_complexity']:.4f} {'>' if qa['background_complexity'] > thresholds['complex_background_threshold'] else '<='} {thresholds['complex_background_threshold']} (임계값)")
        print(f"    - 노이즈: {qa['noise_level']:.2f} {'>' if qa['noise_level'] > thresholds['noise_threshold'] else '<='} {thresholds['noise_threshold']} (임계값)")

    # 결과 저장
    output_dir = Path(config.DEBUG_DIR) / "adaptive_processing_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem
    output_path = output_dir / f"{image_name}_adaptive_{result['method']}.jpg"

    # 결과 이미지를 3채널로 변환하여 저장
    processed_img = result['processed_image']
    if len(processed_img.shape) == 2:
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(str(output_path), processed_img)
    print(f"\n  💾 저장: {output_path}")

    print(f"\n✅ 적응형 전처리 테스트 완료!")
    return result


def compare_methods(image_path):
    """
    기존 방식 vs Contour 방식 비교

    Args:
        image_path: 이미지 경로
    """
    print(f"\n{'='*60}")
    print(f"전처리 방법 비교: {image_path}")
    print(f"{'='*60}")

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 불러올 수 없습니다: {image_path}")
        return

    processor = ImageProcessor()

    # 1. 기존 방식
    print(f"\n1️⃣  기존 방식 (process):")
    standard_result = processor.process(image)

    # 2. Contour 방식
    print(f"2️⃣  Contour 방식 (process_with_char_segmentation):")
    contour_result = processor.process_with_char_segmentation(image, plate_type='general')

    # 결과 저장
    output_dir = Path(config.DEBUG_DIR) / "method_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem

    # 원본
    cv2.imwrite(str(output_dir / f"{image_name}_0_original.jpg"), image)

    # 기존 방식
    standard_bgr = cv2.cvtColor(standard_result, cv2.COLOR_GRAY2BGR) if len(standard_result.shape) == 2 else standard_result
    cv2.imwrite(str(output_dir / f"{image_name}_1_standard.jpg"), standard_bgr)

    # Contour 방식
    contour_bgr = cv2.cvtColor(contour_result, cv2.COLOR_GRAY2BGR) if len(contour_result.shape) == 2 else contour_result
    cv2.imwrite(str(output_dir / f"{image_name}_2_contour.jpg"), contour_bgr)

    print(f"\n  💾 결과 저장: {output_dir}")
    print(f"\n✅ 비교 테스트 완료!")


def main():
    """메인 테스트 함수"""
    print("\n" + "="*60)
    print("Character Segmentation 테스트 시작")
    print("="*60)

    # 테스트할 이미지 경로 (실제 프로젝트의 샘플 이미지 경로로 수정하세요)
    test_images = [
        # 예시 경로 - 실제 이미지 경로로 변경 필요
        "data/raw/sample_plate_1.jpg",
        "data/raw/sample_plate_2.jpg",
        "data/raw/sample_plate_3.jpg",
    ]

    # 테스트 1: 단일 이미지 Character Segmentation
    print("\n" + "="*60)
    print("TEST 1: Character Segmentation (단일 이미지)")
    print("="*60)

    for img_path in test_images:
        if os.path.exists(img_path):
            test_single_image(img_path, plate_type='general')
        else:
            print(f"⚠️  이미지를 찾을 수 없습니다: {img_path}")

    # 테스트 2: 적응형 전처리
    print("\n" + "="*60)
    print("TEST 2: 적응형 전처리 (품질 기반 자동 선택)")
    print("="*60)

    for img_path in test_images:
        if os.path.exists(img_path):
            test_adaptive_processing(img_path)

    # 테스트 3: 방법 비교
    print("\n" + "="*60)
    print("TEST 3: 전처리 방법 비교 (기존 vs Contour)")
    print("="*60)

    for img_path in test_images:
        if os.path.exists(img_path):
            compare_methods(img_path)

    print("\n" + "="*60)
    print("모든 테스트 완료!")
    print("="*60)
    print(f"\n결과 확인: {config.DEBUG_DIR}")


if __name__ == "__main__":
    main()
